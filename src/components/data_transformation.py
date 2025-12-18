# src/components/data_transformation.py

import os
import sys
import json
from typing import Dict, List, Tuple
from collections import defaultdict

from PIL import Image

from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig):
        self.ingestion_artifact = data_ingestion_artifact
        self.config = data_transformation_config

    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _list_class_folders(self, raw_dir: str) -> List[str]:
        if not os.path.isdir(raw_dir):
            return []
        return sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])

    def _list_images(self, class_dir: str) -> List[str]:
        files = []
        for fn in os.listdir(class_dir):
            fp = os.path.join(class_dir, fn)
            if os.path.isfile(fp):
                files.append(fp)
        return sorted(files)

    def _transform_one(self, src_path: str, dst_path: str, target_size: Tuple[int, int]) -> Tuple[bool, str | None]:
        """
        Returns (success, error_message)
        """
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img = img.resize((target_size[1], target_size[0]))  # PIL uses (W,H)

                # ensure output directory exists
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                # Save consistently as JPEG
                # (Even if input is PNG, this normalizes format.)
                img.save(dst_path, format="JPEG", quality=95)

            return True, None

        except Exception as e:
            return False, str(e)

    def _compute_resolution_stats(self, size_list: List[Tuple[int, int]]) -> dict:
        if not size_list:
            return {"count": 0, "width": {"min": None, "max": None, "mean": None}, "height": {"min": None, "max": None, "mean": None}}

        ws = [w for (w, _) in size_list]
        hs = [h for (_, h) in size_list]

        def agg(arr: List[int]) -> dict:
            return {"min": min(arr), "max": max(arr), "mean": sum(arr) / len(arr)}

        return {"count": len(size_list), "width": agg(ws), "height": agg(hs)}

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            raw_dir = self.ingestion_artifact.raw_data_dir
            classes = self._list_class_folders(raw_dir)

            if not classes:
                raise ValueError(f"No class folders found in raw_dir: {raw_dir}")

            target_h, target_w = self.config.target_size
            logging.info(f"Transforming images to RGB + resize={target_h}x{target_w}")

            per_class_processed = defaultdict(int)
            per_class_failed = defaultdict(int)

            total_input = 0
            total_processed = 0
            total_failed = 0

            # we'll confirm processed output resolution as sanity: should be constant
            processed_sizes: List[Tuple[int, int]] = []

            # Preview settings
            preview_per_class = 10  # small but useful for quick inspection

            for class_name in classes:
                class_in_dir = os.path.join(raw_dir, class_name)
                images = self._list_images(class_in_dir)
                total_input += len(images)

                # Output folders
                class_out_all = os.path.join(self.config.processed_all_dir, class_name)
                class_out_preview = os.path.join(self.config.processed_preview_dir, class_name)
                os.makedirs(class_out_all, exist_ok=True)
                os.makedirs(class_out_preview, exist_ok=True)

                # Transform all images
                for i, src_path in enumerate(images):
                    base = os.path.splitext(os.path.basename(src_path))[0]
                    dst_name = f"{base}.jpg"

                    dst_all = os.path.join(class_out_all, dst_name)
                    ok, err = self._transform_one(src_path, dst_all, self.config.target_size)

                    if ok:
                        per_class_processed[class_name] += 1
                        total_processed += 1
                        processed_sizes.append((target_w, target_h))  # constant by design
                    else:
                        per_class_failed[class_name] += 1
                        total_failed += 1

                    # Create preview for first N successful transforms per class
                    if ok and per_class_processed[class_name] <= preview_per_class:
                        dst_prev = os.path.join(class_out_preview, dst_name)
                        # Copy from already transformed output to preview (cheap + consistent)
                        try:
                            # If you prefer to re-transform, you can, but copy is fine
                            with open(dst_all, "rb") as fsrc, open(dst_prev, "wb") as fdst:
                                fdst.write(fsrc.read())
                        except Exception:
                            pass

                logging.info(
                    f"[{class_name}] input={len(images)} processed={per_class_processed[class_name]} failed={per_class_failed[class_name]}"
                )

            # Reports
            transform_report = {
                "input_raw_dir": raw_dir,
                "output_processed_all_dir": self.config.processed_all_dir,
                "output_processed_preview_dir": self.config.processed_preview_dir,
                "target_size": {"height": target_h, "width": target_w},
                "total_input": total_input,
                "total_processed": total_processed,
                "total_failed": total_failed,
                "per_class_processed": dict(per_class_processed),
                "per_class_failed": dict(per_class_failed),
            }
            self._write_json(self.config.transform_report_file_path, transform_report)

            processed_stats = {
                "total_images": total_processed,
                "per_class_counts": dict(per_class_processed),
                "processed_resolution": {"height": target_h, "width": target_w},
                "resolution_stats": {
                    "overall": self._compute_resolution_stats(processed_sizes)
                },
            }
            self._write_json(self.config.dataset_stats_processed_file_path, processed_stats)

            logging.info("Data transformation completed successfully.")

            return DataTransformationArtifact(
                processed_all_dir=self.config.processed_all_dir,
                processed_preview_dir=self.config.processed_preview_dir,
                metadata_dir=self.config.metadata_dir,
                transform_report_file_path=self.config.transform_report_file_path,
                dataset_stats_processed_file_path=self.config.dataset_stats_processed_file_path,
                total_input=total_input,
                total_processed=total_processed,
                total_failed=total_failed,
                per_class_processed=dict(per_class_processed),
                per_class_failed=dict(per_class_failed),
            )

        except Exception as e:
            raise CustomException(e, sys)
