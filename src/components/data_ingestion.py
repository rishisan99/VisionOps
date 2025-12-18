# src/components/data_ingestion.py

import os
import sys
import json
import shutil
import random
from typing import Dict, List, Tuple
from collections import defaultdict

from PIL import Image

from src.logging.logger import logging
from src.exception.exception import CustomException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.config = data_ingestion_config

    def _list_images_in_class_dir(self, class_dir: str) -> List[str]:
        files = []
        if not os.path.isdir(class_dir):
            return files

        for root, _, filenames in os.walk(class_dir):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in self.config.allowed_ext:
                    files.append(os.path.join(root, fn))
        return files

    def _is_valid_image(self, file_path: str) -> Tuple[bool, Tuple[int, int] | None, str | None]:
        """
        Returns:
            (is_valid, (width, height) if valid else None, error_msg if invalid else None)
        """
        try:
            with Image.open(file_path) as img:
                img.verify()  # quick integrity check

            # re-open to safely read size after verify()
            with Image.open(file_path) as img2:
                w, h = img2.size
            return True, (w, h), None

        except Exception as e:
            return False, None, str(e)

    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _scan_source(self) -> Dict[str, List[str]]:
        """
        Scan only the target classes from source_root_dir.
        """
        source_root = self.config.source_root_dir
        class_targets = self.config.class_targets

        logging.info(f"Scanning source root: {source_root}")
        class_to_files: Dict[str, List[str]] = {}

        for class_name in class_targets.keys():
            class_dir = os.path.join(source_root, class_name)
            files = self._list_images_in_class_dir(class_dir)
            class_to_files[class_name] = files
            logging.info(f"Found {len(files)} files in source class '{class_name}'")

        return class_to_files

    def _select_files(self, class_to_files: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Deterministically select exactly N files per class (seeded).
        """
        rng = random.Random(self.config.seed)
        selected: Dict[str, List[str]] = {}

        for class_name, files in class_to_files.items():
            target_n = self.config.class_targets[class_name]
            if len(files) < target_n:
                raise ValueError(
                    f"Not enough images for class '{class_name}'. "
                    f"Required={target_n}, Found={len(files)}. "
                    f"Add more images or reduce target."
                )

            # stable selection
            files_sorted = sorted(files)
            rng.shuffle(files_sorted)
            chosen = files_sorted[:target_n]
            selected[class_name] = chosen

            logging.info(f"Selected {len(chosen)} images for class '{class_name}' (target={target_n})")

        return selected

    def _copy_selected(self, selected: Dict[str, List[str]]) -> tuple[dict, dict, dict, dict]:
        """
        Validate + copy images to raw/raw_bad, and collect stats.
        Returns:
            per_class_copied, per_class_corrupted, resolution_stats, corrupted_details
        """
        per_class_copied = defaultdict(int)
        per_class_corrupted = defaultdict(int)

        # resolution stats per class
        # store list of (w,h) for aggregate stats
        resolutions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        corrupted_details: Dict[str, List[dict]] = defaultdict(list)

        for class_name, paths in selected.items():
            out_class_dir = os.path.join(self.config.raw_data_dir, class_name)
            bad_class_dir = os.path.join(self.config.raw_bad_data_dir, class_name)
            os.makedirs(out_class_dir, exist_ok=True)
            os.makedirs(bad_class_dir, exist_ok=True)

            for src_path in paths:
                valid, size, err = self._is_valid_image(src_path)
                base_name = os.path.basename(src_path)

                if valid:
                    dst_path = os.path.join(out_class_dir, base_name)
                    shutil.copy2(src_path, dst_path)
                    per_class_copied[class_name] += 1
                    if size:
                        resolutions[class_name].append(size)
                else:
                    dst_bad = os.path.join(bad_class_dir, base_name)
                    try:
                        shutil.copy2(src_path, dst_bad)
                    except Exception:
                        # if copy also fails, still record it
                        pass

                    per_class_corrupted[class_name] += 1
                    corrupted_details[class_name].append(
                        {"file": src_path, "error": err}
                    )

        return per_class_copied, per_class_corrupted, resolutions, corrupted_details

    def _compute_resolution_stats(self, resolutions: Dict[str, List[Tuple[int, int]]]) -> dict:
        """
        Compute min/max/mean for widths/heights per class and overall.
        """
        def agg(stats_list: List[int]) -> dict:
            if not stats_list:
                return {"min": None, "max": None, "mean": None}
            return {
                "min": min(stats_list),
                "max": max(stats_list),
                "mean": sum(stats_list) / len(stats_list),
            }

        per_class = {}
        all_w, all_h = [], []

        for class_name, sizes in resolutions.items():
            ws = [w for (w, _) in sizes]
            hs = [h for (_, h) in sizes]
            all_w.extend(ws)
            all_h.extend(hs)
            per_class[class_name] = {
                "count": len(sizes),
                "width": agg(ws),
                "height": agg(hs),
            }

        overall = {
            "count": len(all_w),
            "width": agg(all_w),
            "height": agg(all_h),
        }

        return {"per_class": per_class, "overall": overall}

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            # 1) scan
            class_to_files = self._scan_source()

            source_scan_payload = {
                "source_root_dir": self.config.source_root_dir,
                "allowed_ext": self.config.allowed_ext,
                "per_class_counts": {k: len(v) for k, v in class_to_files.items()},
            }
            self._write_json(self.config.source_scan_file_path, source_scan_payload)

            # 2) class map (identity map here, but we still write it for traceability)
            class_map_payload = {
                "mapping": {k: k for k in self.config.class_targets.keys()},
                "note": "Source folder names already match project labels.",
            }
            self._write_json(self.config.class_map_file_path, class_map_payload)

            # 3) deterministic select
            selected = self._select_files(class_to_files)
            selected_payload = {
                "seed": self.config.seed,
                "targets": self.config.class_targets,
                "selected_files": selected,
            }
            self._write_json(self.config.selected_files_file_path, selected_payload)

            total_selected = sum(len(v) for v in selected.values())

            # 4) validate + copy
            per_class_copied, per_class_corrupted, resolutions, corrupted_details = self._copy_selected(selected)

            total_copied = sum(per_class_copied.values())
            total_corrupted = sum(per_class_corrupted.values())

            corrupt_rate = (total_corrupted / max(total_selected, 1))

            # 5) fail-fast check
            if corrupt_rate > self.config.max_corrupt_rate:
                raise ValueError(
                    f"Corruption rate too high: {corrupt_rate:.4f} "
                    f"(max allowed {self.config.max_corrupt_rate})."
                )

            # 6) write ingest report
            ingest_report_payload = {
                "source_root_dir": self.config.source_root_dir,
                "raw_output_dir": self.config.raw_data_dir,
                "raw_bad_output_dir": self.config.raw_bad_data_dir,
                "seed": self.config.seed,
                "targets": self.config.class_targets,
                "total_selected": total_selected,
                "total_copied": total_copied,
                "total_corrupted": total_corrupted,
                "corrupt_rate": corrupt_rate,
                "per_class_selected": {k: len(v) for k, v in selected.items()},
                "per_class_copied": dict(per_class_copied),
                "per_class_corrupted": dict(per_class_corrupted),
                "corrupted_details_sample": {
                    k: corrupted_details[k][:5] for k in corrupted_details.keys()
                },
            }
            self._write_json(self.config.ingest_report_file_path, ingest_report_payload)

            # 7) dataset stats (raw subset)
            resolution_stats = self._compute_resolution_stats(resolutions)
            dataset_stats_payload = {
                "total_images": total_copied,
                "per_class_counts": dict(per_class_copied),
                "resolution_stats": resolution_stats,
            }
            self._write_json(self.config.dataset_stats_raw_file_path, dataset_stats_payload)

            logging.info("Data ingestion completed successfully.")

            return DataIngestionArtifact(
                raw_data_dir=self.config.raw_data_dir,
                raw_bad_data_dir=self.config.raw_bad_data_dir,
                metadata_dir=self.config.metadata_dir,
                class_map_file_path=self.config.class_map_file_path,
                source_scan_file_path=self.config.source_scan_file_path,
                selected_files_file_path=self.config.selected_files_file_path,
                ingest_report_file_path=self.config.ingest_report_file_path,
                dataset_stats_raw_file_path=self.config.dataset_stats_raw_file_path,
                total_selected=total_selected,
                total_copied=total_copied,
                total_corrupted=total_corrupted,
                per_class_selected={k: len(v) for k, v in selected.items()},
                per_class_copied=dict(per_class_copied),
                per_class_corrupted=dict(per_class_corrupted),
            )

        except Exception as e:
            raise CustomException(e, sys)
