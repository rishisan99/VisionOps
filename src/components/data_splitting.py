# src/components/data_splitting.py

import os
import sys
import csv
import json
import shutil
import random
from typing import Dict, List, Tuple
from collections import defaultdict

from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import DataSplittingConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataSplittingArtifact


class DataSplitting:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, data_splitting_config: DataSplittingConfig):
        self.transformation_artifact = data_transformation_artifact
        self.config = data_splitting_config

    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _list_class_folders(self, processed_all_dir: str) -> List[str]:
        return sorted([d for d in os.listdir(processed_all_dir) if os.path.isdir(os.path.join(processed_all_dir, d))])

    def _list_images(self, class_dir: str) -> List[str]:
        files = []
        for fn in os.listdir(class_dir):
            fp = os.path.join(class_dir, fn)
            if os.path.isfile(fp):
                files.append(fp)
        return sorted(files)

    def _split_class(self, files: List[str], seed: int, train_ratio: float, val_ratio: float) -> Tuple[List[str], List[str], List[str]]:
        """
        Stratified split per class (simple + deterministic):
          - shuffle with seed
          - allocate by ratios
        """
        rng = random.Random(seed)
        files_sorted = sorted(files)
        rng.shuffle(files_sorted)

        n = len(files_sorted)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # rest goes to test to ensure total matches n
        n_test = n - n_train - n_val

        train_files = files_sorted[:n_train]
        val_files = files_sorted[n_train:n_train + n_val]
        test_files = files_sorted[n_train + n_val:]

        # sanity
        assert len(train_files) + len(val_files) + len(test_files) == n
        assert len(test_files) == n_test

        return train_files, val_files, test_files

    def _copy_split_files(self, split_name: str, class_name: str, files: List[str], dest_root: str) -> int:
        """
        Copy files into: dest_root/<split_name>/<class_name>/
        """
        dest_dir = os.path.join(dest_root, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        copied = 0
        for src_path in files:
            dst_path = os.path.join(dest_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            copied += 1
        return copied

    def _write_splits_csv(self, rows: List[Tuple[str, str, str]], out_csv: str) -> None:
        """
        rows: (relative_filepath_from_processed_all, label, split)
        """
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "split"])
            writer.writerows(rows)

    def initiate_data_splitting(self) -> DataSplittingArtifact:
        try:
            processed_all_dir = self.transformation_artifact.processed_all_dir
            if not os.path.isdir(processed_all_dir):
                raise ValueError(f"processed_all_dir not found: {processed_all_dir}")

            classes = self._list_class_folders(processed_all_dir)
            if not classes:
                raise ValueError(f"No class folders found in: {processed_all_dir}")

            logging.info(f"Splitting data from: {processed_all_dir}")
            logging.info(f"Ratios train/val/test = {self.config.train_ratio}/{self.config.val_ratio}/{self.config.test_ratio} | seed={self.config.seed}")

            all_rows: List[Tuple[str, str, str]] = []

            per_class_train = defaultdict(int)
            per_class_val = defaultdict(int)
            per_class_test = defaultdict(int)

            train_count = val_count = test_count = 0
            total_images = 0

            # We'll materialize split data under processed_split_dir
            # We already created train/val/test dirs in config init
            dest_root = self.config.processed_split_dir

            for class_name in classes:
                class_dir = os.path.join(processed_all_dir, class_name)
                files = self._list_images(class_dir)
                total_images += len(files)

                train_files, val_files, test_files = self._split_class(
                    files=files,
                    seed=self.config.seed,
                    train_ratio=self.config.train_ratio,
                    val_ratio=self.config.val_ratio,
                )

                # CSV rows use relative paths from processed_all_dir for portability
                def rel(p: str) -> str:
                    return os.path.relpath(p, processed_all_dir)

                for p in train_files:
                    all_rows.append((rel(p), class_name, "train"))
                for p in val_files:
                    all_rows.append((rel(p), class_name, "val"))
                for p in test_files:
                    all_rows.append((rel(p), class_name, "test"))

                # Copy to materialized folders
                train_c = self._copy_split_files("train", class_name, train_files, dest_root)
                val_c = self._copy_split_files("val", class_name, val_files, dest_root)
                test_c = self._copy_split_files("test", class_name, test_files, dest_root)

                per_class_train[class_name] += train_c
                per_class_val[class_name] += val_c
                per_class_test[class_name] += test_c

                train_count += train_c
                val_count += val_c
                test_count += test_c

                logging.info(f"[{class_name}] total={len(files)} train={train_c} val={val_c} test={test_c}")

            # Write splits.csv
            self._write_splits_csv(all_rows, self.config.splits_csv_path)

            # Write split report
            split_report = {
                "input_processed_all_dir": processed_all_dir,
                "splits_csv_path": self.config.splits_csv_path,
                "materialized_split_dir": self.config.processed_split_dir,
                "ratios": {"train": self.config.train_ratio, "val": self.config.val_ratio, "test": self.config.test_ratio},
                "seed": self.config.seed,
                "total_images": total_images,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
                "per_class_train": dict(per_class_train),
                "per_class_val": dict(per_class_val),
                "per_class_test": dict(per_class_test),
            }
            self._write_json(self.config.split_report_file_path, split_report)

            logging.info("Data splitting completed successfully.")

            return DataSplittingArtifact(
                splits_csv_path=self.config.splits_csv_path,
                split_report_file_path=self.config.split_report_file_path,
                processed_split_dir=self.config.processed_split_dir,
                train_dir=self.config.train_dir,
                val_dir=self.config.val_dir,
                test_dir=self.config.test_dir,
                total_images=total_images,
                train_count=train_count,
                val_count=val_count,
                test_count=test_count,
                per_class_train=dict(per_class_train),
                per_class_val=dict(per_class_val),
                per_class_test=dict(per_class_test),
            )

        except Exception as e:
            raise CustomException(e, sys)
