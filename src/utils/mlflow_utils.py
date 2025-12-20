# src/utils/mlflow_utils.py
import os
import mlflow

def setup_mlflow(experiment_name: str, tracking_uri: str | None = None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def start_run(run_name: str | None = None, tags: dict | None = None):
    run = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    return run

def log_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)

def log_metrics(metrics: dict, step: int | None = None):
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v), step=step if step is not None else 0)

def log_artifact(path: str):
    if os.path.exists(path):
        mlflow.log_artifact(path)

def log_artifacts(dir_path: str):
    if os.path.isdir(dir_path):
        mlflow.log_artifacts(dir_path)

def end_run():
    mlflow.end_run()
