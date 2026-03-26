import mlflow
import os
import sys

def verify_model():
    if not os.path.exists("model_info.txt"):
        sys.exit(1)

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = mlflow.tracking.MlflowClient()
    
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)

    if accuracy >= 0.85:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    verify_model()
