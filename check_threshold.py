import mlflow
import os

with open("model_info.txt", "r") as f:
    run_id = f.read().replace("RUN_ID=", "").strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0)

print(f"Model Accuracy: {accuracy}")

if accuracy < 0.85:
    print("Accuracy below threshold! Failing pipeline.")
    exit(1)
else:
    print("Accuracy passed! Proceeding to deploy.")
    exit(0)
