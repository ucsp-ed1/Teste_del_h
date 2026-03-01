
import pandas as pd
import numpy as np
import sys
import json

def main(pred_path, metadata_path):
    
    metadata_keys = ["team", "run_id", "type", "model"]
    preds = pd.read_csv(pred_path)
    test_ids = np.arange(1, 154)
    
    with open(metadata_path) as f:
        metadata = json.load(f) 

    # Validate the submission file
    if "id" not in preds.columns or "y_pred" not in preds.columns:
        raise ValueError("predictions.csv must contain id and y_pred")

    if preds["id"].duplicated().any():
        raise ValueError("Duplicate IDs found")

    if preds["y_pred"].isna().any():
        raise ValueError("NaN predictions found")

    if ((preds["y_pred"] < 0) | (preds["y_pred"] > 1)).any():
        raise ValueError("Predictions must be in [0,1]")

    if set(preds["id"]) != set(test_ids):
        raise ValueError("Prediction IDs do not match test nodes")
    
    # Validate the metadata file
    if not all(key in metadata for key in metadata_keys):
        raise ValueError("The metadata file does not respect the structure")
    
    if not all(isinstance(v, str) for v in metadata.values()):
        raise ValueError("Some of the provided values are not strings")
    
    if metadata["type"] not in ["human", "llm-only", "human+llm"]:
        raise ValueError("The type should be either human, llm, or both")

    print("VALID SUBMISSION")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
