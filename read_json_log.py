import json
import os

def load_metrics_from_path(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]
    val_dices = {
        "WT": data["val_dices_wt"],
        "TC": data["val_dices_tc"],
        "ET": data["val_dices_et"]
    }
    val_mean_dices = data["val_mean_dices"]
    val_hd95s = data["val_hd95s"]
    
    return train_losses, val_losses, val_dices, val_mean_dices, val_hd95s


def main():
    filepath = "metrics/5.30_SpikingSwinUnet_4layers/fold_0_metrics.json"
    train_losses, val_losses, val_dices, val_mean_dices, val_hd95s = load_metrics_from_path(filepath)

    print(f'train_losses: {[f"{loss:.4f}" for loss in train_losses[-30:-1]]}')
    print(f'val_losses: {[f"{loss:.4f}" for loss in val_losses[-30:-1]]}')
    print(f'dice-WT: {[f"{loss:.4f}" for loss in val_dices["WT"][-30:-1]]}')
    print(f'dice-TC: {[f"{loss:.4f}" for loss in val_dices["TC"][-30:-1]]}')
    print(f'dice-ET: {[f"{loss:.4f}" for loss in val_dices["ET"][-30:-1]]}')
    print(f'dice-Mean: {[f"{loss:.4f}" for loss in val_mean_dices[-30:-1]]}')

if __name__ == "__main__":
    main()