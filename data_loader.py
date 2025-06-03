import os
from monai.data import DataLoader
from dataset import BraTSDataset
from sklearn.model_selection import KFold
from config import config as cfg


def worker_init_fn(worker_id):
    print(f"[Worker {worker_id}] initialized in PID {os.getpid()}")

def build_data_dicts(case_dirs):
    """
    根据 case_dirs 构造 MONAI 所需的 dict 格式
    """
    data_dicts = []
    modalities = cfg.modalities
    sep = cfg.modality_separator
    suffix = cfg.image_suffix
    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir.rstrip('/'))
        image_paths = [os.path.join(case_dir, f"{case_name}{sep}{m}{suffix}") for m in modalities]
        label_path = os.path.join(case_dir, f"{case_name}{sep}seg{suffix}")
        data_dicts.append({"image": image_paths, "label": label_path})
    return data_dicts

def get_data_loaders(train_dirs, val_dirs, patch_size, batch_size, T, encode_method, num_workers):
    train_data_dicts = build_data_dicts(train_dirs)
    val_data_dicts = build_data_dicts(val_dirs)

    train_dataset = BraTSDataset(train_data_dicts, patch_size=patch_size, T=T, mode="train",encode_method=encode_method)
    val_dataset = BraTSDataset(val_data_dicts, patch_size=patch_size, T=T, mode="val",encode_method=encode_method)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader



def main():
    from config import config as cfg
    print("TEST START")
    case_dirs = [os.path.join(cfg.root_dir, d) for d in os.listdir(cfg.root_dir) if os.path.isdir(os.path.join(cfg.root_dir, d))]
    
    
    kf = KFold(n_splits=cfg.k_folds, shuffle=True)

    # 开始交叉验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(case_dirs)):

        # 根据交叉验证划分数据集
        train_case_dirs = [case_dirs[i] for i in train_idx]
        val_case_dirs = [case_dirs[i] for i in val_idx]

        # 训练和验证数据加载器
        train_loader, val_loader = get_data_loaders(train_case_dirs, val_case_dirs, cfg.batch_size, cfg.T, cfg.num_workers)
        

        print("Iterate one batch from val_loader:")
        for x, y in val_loader:
            print(f"val batch x shape: {x.shape}, x min/max: {x.min().item()}/{x.max().item()}")
            print(f"val batch y shape: {y.shape}, y min/max: {y.min().item()}/{y.max().item()}")
            break
    
    
if __name__ == "__main__":
    main()