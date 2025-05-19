from torch.utils.data import DataLoader

from dataset import BraTSDataset

def get_data_loaders(train_dirs, val_dirs, batch_size, T, num_workers):
    train_dataset = BraTSDataset(train_dirs, T=T)
    val_dataset = BraTSDataset(val_dirs, T=T)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader