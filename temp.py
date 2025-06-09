#!/usr/bin/env python3

import os

import numpy as np

import nibabel as nib

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.model_selection import train_test_split

from skimage.filters import threshold_otsu

from torch.optim import Adam

from tqdm import tqdm
 
# Optional: restrict to one GPU

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
 
# Check GPU availability

#num_gpus = torch.cuda.device_count()

#print(f"Number of GPUs available: {num_gpus}")

#if num_gpus > 0:

    #for i in range(num_gpus):

     #   print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

#else:

  #  print("No GPUs detected. Running on CPU.")
 
gpu_name = 'cuda'

device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
 
 
class LungDataset(Dataset):

    def __init__(self, ct_paths, mask_paths, shape=(64, 64, 64)):

        self.ct_paths = ct_paths

        self.mask_paths = mask_paths

        self.shape = shape
 
    def __len__(self):

        return len(self.ct_paths)
 
    def __getitem__(self, idx):

        ct = nib.load(self.ct_paths[idx]).get_fdata().astype(np.float32)

        mask = nib.load(self.mask_paths[idx]).get_fdata().astype(np.float32)

        if ct.shape != mask.shape:

            raise ValueError(f"Shape mismatch for {self.ct_paths[idx]}")
 
        positive_voxels = ct[ct > 0]

        if positive_voxels.size < 1000:

            print(f"Skipping CT with too few non-zero voxels: {self.ct_paths[idx]}")

            return self.__getitem__((idx + 1) % len(self))
 
        if np.isclose(positive_voxels.min(), positive_voxels.max()):

            print(f"Skipping CT with constant non-zero voxels: {self.ct_paths[idx]}")

            return self.__getitem__((idx + 1) % len(self))
 
        threshold = threshold_otsu(positive_voxels)

        ct = (ct - positive_voxels.mean()) / (positive_voxels.std() + 1e-6)
 
        cs = [(s - self.shape[i]) // 2 for i, s in enumerate(ct.shape)]

        ce = [cs[i] + self.shape[i] for i in range(3)]

        ct_crop = ct[cs[0]:ce[0], cs[1]:ce[1], cs[2]:ce[2]]

        mask_crop = mask[cs[0]:ce[0], cs[1]:ce[1], cs[2]:ce[2]]
 
        ct_crop = np.expand_dims(ct_crop, 0)

        mask_crop = np.expand_dims((mask_crop > 0.5).astype(np.float32), 0)

        return torch.tensor(ct_crop), torch.tensor(mask_crop)
 
 
class UNet3D(nn.Module):

    def __init__(self):

        super().__init__()

        def block(in_c, out_c):

            return nn.Sequential(

                nn.Conv3d(in_c, out_c, 3, padding=1),

                nn.ReLU(),

                nn.Conv3d(out_c, out_c, 3, padding=1),

                nn.ReLU()

            )
 
        self.enc1 = block(1, 32)

        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = block(32, 64)

        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = block(64, 128)

        self.up1 = nn.Upsample(scale_factor=2)

        self.dec1 = block(128 + 64, 64)

        self.up2 = nn.Upsample(scale_factor=2)

        self.dec2 = block(64 + 32, 32)

        self.out = nn.Conv3d(32, 1, 1)
 
    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))

        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))

        return torch.sigmoid(self.out(d2))
 
 
def dice_loss(pred, target, smooth=1e-5):

    pred = pred.view(-1)

    target = target.view(-1)

    intersection = (pred * target).sum()

    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
 
 
def train(model, loader, optimizer):

    model.train()

    total_loss = 0

    for x, y in tqdm(loader):

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = dice_loss(pred, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
 
 
def evaluate(model, loader):

    model.eval()

    dice_scores = []

    with torch.no_grad():

        for x, y in loader:

            x, y = x.to(device), y.to(device)

            pred = model(x)

            dice_scores.append(1 - dice_loss(pred, y).item())

    return np.mean(dice_scores)
 
 
def main():

    root = "/hpc/gmeh097/19_May_preprocess"

    train_ids = [f"R01_{i:03d}" for i in range(2, 146)]

    ct_paths, mask_paths = [], []
 
    for pid in train_ids:

        ct = os.path.join(root, pid, "final_preproc19May_ct.nii.gz")

        mask = os.path.join(root, pid, "final_preproc19May_tumor_mask.nii.gz")

        if os.path.exists(ct) and os.path.exists(mask):

            ct_paths.append(ct)

            mask_paths.append(mask)
 
    dataset = LungDataset(ct_paths, mask_paths)

    train_size = int(0.8 * len(dataset))

    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=1, num_workers=8, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=8, shuffle=False)
 
    model = UNet3D().to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)
 
    best_dice = 0

    for epoch in range(1, 51):

        print(f"\nEpoch {epoch}")

        train_loss = train(model, train_loader, optimizer)

        val_dice = evaluate(model, val_loader)

        print(f"Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(model.state_dict(), "best_model.pt")
 
    # Test on R01_146

    test_ct = os.path.join(root, "R01_146", "final_preproc19May_ct.nii.gz")

    test_mask = os.path.join(root, "R01_146", "final_preproc19May_tumor_mask.nii.gz")

    if os.path.exists(test_ct) and os.path.exists(test_mask):

        test_ds = LungDataset([test_ct], [test_mask])

        test_loader = DataLoader(test_ds, batch_size=1)

        test_dice = evaluate(model, test_loader)

        print(f"Test Dice Score: {test_dice:.4f}")

    else:

        print("Test case not found.")
 
 
if __name__ == "__main__":

    main()

 