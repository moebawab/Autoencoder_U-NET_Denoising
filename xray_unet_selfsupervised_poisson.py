#pip install torch torchvision opencv-python numpy tqdm pytorch-msssim

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResidualUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = ConvBlock(1, 32)
        self.p1 = nn.MaxPool2d(2)

        self.c2 = ConvBlock(32, 64)
        self.p2 = nn.MaxPool2d(2)

        self.c3 = ConvBlock(64, 128)
        self.p3 = nn.MaxPool2d(2)

        self.b = ConvBlock(128, 256)

        self.u3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c4 = ConvBlock(256 + 128, 128)

        self.u2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c5 = ConvBlock(128 + 64, 64)

        self.u1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c6 = ConvBlock(64 + 32, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.p1(c1))
        c3 = self.c3(self.p2(c2))

        b = self.b(self.p3(c3))

        d3 = self.c4(torch.cat([self.u3(b), c3], dim=1))
        d2 = self.c5(torch.cat([self.u2(d3), c2], dim=1))
        d1 = self.c6(torch.cat([self.u1(d2), c1], dim=1))

        residual = self.out(d1)
        return x + residual

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def add_noise(img):
    # scale image to simulate photon counts
    vals = 255  # controls noise strength (photon level) (50 - very low dose , 255 - mild noise) 
    noisy = np.random.poisson(img * vals) / float(vals)
    return np.clip(noisy, 0, 1).astype(np.float32)

class SelfSupervisedXrayDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []

        for f in os.listdir(root_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                self.files.append(os.path.join(root_dir, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        img = cv2.resize(img, (296, 296))
        img = img.astype(np.float32) / 255.0

        target = img.copy()
        input_img = add_noise(img).astype(np.float32)

        input_img = torch.from_numpy(input_img).unsqueeze(0).float()
        target = torch.from_numpy(target).unsqueeze(0).float()

        return input_img, target

from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResidualUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
l1 = nn.L1Loss()

dataset = SelfSupervisedXrayDataset("baseline")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for epoch in range(1, 2): #up to 51
    model.train()
    epoch_loss = 0

    for x, y in tqdm(loader, desc=f"Epoch {epoch}"):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = l1(pred, y) + (1 - ssim(pred, y, data_range=1.0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch} | Loss: {epoch_loss / len(loader):.4f}")

torch.save(model.state_dict(), "residual_unet_selfsupervised_poisson.pth")

import os

model.load_state_dict(torch.load("residual_unet_selfsupervised_poisson.pth"))
model.eval()

input_root = "baseline"
output_root = "poisson"

os.makedirs(output_root, exist_ok=True)

for fname in os.listdir(input_root):

    if not fname.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    ):
        continue

    path = os.path.join(input_root, fname)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Skipping unreadable file: {path}")
        continue

    img = cv2.resize(img, (296, 296))
    img = img.astype(np.float32) / 255.0

    x = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x).cpu().numpy()[0, 0]

    out = np.clip(out * 255, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_root, fname), out)

print("All images processed and saved.")