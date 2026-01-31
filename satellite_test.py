import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.div2k_dual_disk import DIV2KDualDisk
from model.edsr_dual import EDSRDual

# ------------------------
# Configuration
# ------------------------
DATASET_ROOT = r"C:\PROJECTS\modified-edsr\DIV2K_DUAL_LR"

BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4
SCALE = 4
NUM_WORKERS = 4

# ------------------------
# Device
# ------------------------
device = torch.device("cuda")

# ------------------------
# Dataset & Loader
# ------------------------
dataset = DIV2KDualDisk(DATASET_ROOT)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ------------------------
# Model
# ------------------------
model = EDSRDual(scale=SCALE).to(device)

# ------------------------
# Loss & Optimizer
# ------------------------
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------------
# Training Loop
# ------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for lr1, lr2, hr in loader:
        lr1 = lr1.to(device, non_blocking=True)
        lr2 = lr2.to(device, non_blocking=True)
        hr  = hr.to(device, non_blocking=True)

        # Forward
        sr = model(lr1, lr2)

        # Loss
        loss = criterion(sr, hr)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | L1 Loss: {avg_loss:.4f}")

# ------------------------
# Save Model
# ------------------------
torch.save(model.state_dict(), "edsr_dual.pth")
