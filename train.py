import torch
from torch.utils.data import DataLoader
from data.div2k_dual_disk import DIV2KDualDisk
from model.edsr_dual import EDSRDual
import torch.nn as nn
import torch.optim as optim

DATASET_ROOT = r"C:\PROJECTS\modified-edsr\DIV2K_DUAL_LR"

BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4
SCALE = 4

dataset = DIV2KDualDisk(DATASET_ROOT)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda")
model = EDSRDual(scale=SCALE).to(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for lr1, lr2, hr in loader:
        lr1, lr2, hr = lr1.to(device), lr2.to(device), hr.to(device)

        sr = model(lr1, lr2)
        loss = criterion(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss/len(loader):.4f}")

torch.save(model.state_dict(), "edsr_dual.pth")