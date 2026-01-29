import os
import sys
import cv2
import torch
import numpy as np
sys.path.append(r"C:\PROJECTS\modified-edsr")
from model.edsr_dual import EDSRDual

# =========================
# PATHS
# =========================
HR_DIR = r"C:\PROJECTS\modified-edsr\DIV2K\DIV2K_train_HR\DIV2K_train_HR"
MODEL_PATH = r"C:\PROJECTS\modified-edsr\edsr_dual.pth"

OUT_DIR = r"C:\PROJECTS\modified-edsr\visual_check"
LR1_OUT = os.path.join(OUT_DIR, "LR1")
LR2_OUT = os.path.join(OUT_DIR, "LR2")
SR_OUT  = os.path.join(OUT_DIR, "SR")
HR_OUT  = os.path.join(OUT_DIR, "HR")

for d in [LR1_OUT, LR2_OUT, SR_OUT, HR_OUT]:
    os.makedirs(d, exist_ok=True)

# =========================
# PARAMETERS
# =========================
SCALE = 4
CROP_SIZE = 192
SHIFT = 1
NUM_IMAGES = 5

# =========================
# LOAD MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EDSRDual(scale=SCALE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =========================
# HELPERS
# =========================
def bicubic_downsample(img, scale):
    h, w, _ = img.shape
    return cv2.resize(
        img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC
    )

def save_img(path, img):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# =========================
# VISUAL CHECK LOOP
# =========================
images = sorted(os.listdir(HR_DIR))[:NUM_IMAGES]

with torch.no_grad():
    for name in images:
        hr_path = os.path.join(HR_DIR, name)

        hr = cv2.imread(hr_path)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        hr = hr.astype(np.float32) / 255.0

        H, W, _ = hr.shape
        x, y = 0, 0  # fixed crop for consistency

        crop1 = hr[y:y+CROP_SIZE, x:x+CROP_SIZE]
        crop2 = hr[y+SHIFT:y+SHIFT+CROP_SIZE,
                   x+SHIFT:x+SHIFT+CROP_SIZE]

        lr1 = bicubic_downsample(crop1, SCALE)
        lr2 = bicubic_downsample(crop2, SCALE)

        # Convert to tensor
        lr1_t = torch.from_numpy(lr1).permute(2, 0, 1).unsqueeze(0).to(device)
        lr2_t = torch.from_numpy(lr2).permute(2, 0, 1).unsqueeze(0).to(device)

        # Inference
        sr = model(lr1_t, lr2_t).squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Save images
        save_img(os.path.join(LR1_OUT, name), lr1)
        save_img(os.path.join(LR2_OUT, name), lr2)
        save_img(os.path.join(SR_OUT,  name), sr)
        save_img(os.path.join(HR_OUT,  name), crop1)

print("✅ Visual sanity check images saved.")