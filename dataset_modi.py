import os
import cv2
import random
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================
HR_DIR = r"C:\PROJECTS\modified-edsr\DIV2K\DIV2K_train_HR\DIV2K_train_HR"
OUT_DIR = r"C:\PROJECTS\modified-edsr\DIV2K_DUAL_LR"

SCALE = 4
CROP_SIZE = 192

# Gaussian blur parameters for LR2
BLUR_KERNEL = (5, 5)
BLUR_SIGMA = 1.2

# =========================
# OUTPUT DIRS
# =========================
HR_OUT  = os.path.join(OUT_DIR, "HR")
LR1_OUT = os.path.join(OUT_DIR, "LR1")
LR2_OUT = os.path.join(OUT_DIR, "LR2")

os.makedirs(HR_OUT, exist_ok=True)
os.makedirs(LR1_OUT, exist_ok=True)
os.makedirs(LR2_OUT, exist_ok=True)

# =========================
# HELPERS
# =========================
def bicubic_downsample(img, scale):
    h, w, _ = img.shape
    return cv2.resize(
        img,
        (w // scale, h // scale),
        interpolation=cv2.INTER_CUBIC
    )

def gaussian_blur(img, kernel, sigma):
    return cv2.GaussianBlur(img, kernel, sigma)

# =========================
# DATASET CREATION
# =========================
images = sorted(os.listdir(HR_DIR))

print(f"Creating dual-LR dataset from {len(images)} HR images...")

for img_name in tqdm(images):
    hr_path = os.path.join(HR_DIR, img_name)

    hr = cv2.imread(hr_path)
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

    H, W, _ = hr.shape

    # Random aligned crop
    x = random.randint(0, W - CROP_SIZE)
    y = random.randint(0, H - CROP_SIZE)

    hr_crop = hr[y:y + CROP_SIZE, x:x + CROP_SIZE]

    # -------------------------
    # LR1: Bicubic downsampling
    # -------------------------
    lr1 = bicubic_downsample(hr_crop, SCALE)

    # -------------------------
    # LR2: Blur + bicubic downsampling
    # -------------------------
    blurred = gaussian_blur(hr_crop, BLUR_KERNEL, BLUR_SIGMA)
    lr2 = bicubic_downsample(blurred, SCALE)

    # -------------------------
    # Save (OpenCV expects BGR)
    # -------------------------
    cv2.imwrite(
        os.path.join(HR_OUT, img_name),
        cv2.cvtColor(hr_crop, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        os.path.join(LR1_OUT, img_name),
        cv2.cvtColor(lr1, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        os.path.join(LR2_OUT, img_name),
        cv2.cvtColor(lr2, cv2.COLOR_RGB2BGR)
    )

print("✅ Dual-LR dataset creation complete.")
