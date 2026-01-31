import cv2
import numpy as np

# =========================
# CONFIG
# =========================
HR_IMAGE_PATH = r"C:\PROJECTS\modified-edsr\eval.png"

LR1_OUTPUT_PATH = r"C:\PROJECTS\modified-edsr\eval_lr1.png"
LR2_OUTPUT_PATH = r"C:\PROJECTS\modified-edsr\eval_lr2.png"

SCALE = 4

# Same blur params used in dataset creation
BLUR_KERNEL = (5, 5)
BLUR_SIGMA = 1.2

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

# =========================
# LOAD HR IMAGE
# =========================
hr = cv2.imread(HR_IMAGE_PATH)
if hr is None:
    raise FileNotFoundError(f"Could not read image: {HR_IMAGE_PATH}")

hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

# =========================
# LR1: Bicubic downsampling
# =========================
lr1 = bicubic_downsample(hr, SCALE)

# =========================
# LR2: Blur + Bicubic
# =========================
blurred = cv2.GaussianBlur(hr, BLUR_KERNEL, BLUR_SIGMA)
lr2 = bicubic_downsample(blurred, SCALE)

# =========================
# SAVE (convert back to BGR)
# =========================
cv2.imwrite(LR1_OUTPUT_PATH, cv2.cvtColor(lr1, cv2.COLOR_RGB2BGR))
cv2.imwrite(LR2_OUTPUT_PATH, cv2.cvtColor(lr2, cv2.COLOR_RGB2BGR))

print("✅ Dual-LR images created successfully:")
print(f"LR1 saved at: {LR1_OUTPUT_PATH}")
print(f"LR2 saved at: {LR2_OUTPUT_PATH}")
