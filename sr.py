import cv2
import torch
import numpy as np

from model.edsr_dual import EDSRDual

# =========================
# CONFIG
# =========================
LR1_PATH = r"C:\PROJECTS\modified-edsr\eval_lr1.png"
LR2_PATH = r"C:\PROJECTS\modified-edsr\eval_lr2.png"

MODEL_PATH = r"C:\PROJECTS\modified-edsr\edsr_dual.pth"
OUTPUT_HR_PATH = r"C:\PROJECTS\modified-edsr\sr_output.png"

SCALE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# IMAGE LOADER
# =========================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

# =========================
# LOAD LR IMAGES
# =========================
lr1 = load_image(LR1_PATH).to(device)
lr2 = load_image(LR2_PATH).to(device)

# =========================
# LOAD MODEL
# =========================
model = EDSRDual(scale=SCALE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =========================
# SUPER-RESOLUTION
# =========================
with torch.no_grad():
    sr = model(lr1, lr2)

# =========================
# SAVE OUTPUT
# =========================
sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()
sr_np = np.clip(sr_np, 0.0, 1.0)

sr_bgr = cv2.cvtColor((sr_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite(OUTPUT_HR_PATH, sr_bgr)

print("✅ Super-resolution complete")
print(f"HR output saved at: {OUTPUT_HR_PATH}")
