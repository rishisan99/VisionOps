# app/backend/utils/image_ops.py

import io
import base64
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def read_image_from_bytes(contents: bytes) -> Image.Image:
    return Image.open(io.BytesIO(contents)).convert("RGB")

def cam_overlay_to_png_bytes(rgb_img: Image.Image, cam_01: torch.Tensor) -> bytes:
    cam_np = cam_01.detach().cpu().numpy()
    img_np = np.array(rgb_img).astype(np.float32) / 255.0

    # Resize CAM to original image size
    if cam_np.shape[0] != img_np.shape[0] or cam_np.shape[1] != img_np.shape[1]:
        cam_img = Image.fromarray((cam_np * 255).astype(np.uint8))
        cam_img = cam_img.resize((img_np.shape[1], img_np.shape[0]), resample=Image.BILINEAR)
        cam_np = np.array(cam_img).astype(np.float32) / 255.0

    fig = plt.figure()
    plt.imshow(img_np)
    plt.imshow(cam_np, alpha=0.45)  # default colormap
    plt.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()

def cam_overlay_to_base64(rgb_img: Image.Image, cam_01: torch.Tensor) -> str:
    png_bytes = cam_overlay_to_png_bytes(rgb_img, cam_01)
    return base64.b64encode(png_bytes).decode("utf-8")
