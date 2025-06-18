import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from math import ceil

def main():
    parser = argparse.ArgumentParser(description="Create mask from input image")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to masked input image.")
    parser.add_argument("--padding_offset", type=int, default=64, help="Padding offset, usually 64.")
    args = parser.parse_args()

    input_path = args.input
    input_dir = os.path.dirname(input_path)

    # 1. Input image oku
    img = Image.open(input_path).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0
    input_h, input_w, _ = img_np.shape

    # 2. Maskeyi çıkar
    threshold = 0.1
    mask = ((img_np[:,:,0] < threshold) & 
            (img_np[:,:,1] < threshold) & 
            (img_np[:,:,2] < threshold)).astype(np.float32)

    # 3. Maskeyi torch tensor yap
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # 4. Maskeye input gibi padding uygula
    padding_offset = args.padding_offset
    pad_h = (padding_offset - input_h % padding_offset) % padding_offset
    pad_w = (padding_offset - input_w % padding_offset) % padding_offset

    if pad_h != 0 or pad_w != 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    # 5. Maskeyi kaydet
    mask = mask.squeeze().cpu().numpy()
    input_basename = os.path.basename(input_path)
    input_name, _ = os.path.splitext(input_basename)
    mask_filename = f"{input_name}_mask.jpg"
    mask_save_path = os.path.join(input_dir, mask_filename)

    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(mask_save_path)

    print(f"✅ Padded Mask saved at: {mask_save_path}")

if __name__ == "__main__":
    main()
