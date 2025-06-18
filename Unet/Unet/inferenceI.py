import argparse
import torch
from model import UNet
from PIL import Image
import numpy as np
import os

# 🔥 ARGÜMANLARI AL
parser = argparse.ArgumentParser(description='PyTorch UNet Inpainting Inference')
parser.add_argument('-i', '--input', type=str, required=True, help='Input masked image path')
parser.add_argument('-o', '--output', type=str, required=True, help='Output folder or file')
parser.add_argument('-m', '--model', type=str, required=False, default='./checkpoints/CP100.pth', help='Model checkpoint path')
args = parser.parse_args()

input_path = args.input
output_dir = args.output
model_path = args.model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 📂 Eğer sadece klasör verdiyse, otomatik dosya ismi üretelim
if os.path.isdir(output_dir):
    input_basename = os.path.basename(input_path)
    input_name, _ = os.path.splitext(input_basename)
    output_filename = f"{input_name}_output.png"
    output_dir = os.path.join(output_dir, output_filename)
    print(f"📂 Output file auto-generated as: {output_dir}")

# 📂 Klasör varsa yoksa oluştur
os.makedirs(os.path.dirname(output_dir), exist_ok=True)

# Modeli yükle
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 1. MASKELİ GÖRSELİ OKU
img = Image.open(input_path).convert('RGB')
img = img.resize((128, 128))  # Eğitimdeki gibi

input_rgb = np.array(img, dtype=np.float32) / 255.0  # (H,W,3)

# 2. DOĞRU MASK ÇIKAR
threshold = 0.1  # Siyah için tolerans
mask = ((input_rgb[:,:,0] < threshold) & 
        (input_rgb[:,:,1] < threshold) & 
        (input_rgb[:,:,2] < threshold)).astype(np.float32)

mask = 1 - mask

# 3. 4 KANALLI INPUT YAP
img_input = np.concatenate((input_rgb, mask[..., np.newaxis]), axis=-1)  # (H,W,4)

# Tensor'a çevir
input_tensor = torch.from_numpy(img_input).permute(2,0,1).unsqueeze(0).to(device)

# 4. MODEL INFERENCE
with torch.no_grad():
    output = model(input_tensor)

output = output.squeeze(0).permute(1,2,0).cpu().numpy()
output = np.clip(output, 0, 1)

# 5. OUTPUT GÖRSELİ KAYDET
output_img = (output * 255.0).astype(np.uint8)
output_img = Image.fromarray(output_img)
output_img.save(output_dir)

print(f"✅ Model output saved at: {output_dir}")