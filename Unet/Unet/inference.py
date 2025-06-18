import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import ToTensor, Compose, CenterCrop, ToPILImage
import os

# 🔥 ARGÜMANLARI AL
parser = argparse.ArgumentParser(description='PyTorch Inference')
parser.add_argument('-i', '--input', type=str, required=True, help='Input image path')
parser.add_argument('-o', '--output', type=str, required=True, help='Output image path or folder')
parser.add_argument('-m', '--model', type=str, required=True, help='Model checkpoint path')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 📂 Eğer sadece klasör verdiyse, otomatik isim ver
if os.path.isdir(args.output):
    input_basename = os.path.basename(args.input)
    input_name, _ = os.path.splitext(input_basename)
    output_filename = f"{input_name}.png"
    args.output = os.path.join(args.output, output_filename)
    print(f"📂 Output file auto-generated as: {args.output}")

# 📷 Görsel oku
img = Image.open(args.input).convert('RGB')

# 📐 Crop işlemi
crop_size = img.size[0]
crop_size = crop_size - (crop_size % 4)
transform = Compose([
    CenterCrop(crop_size),
    ToTensor(),
])

img_tensor = transform(img)
data = img_tensor.unsqueeze(0).to(device)
print(f"Input tensor shape: {data.shape}")

# 🔥 Modeli yükle
model = torch.load(args.model, map_location=device)
model = model.to(device)
model.eval()

if torch.cuda.is_available():
    cudnn.benchmark = True

# 🔮 İnference
with torch.no_grad():
    output_tensor = model(data)

output_tensor = output_tensor.squeeze(0).cpu()
print(f"Output tensor shape: {output_tensor.shape}")

# 💾 Çıkış görseli kaydet (SADECE OUTPUT)
output_image = ToPILImage()(output_tensor).convert('RGB')
output_image.save(args.output)

print(f"✅ Output image (model output only) saved at: {args.output}")