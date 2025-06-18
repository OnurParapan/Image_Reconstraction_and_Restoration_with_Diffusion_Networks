import os
import argparse
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from glob import glob
from tqdm import tqdm

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.

def compare_folders(gt_dir, pred_dir):
    gt_images = sorted(glob(os.path.join(gt_dir, '*')))
    pred_images = sorted(glob(os.path.join(pred_dir, '*')))

    assert len(gt_images) == len(pred_images), "Number of images must match!"

    psnr_list = []
    ssim_list = []

    for gt_path, pred_path in tqdm(zip(gt_images, pred_images), total=len(gt_images)):
        gt = load_image(gt_path)
        pred = load_image(pred_path)

        psnr_val = psnr(gt, pred, data_range=1.0)
        ssim_val = ssim(gt, pred, multichannel=True, data_range=1.0)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print(f"\n🔎 Comparison Results:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    return psnr_list, ssim_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True, help='Ground truth image folder')
    parser.add_argument('--pred_dir', type=str, required=True, help='Predicted image folder (model output)')
    args = parser.parse_args()

    compare_folders(args.gt_dir, args.pred_dir)
