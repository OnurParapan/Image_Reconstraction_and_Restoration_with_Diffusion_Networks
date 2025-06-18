import os
import requests
from os.path import join, exists
from torchvision.transforms import Compose, Resize, ToTensor

from .dataset import DatasetFromFolder

def download_bsd300(dest="./dataset"):
    output_image_dir = join(dest, "RealSet65")
    if not exists(output_image_dir):
        os.makedirs(output_image_dir, exist_ok=True)

    # GitHub API URL to list contents of the RealSet65 directory
    api_url = "https://api.github.com/repos/zsyOAOA/ResShift/contents/testdata/RealSet65"

    headers = {
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch file list from GitHub API. Status code: {response.status_code}")

    file_list = response.json()

    for file_info in file_list:
        file_name = file_info["name"]
        download_url = file_info["download_url"]
        file_path = join(output_image_dir, file_name)

        if not exists(file_path):
            print(f"Downloading {file_name}...")
            r = requests.get(download_url, stream=True)
            if r.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                print(f"Failed to download {file_name}. Status code: {r.status_code}")

    print("RealSet65 dataset has been downloaded successfully.")
    return output_image_dir

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform(crop_size, upscale_factor):
    lr_size = crop_size // upscale_factor
    return Compose([
        Resize((lr_size, lr_size)),  # Input'u küçük boyuta çek
        ToTensor(),
    ])

def target_transform(crop_size):
    return Compose([
        Resize((crop_size, crop_size)),  # Target'ı da sabit büyük boyuta çek
        ToTensor(),
    ])

def get_training_set(upscale_factor):
    root_dir = download_bsd300()
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(root_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))

def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(root_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
