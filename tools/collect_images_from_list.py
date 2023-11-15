import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect images from list", add_help=True)
    parser.add_argument("--list", "-l", type=str, required=True, help="path to list")
    parser.add_argument("--src_dir", "-s", type=str, required=True, help="Source directory")
    parser.add_argument("--des_dir", "-d", type=str, required=True, help="Destination directory")
    args = parser.parse_args()

    if not os.path.exists(args.list):
        raise FileExistsError
    os.makedirs(args.des_dir, exist_ok=True)
    with open(args.list, 'r') as f:
        list_lines = f.readlines()
        for i, ln in enumerate(tqdm(list_lines)):
            ln = ln.strip()
            src_image_pth = os.path.join(args.src_dir, ln)
            des_image_pth = os.path.join(args.des_dir, ln)
            if os.path.exists(src_image_pth):
                shutil.copyfile(src_image_pth, des_image_pth)
            else:
                print("Image pth: {} does not exist!".format(src_image_pth))
