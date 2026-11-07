import argparse
import os

import numpy as np

from utils.image_utils import load_images_from_folder, preprocess_images


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Load the two class folders as 128x128 grayscale, scale to [0, 1] and save .npy arrays.")
    parser.add_argument("--brain-tumor-dir", default=os.path.join("data", "brain_tumor"),
                        help="folder with brain tumor images (default: data/brain_tumor)")
    parser.add_argument("--non-brain-tumor-dir", default=os.path.join("data", "non_brain_tumor"),
                        help="folder with non brain tumor images (default: data/non_brain_tumor)")
    parser.add_argument("--output-dir", default="data",
                        help="folder for brain_tumor_images.npy and non_brain_tumor_images.npy (default: data)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    brain_tumor_images = load_images_from_folder(args.brain_tumor_dir)
    non_brain_tumor_images = load_images_from_folder(args.non_brain_tumor_dir)

    brain_tumor_images = preprocess_images(brain_tumor_images)  # Normalize images to [0, 1]
    non_brain_tumor_images = preprocess_images(non_brain_tumor_images)  # Normalize images to [0, 1]

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'brain_tumor_images.npy'), brain_tumor_images)
    np.save(os.path.join(args.output_dir, 'non_brain_tumor_images.npy'), non_brain_tumor_images)


if __name__ == "__main__":
    main()
