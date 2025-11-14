import os
import glob
import random
import io

import numpy as np
from PIL import Image, ImageFilter
import pywt
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

# ---------------------------------------------------------------
# 1) CONFIGURATION:
# ---------------------------------------------------------------
DATASET_PATHS = {
    'DDPM':           'data/DDPM',
    'LDM':            'data/LDM',
    'DiffSwap':       'data/DiffSwap',
    'FLUX1_pro':      'data/FLUX1_pro',
    'SDXL':           'data/SDXL',
    'SDv15_Img2Img':  'data/SDv15_Img2Img',
    'SDv21_Img2Img':  'data/SDv21_Img2Img',
    'StyleGan1':      'data/StyleGan1',
    'StyleGan2_ada':  'data/StyleGan2_ada',
    'StyleGan3':      'data/StyleGan3',
    'Stargan_v2':     'data/Stargan_v2',
    'UVCGANv2':       'data/UVCGANv2',
    'EG3D':           'data/EG3D'
}

# If None or 0, uses all images in each directory.
MAX_SAMPLES_PER_CLASS = 3200

# ---------------------------------------------------------------
# 2) DWTOnYCbCr: Aligned SWT extractor (3 levels, 10 bands) with fixed‐choice
#    JPEG & blur
# ---------------------------------------------------------------
class DWTOnYCbCr:
    """
    Compute a 3‐level undecimated SWT (db4) on each Y/Cb/Cr channel of the *same* augmented image.
    Output shape: (30, size, size) = 3 channels × (3 levels × 3 bands + 1 approximation).

    If apply_augment=True, each call will:
      1) Random flip (horizontal & vertical, p=0.5 each)
      2) Random rotation (±15°)
      3) Random resized crop (90%–100% area → resize to size)
      4) Resize to (size, size)
      5) Random fixed‐choice JPEG compression
      6) Convert to YCbCr, split
      7) Random fixed‐choice Gaussian blur per channel
      8) Compute SWT per channel → 10 bands
      9) Stack and normalize each band to [0,1]
    """
    def __init__(self,
                 size=256,
                 wavelet='db4',
                 levels=3,
                 apply_augment=True,
                 jpeg_choices=(None, 90, 85, 80, 75),
                 blur_choices=(None, 0.5, 1.0, 1.5, 2.0)):
        self.size = size
        self.wavelet = wavelet
        self.levels = levels
        self.apply_augment = apply_augment
        self.jpeg_choices = jpeg_choices
        self.blur_choices = blur_choices

    def __call__(self, pil_img: Image.Image) -> np.ndarray:
        # 1) Geometric augment & resize on RGB
        img = pil_img.convert('RGB')
        if self.apply_augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
            if random.random() < 0.5:
                img = TF.vflip(img)
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle, interpolation=Image.BILINEAR)
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(0.9, 1.0), ratio=(0.9, 1.1)
            )
            img = TF.crop(img, i, j, h, w)
        # Resize (for both aug and no-aug)
        img = img.resize((self.size, self.size), resample=Image.BILINEAR)

        # 2) JPEG compression once
        if self.apply_augment:
            q = random.choice(self.jpeg_choices)
            if q is not None:
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=q)
                buf.seek(0)
                img = Image.open(buf).convert('RGB')

        # 3) Convert to YCbCr & split
        ycbcr = img.convert('YCbCr')
        channels = ycbcr.split()  # Y, Cb, Cr

        bands_all = []
        for ch in channels:
            arr = np.array(ch, dtype=np.float32) / 255.0
            # 4) Gaussian blur per channel
            if self.apply_augment:
                sigma = random.choice(self.blur_choices)
                if sigma is not None and sigma > 0.0:
                    tmp = Image.fromarray((arr * 255).astype(np.uint8))
                    tmp = tmp.filter(ImageFilter.GaussianBlur(radius=sigma))
                    arr = np.array(tmp, dtype=np.float32) / 255.0

            # 5) Compute SWT
            coeffs = pywt.swt2(arr, self.wavelet, level=self.levels)
            bands = []
            for _, (cH, cV, cD) in coeffs:
                bands.extend([cH, cV, cD])
            bands.append(coeffs[-1][0])  # final LL
            bands_all.append(np.stack(bands, axis=0))  # (10, size, size)

        # 6) Stack channels -> (30, size, size) and normalize per band
        stack = np.concatenate(bands_all, axis=0)
        mins = stack.reshape(stack.shape[0], -1).min(axis=1)[:, None, None]
        maxs = stack.reshape(stack.shape[0], -1).max(axis=1)[:, None, None]
        stack = (stack - mins) / (maxs - mins + 1e-8)
        return stack.astype(np.float32)

# ---------------------------------------------------------------
# 3) Helper: Build a list of all (image_path, label) pairs
# ---------------------------------------------------------------
def get_filepaths_and_labels(paths_dict: dict[str, str],
                               max_samples: int = None) -> tuple[list[str], list[int], dict[str, int]]:
    """
    Generates a list of file paths and their corresponding labels from a
    dictionary of class paths.

    Args:
        paths_dict (dict[str, str]): A dictionary mapping class names to their
            corresponding directory paths.
        max_samples (int, optional): The maximum number of samples to load from
            each class directory. Defaults to None.

    Returns:
        tuple: A tuple containing the following elements:
            - all_filepaths (list[str]): A list of absolute or relative paths to
              the images.
            - all_labels (list[int]): A list of integer labels corresponding to
              each image.
            - name_to_label (dict[str, int]): A dictionary mapping class names
              to their corresponding integer labels.
    """
    class_names = sorted(paths_dict.keys())
    name_to_label = {name: idx for idx, name in enumerate(class_names)}

    all_filepaths, all_labels = [], []
    for class_name in class_names:
        folder = paths_dict[class_name]
        imgs = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
        imgs = sorted(imgs)
        if max_samples and len(imgs) > max_samples:
            imgs = random.sample(imgs, max_samples)
        label = name_to_label[class_name]
        for fp in imgs:
            all_filepaths.append(fp)
            all_labels.append(label)

    return all_filepaths, all_labels, name_to_label

# ---------------------------------------------------------------
# 4) Helper: Split into 70% train, 20% val, 10% test
# ---------------------------------------------------------------
def split_filepaths_labels(paths_dict: dict[str, str],
                             max_samples: int = None,
                             seed: int = 42) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int], dict[str, int]]:
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        paths_dict (dict[str, str]): A dictionary mapping class names to their
            corresponding directory paths.
        max_samples (int, optional): The maximum number of samples to load from
            each class directory. Defaults to None.
        seed (int, optional): The random seed to use for shuffling the dataset.
            Defaults to 42.

    Returns:
        tuple: A tuple containing the following elements:
            - train_paths (list[str]): A list of file paths for the training set.
            - train_labels (list[int]): A list of labels for the training set.
            - val_paths (list[str]): A list of file paths for the validation set.
            - val_labels (list[int]): A list of labels for the validation set.
            - test_paths (list[str]): A list of file paths for the test set.
            - test_labels (list[int]): A list of labels for the test set.
            - name_to_label (dict[str, int]): A dictionary mapping class names
              to their corresponding integer labels.
    """
    filepaths, labels, name_to_label = get_filepaths_and_labels(paths_dict, max_samples)
    combined = list(zip(filepaths, labels))
    random.Random(seed).shuffle(combined)
    total = len(combined)
    n_train = int(0.7 * total)
    n_val = int(0.2 * total)
    train_comb = combined[:n_train]
    val_comb = combined[n_train:n_train + n_val]
    test_comb = combined[n_train + n_val:]
    train_paths, train_labels = zip(*train_comb)
    val_paths, val_labels = zip(*val_comb)
    test_paths, test_labels = zip(*test_comb)
    return list(train_paths), list(train_labels), list(val_paths), list(val_labels), list(test_paths), list(test_labels), name_to_label

# ---------------------------------------------------------------
# 5) SWTGeneratorDataset: PyTorch Dataset returning (30,H,W) tensor + label
# ---------------------------------------------------------------
class SWTGeneratorDataset(Dataset):
    """
    PyTorch Dataset for loading images, extracting SWT features, and returning a
    (30, H, W) tensor and its corresponding label.

    Args:
        filepaths (list[str]): A list of absolute or relative paths to the images.
        labels (list[int]): A list of integer labels corresponding to each image.
        extractor (DWTOnYCbCr): An instance of the DWTOnYCbCr class, which is
            responsible for extracting SWT features from the images.
        target_size (int, optional): The target size for the output tensor.
            Defaults to 240.
        mean (torch.Tensor, optional): A tensor of mean values for each of the
            30 channels, used for normalization. Defaults to None.
        std (torch.Tensor, optional): A tensor of standard deviation values for
            each of the 30 channels, used for normalization. Defaults to None.
    """
    def __init__(self,
                 filepaths: list[str],
                 labels: list[int],
                 extractor: DWTOnYCbCr,
                 target_size: int = 240,
                 mean: torch.Tensor = None,
                 std: torch.Tensor = None):
        assert len(filepaths) == len(labels), "filepaths and labels must be same length"
        self.filepaths = filepaths
        self.labels = labels
        self.extractor = extractor  # returns full 30-band array
        self.target_size = target_size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('RGB')
        bands30 = self.extractor(img)                  # (30, size, size)
        tensor30 = torch.from_numpy(bands30)           # to tensor
        tensor30 = TF.resize(tensor30, [self.target_size, self.target_size])
        if self.mean is not None and self.std is not None:
            tensor30 = (tensor30 - self.mean[:, None, None]) / self.std[:, None, None]
        label = self.labels[idx]
        return tensor30.float(), label
