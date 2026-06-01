#data perprocessing
import os
import glob
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
from tqdm import tqdm
import re

import torch
import torch.nn as nn
import torch.nn.functional as F   

# ---------- Parameter Area ----------
IMG_ROOT = "!!!Your Path!!!" + "/dataset_Things_eeg/Image_set"
EEG_ROOT = "!!!Your Path!!!" + "/dataset_Things_eeg/Preprocessed_data_250Hz"

# Image loading amount (None=load all)
N_IMG_TRAIN: Optional[int] = None
N_IMG_TEST:  Optional[int] = None

# EEG loading amount + sub range to load (inclusive)
N_EEG_TRAIN: Optional[int] = None
N_EEG_TEST:  Optional[int] = None
SUB_RANGE   = range(1, 11)        
# --------------------------------

def natural_sort(files: List[str]) -> List[str]:
    import re
    return sorted(files, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
'''
def load_images(split: str, n_samples: Optional[int]) -> List[np.ndarray]:
    stage = "training_images" if split == "training" else "test_images"
    pattern = os.path.join(IMG_ROOT, stage, "*_*", "*.jpg")
    paths = natural_sort(glob.glob(pattern))
    if n_samples is not None:
        paths = paths[:n_samples]
    imgs = [np.array(Image.open(p).convert("RGB")) for p in tqdm(paths, desc=f"Loading {split} images")]
    return imgs
'''

def _num_key(path: str) -> int:
    """
    Extract the first continuous digit sequence from path and convert to int, ignoring subsequent letters.
    Example: /.../12a_cat/cat_3b.jpg -> 12 (folder) or 3 (image)
    """
    fname = os.path.basename(path)          # File name
    stem = os.path.splitext(fname)[0]       # Remove extension
    match = re.search(r'\d+', stem)
    if not match:
        raise ValueError(f"Cannot extract number: {path}")
    return int(match.group())


def load_images(split: str, n_samples: Optional[int]) -> List[np.ndarray]:
    stage = "training_images" if split == "training" else "test_images"

    # 1. Sort all folders by img_type_num
    folder_pattern = os.path.join(IMG_ROOT, stage, "*_*")
    folders = sorted(glob.glob(folder_pattern), key=_num_key)

    all_paths = []
    for folder in folders:
        # 2. Sort images within each folder by img_num
        img_pattern = os.path.join(folder, "*.jpg")
        img_paths = sorted(glob.glob(img_pattern), key=_num_key)
        all_paths.extend(img_paths)

        # 3. Early truncation
        if n_samples is not None and len(all_paths) >= n_samples:
            all_paths = all_paths[:n_samples]
            break

    # 4. Unified loading
    imgs = [np.array(Image.open(p).convert("RGB"))
            for p in tqdm(all_paths, desc=f"Loading {split} images (folder-by-folder)")]

    return imgs

def load_eeg(split: str, n_samples: Optional[int], sub_range: range) -> Tuple[np.ndarray, List[int]]:
    file_key = f"preprocessed_eeg_{split}.npy"
    eeg_list, subj_list = [], []
    for sub_id in sub_range:
        sub_dir = os.path.join(EEG_ROOT, f"sub-{sub_id:02d}")
        npy_path = os.path.join(sub_dir, file_key)
        if not os.path.isfile(npy_path):
            print(f"[WARN] {npy_path} does not exist, skipping")
            continue

        data = np.load(npy_path, allow_pickle=True)  # Load as dictionary
        eeg_array = data["preprocessed_eeg_data"]  # Extract EEG data portion
        eeg_list.append(eeg_array)
        subj_list.extend([sub_id] * eeg_array.shape[0])

    if not eeg_list:
        raise FileNotFoundError("No .npy files found in specified sub range")
    all_eeg = np.concatenate(eeg_list, axis=0)  # (total_trials, C, T)
    if n_samples is not None:
        all_eeg = all_eeg[:n_samples]
        subj_list = subj_list[:n_samples]
    return torch.from_numpy(all_eeg).float(), subj_list  

# ----------  Global cache: EEG by subject+split, images by split global cache  ----------
_SUBJECT_EEG_CACHE = {}  # key: (sub_id, split) -> eeg_tensor
_SPLIT_IMAGE_CACHE = {}  # key: split -> img_tensor

def get_or_load_images(split: str, n_samples: Optional[int] = None, logger=None) -> torch.Tensor:
    """
    Get image data for specified split, with global cache, ensuring each split is loaded only once
    
    Args:
        split: 'training' or 'test'
        n_samples: Number of samples, default None (load all)
        logger: Logger
    
    Returns:
        torch.Tensor: Image data
    """
    if split not in _SPLIT_IMAGE_CACHE:
        if logger:
            logger.info(f"Image cache miss, loading {split} split image data (only once)")
        _SPLIT_IMAGE_CACHE[split] = load_images_subset(split, n_samples)
    else:
        if logger:
            logger.debug(f"Image cache hit, reusing {split} split image data")
    
    return _SPLIT_IMAGE_CACHE[split]


def load_subject_eeg(sub_id: int, split: str, logger=None) -> torch.Tensor:
    """
    Load single subject's EEG data, with cache
    
    Args:
        sub_id: Subject ID
        split: 'training' or 'test'
        logger: Logger
    
    Returns:
        torch.Tensor: EEG data
    """
    key = (sub_id, split)
    
    if key not in _SUBJECT_EEG_CACHE:
        if logger:
            logger.info(f"EEG cache miss, loading subject {sub_id}'s {split} EEG data")
        
        # Only load this subject's EEG data
        eeg_data, _ = load_eeg_subset(split, [sub_id], n_samples=None)
        _SUBJECT_EEG_CACHE[key] = eeg_data
    else:
        if logger:
            logger.debug(f"EEG cache hit, reusing subject {sub_id}'s {split} EEG data")
    
    return _SUBJECT_EEG_CACHE[key]


def load_eeg_subset(split: str, sub_list: List[int], n_samples: Optional[int]) -> Tuple[torch.Tensor, List[int]]:
    """
    Only load EEG data for specified subject list, skip unnecessary subjects
    """
    file_key = f"preprocessed_eeg_{split}.npy"
    eeg_list, subj_list = [], []
    
    # Only iterate through required subjects
    for sub_id in sub_list:
        sub_dir = os.path.join(EEG_ROOT, f"sub-{sub_id:02d}")
        npy_path = os.path.join(sub_dir, file_key)
        if not os.path.isfile(npy_path):
            print(f"[WARN] {npy_path} does not exist, skipping")
            continue

        data = np.load(npy_path, allow_pickle=True)
        eeg_array = data["preprocessed_eeg_data"]
        eeg_list.append(eeg_array)
        subj_list.extend([sub_id] * eeg_array.shape[0])

    if not eeg_list:
        raise FileNotFoundError(f"No .npy files found in specified sub range: {sub_list}")
    
    all_eeg = np.concatenate(eeg_list, axis=0)
    if n_samples is not None:
        all_eeg = all_eeg[:n_samples]
        subj_list = subj_list[:n_samples]
    
    return torch.from_numpy(all_eeg).float(), subj_list


def load_images_subset(split: str, n_samples: Optional[int] = None) -> torch.Tensor:
    """
    Load image data (unchanged, as images are not subject-specific)
    Optimization: Directly return tensor format, avoiding subsequent repeated conversion
    """
    stage = "training_images" if split == "training" else "test_images"
    folders = sorted(glob.glob(os.path.join(IMG_ROOT, stage, "*_*")), key=_num_key)

    all_paths = []
    for folder in folders:
        img_pattern = os.path.join(folder, "*.jpg")
        img_paths = sorted(glob.glob(img_pattern), key=_num_key)
        all_paths.extend(img_paths)
        
        if n_samples is not None and len(all_paths) >= n_samples:
            all_paths = all_paths[:n_samples]
            break

    imgs = [np.array(Image.open(p).convert("RGB")) for p in tqdm(all_paths, desc=f"Loading {split} images")]
    # Directly return tensor format
    return torch.stack([torch.from_numpy(im).float() / 255. for im in imgs]).permute(0, 3, 1, 2)


def load_data_dict_by_subjects(sub_list: List[int], split: str, img_data: torch.Tensor, logger=None):
    """
    On-demand loading of specified subjects' EEG data, using passed shared image data
    
    Args:
        sub_list: List of subject IDs to load
        split: 'training' or 'test'
        img_data: Already loaded shared image data
        logger: Logger
    
    Returns:
        dict: {sub: (eeg_tensor, img_tensor)}, where img_tensor is identical across all subs
    """
    if logger:
        logger.info(f"Loading data: subjects={sorted(sub_list)}, split={split}")
    
    data = {}
    for sub_id in sub_list:
        eeg_data = load_subject_eeg(sub_id, split, logger)
        data[sub_id] = (eeg_data, img_data)  # Use passed shared image data
    
    return data

def load_eeg_by_subjects(sub_list: List[int], split: str, logger=None):
    """
    Only load EEG data for specified subjects (no images included)
    
    Args:
        sub_list: List of subject IDs to load
        split: 'training' or 'test'
        logger: Logger
    
    Returns:
        dict: {sub: eeg_tensor}
    """
    if logger:
        logger.info(f"Loading EEG data: subjects={sorted(sub_list)}, split={split}")
    
    data = {}
    for sub_id in sub_list:
        eeg_data = load_subject_eeg(sub_id, split, logger)
        data[sub_id] = eeg_data  # Only store EEG
    
    return data


if __name__ == "__main__":
    print("=== Start loading data ===")
    img_train = load_images("training", N_IMG_TRAIN)
    img_test  = load_images("test",     N_IMG_TEST)
    eeg_train, subj_train = load_eeg("training", N_EEG_TRAIN, SUB_RANGE)
    eeg_test,  subj_test  = load_eeg("test",     N_EEG_TEST,  SUB_RANGE)

    print("\n=== Loading complete ===")
    print(f"Training images: {len(img_train)} images")
    print(f"Test images: {len(img_test)} images")
    print(f"Training EEG array shape: {eeg_train.shape}  (subjects: {sorted(set(subj_train))})")
    print(f"Test EEG array shape: {eeg_test.shape}   (subjects: {sorted(set(subj_test))})")