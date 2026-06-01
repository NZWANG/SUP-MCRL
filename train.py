# train.py
import argparse
import os
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from basic_model import EmotionAligner
from data_perprocessing import load_eeg, load_images, load_data_dict_by_subjects, get_or_load_images, load_eeg_by_subjects
from loss import InfoNCELoss
from logger import get_logger
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torch.utils.data import Dataset, DataLoader

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

USE_aveage_eeg_signal = True
N_WAY_EVAL_ROUNDS = 50

# Note: T_all now serves only as default value, actual usage gets from criterion
T_all = 0.07


def log_args(args, logger):
    """Write all args to log"""
    logger.info("========== All Arguments ==========")
    for k, v in vars(args).items():
        logger.info(f"{k:<20}: {v}")
    logger.info("===================================")

# ==========================  Global Cache  ==========================
# key: (sub_list_tuple, split)  ->  dict {sub: (eeg, img)}
_DATA_CACHE = {}

def load_data_dict_cached(sub_list, split, logger=None):
    """Cached version to avoid repeated I/O during multi-fold training"""
    # Filter out cached subjects, only load new subjects
    key = (tuple(sorted(sub_list)), split)
    
    # If not in cache, create empty dict
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = {}
    
    cached_subs = set(_DATA_CACHE[key].keys())
    required_subs = set(sub_list)
    
    # Find subjects that need to be newly loaded
    subs_to_load = required_subs - cached_subs
    
    if subs_to_load:
        if logger:
            logger.info(f"Loading subject data: {sorted(subs_to_load)} (total {len(subs_to_load)} subjects)")
        # Only load data for new subjects
        new_data = _load_data_dict(list(subs_to_load), split)
        _DATA_CACHE[key].update(new_data)
    
    # Return all required subject data
    return {sub: _DATA_CACHE[key][sub] for sub in required_subs}

def _load_data_dict(sub_list, split):
    """Actual loading logic (original load_data_dict)"""
    data = {}
    for sub in sub_list:
        eeg, _ = load_eeg(split, n_samples=None, sub_range=range(sub, sub + 1))
        img = load_images(split, n_samples=None)
        img = torch.stack([torch.from_numpy(im).float() / 255. for im in img]).permute(0, 3, 1, 2)
        # Decide whether to average EEG data along trial dimension based on global flag
        if USE_aveage_eeg_signal:
            # Average along second dimension (M dim), keep dim: [N, M, C, T] -> [N, 1, C, T]
            eeg = eeg.mean(dim=1, keepdim=True)
        data[sub] = (eeg, img)
    return data

class EEGImagePairDataset(Dataset):
    """Custom dataset for EEG-image one-to-many mapping, avoiding memory copy
    
    Optimized version: Supports multi-subject EEG data mapping to single copy of image data, no image copying
    """
    def __init__(self, eeg_data, img_data, is_train, num_subjects=1):
        """
        Args:
            eeg_data: EEG data, shape (num_subjects*N, M, C, T)
            img_data: Image data, shape (N, C, H, W) - loaded once, not copied
            is_train: Whether in training mode
            num_subjects: Number of subjects, used for image index mapping
        """
        # Verify data consistency: EEG dim 0 must be divisible by subject count
        assert eeg_data.shape[0] % num_subjects == 0, f"EEG dim 0 ({eeg_data.shape[0]}) must be divisible by subject count ({num_subjects})"
        
        self.N = img_data.shape[0]  # Number of images (per subject)
        self.M = eeg_data.shape[1]  # Number of EEG trials per image
        self.C, self.T = eeg_data.shape[2], eeg_data.shape[3]
        self.img_data = img_data  # Keep original image data, no copy
        self.eeg_data = eeg_data
        self.is_train = is_train
        self.num_subjects = num_subjects
        
        # Total EEG samples = num_subjects * N * M
        self.total_eeg_samples = eeg_data.shape[0] * self.M
        
    def __len__(self):
        """Return total EEG samples = num_subjects * N * M"""
        return self.total_eeg_samples
    
    def __getitem__(self, idx):
        """
        Return corresponding EEG and image based on EEG index
        idx: 0 to num_subjects*N*M-1
        
        Index mapping logic:
        - eeg_idx_in_all = idx // M  # Position among all (N*num_subjects) images
        - img_idx = eeg_idx_in_all % self.N  # Map to original image index via modulo
        - eeg_idx_in_img = idx % M  # EEG trial index within this image
        """
        # Compute index at image level (considering multi-subject)
        eeg_idx_in_all = idx // self.M
        
        # Key: Map to original image index via modulo, avoid copying image data
        img_idx = eeg_idx_in_all % self.N
        
        # Compute EEG index within this image
        eeg_idx_in_img = idx % self.M
        
        # Get EEG sample (C, T)
        eeg = self.eeg_data[eeg_idx_in_all, eeg_idx_in_img]
        
        # Get corresponding image (C, H, W) - no copy, direct indexing
        img = self.img_data[img_idx]

        if self.is_train:
            category = img_idx // 10
        else:
            category = img_idx
        
        return eeg, img, category

def validate_model(model, val_loader, device, logger, epoch, fold_idx, criterion, infer_batch_size=256):
    """
    N-way top-K evaluation version (EEG trial → image retrieval)
    Data structure: Each image has multiple EEG trials, no averaging, each trial tested independently
    
    New: Each n_way performs N_WAY_EVAL_ROUNDS independent evaluations (randomly sample different negatives)
          Similarity matrix computed only once, reused outside loop
    
    Modified: Get current learnable temperature through criterion
    
    Returns:
        acc_top1: dict, top-1 accuracy for each N-way (average)
        acc_top5: dict, top-5 accuracy for each N-way (average)
    """
    model.eval()
    
    # Modified: Get current temperature from criterion
    with torch.no_grad():
        temperature = criterion.get_temperature().item()
        logit_scale = criterion.get_logit_scale().item()
    
    logger.info(f"fold{fold_idx} epoch{epoch}: Validation using temperature τ={temperature:.6f}, logit_scale={logit_scale:.4f}")
    
    all_eeg_features = []      # All EEG trial features
    eeg_to_img_id = []         # Image ID corresponding to each trial
    unique_img_features = {}   # Cache: Image ID → feature
    img_id_list = []           # Ordered list of unique image IDs
    total_eeg_trials = 0
    
    with torch.no_grad():
        logger.info(f"fold{fold_idx} epoch{epoch}: Extracting EEG features and image features...")
        
        # Stage 1: Iterate all data, extract features
        for batch_idx, (eeg_batch, img_batch, category_batch) in enumerate(
                tqdm(val_loader, desc=f"fold{fold_idx} epoch{epoch} feature extraction", ncols=100)
            ):
            eeg_feat_batch, img_feat_batch = model(eeg_batch, img_batch)
            all_eeg_features.append(eeg_feat_batch.cpu())
            
            for i in range(eeg_batch.size(0)):
                img_id = category_batch[i].item()
                eeg_to_img_id.append(img_id)
                total_eeg_trials += 1
                
                if img_id not in unique_img_features:
                    unique_img_features[img_id] = img_feat_batch[i:i+1].cpu()
                    img_id_list.append(img_id)
        
        # Stage 2: Build feature matrices
        all_eeg_features = torch.cat(all_eeg_features, dim=0).to(device).float()  # (N_EEG_TOTAL, embed_dim)
        all_img_features = torch.cat([unique_img_features[img_id] for img_id in img_id_list], dim=0).to(device).float()  # (N_IMG, embed_dim)
        
        # Build mapping: Image ID → feature matrix row index
        img_id_to_feat_idx = {img_id: idx for idx, img_id in enumerate(img_id_list)}
        
        # Build positive indices: row index of corresponding image in feature matrix for each EEG trial
        positive_indices = torch.tensor(
            [img_id_to_feat_idx[img_id] for img_id in eeg_to_img_id],
            device=device
        )
        
        # Modified: Compute similarity matrix using logit_scale obtained from criterion
        # Compute raw similarity matrix
        raw_similarities = torch.matmul(all_eeg_features, all_img_features.T) * logit_scale  # (N_EEG_TOTAL, N_IMG)
        similarity_matrix = torch.softmax(raw_similarities, dim=-1)
        
        # Stage 3: N-way top-K evaluation (average over multiple loops)
        n_way_values = [10, 50, 100, 200]
        acc_top1 = {n: 0.0 for n in n_way_values}
        acc_top5 = {n: 0.0 for n in n_way_values}

        for n_way in n_way_values:
            if n_way > all_img_features.size(0):
                logger.warning(f"fold{fold_idx} epoch{epoch}: N-way={n_way} exceeds total image count {all_img_features.size(0)}, skipping")
                continue
            
            # Store results from multiple evaluations
            top1_scores = []
            top5_scores = []
            
            # Perform multiple independent evaluations
            for round_idx in range(N_WAY_EVAL_ROUNDS):
                # Reuse pre-computed similarity matrix
                # Mask positive sample positions (operate on raw similarity matrix)
                masked_sim = similarity_matrix.clone()
                masked_sim[torch.arange(total_eeg_trials), positive_indices] = -float('inf')
                
                # Randomly sample negative samples
                num_neg = all_img_features.size(0) - 1
                all_neg_indices = torch.zeros(total_eeg_trials, num_neg, dtype=torch.long, device=device)
                for i in range(total_eeg_trials):
                    all_neg_indices[i] = torch.where(masked_sim[i] > -float('inf'))[0]
                
                perm = torch.randperm(num_neg, device=device)[:n_way-1]
                sampled_negatives = all_neg_indices[:, perm]
                
                # Build candidate set
                candidate_indices = torch.cat([
                    positive_indices.unsqueeze(1),
                    sampled_negatives
                ], dim=1)
                
                # Get candidate scores (directly index from similarity matrix)
                batch_indices = torch.arange(total_eeg_trials, device=device).unsqueeze(1).expand(-1, n_way)
                candidate_scores = similarity_matrix[batch_indices, candidate_indices]
                
                # Compute positive sample ranking
                rank_in_nway = torch.sum(candidate_scores > candidate_scores[:, 0:1], dim=1)
                
                # Compute accuracy for this round
                correct_top1 = torch.sum(rank_in_nway < 1).item()
                correct_top5 = torch.sum(rank_in_nway < 5).item()
                
                top1_scores.append(correct_top1 / total_eeg_trials * 100)
                top5_scores.append(correct_top5 / total_eeg_trials * 100)
                
                # Clean up temporary variables for this round
                del masked_sim, candidate_indices, batch_indices, candidate_scores
            
            # Compute average over multiple evaluations
            acc_top1[n_way] = sum(top1_scores) / len(top1_scores)
            acc_top5[n_way] = sum(top5_scores) / len(top5_scores)
            
            # Compute statistics
            top1_std = torch.tensor(top1_scores).std().item() if len(top1_scores) > 1 else 0.0
            top5_std = torch.tensor(top5_scores).std().item() if len(top5_scores) > 1 else 0.0
            
            logger.info(f"  N-way={n_way:3d}: "
                       f"top-1={acc_top1[n_way]:.5f}% (±{top1_std:.3f}), "
                       f"top-5={acc_top5[n_way]:.5f}% (±{top5_std:.3f}) | "
                       f"evaluated {N_WAY_EVAL_ROUNDS} rounds")
        
        # Cleanup
        del similarity_matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return acc_top1, acc_top5

def validate_model_heatmap(model, val_loader, device, logger, fold_idx, criterion):
    """
    Validate model and plot EEG-image similarity heatmap
    
    Modified: Get current learnable temperature through criterion
    
    Data structure:
    - If USE_aveage_eeg_signal=False: Each image has multiple EEG trials (default 80)
      Average similarity matrix every M rows to get image-image similarity matrix, add black border to diagonal elements
    - If USE_aveage_eeg_signal=True: Each image has only 1 EEG trial (pre-averaged)
      Directly use similarity matrix as image-image similarity matrix, no border added
    """
    model.eval()
    
    # Modified: Get current temperature from criterion
    with torch.no_grad():
        temperature = criterion.get_temperature().item()
        logit_scale = criterion.get_logit_scale().item()
    
    logger.info(f"fold{fold_idx}: Heatmap validation using temperature τ={temperature:.6f}, logit_scale={logit_scale:.4f}")
    
    all_eeg_features = []
    eeg_to_img_id = []
    unique_img_features = {}
    img_id_list = []
    total_eeg_trials = 0
    
    with torch.no_grad():
        logger.info(f"fold{fold_idx}: Extracting EEG and image features for heatmap plotting...")
        
        # Feature extraction
        for batch_idx, (eeg_batch, img_batch, category_batch) in enumerate(
                tqdm(val_loader, desc=f"fold{fold_idx} heatmap feature extraction", ncols=100)
            ):
            eeg_feat_batch, img_feat_batch = model(eeg_batch, img_batch)
            all_eeg_features.append(eeg_feat_batch.cpu())
            
            for i in range(eeg_batch.size(0)):
                img_id = category_batch[i].item()
                eeg_to_img_id.append(img_id)
                total_eeg_trials += 1
                
                if img_id not in unique_img_features:
                    unique_img_features[img_id] = img_feat_batch[i:i+1].cpu()
                    img_id_list.append(img_id)
        
        # Build feature matrices
        all_eeg_features = torch.cat(all_eeg_features, dim=0).to(device).float()
        all_img_features = torch.cat([unique_img_features[img_id] for img_id in img_id_list], dim=0).to(device).float()
        
        num_images = all_img_features.size(0)
        trials_per_image = total_eeg_trials // num_images
        
        logger.info(f"fold{fold_idx}: Total {total_eeg_trials} EEG trials, {num_images} images, {trials_per_image} trials per image")
        
        # Compute similarity matrix - using logit_scale obtained from criterion
        similarity_matrix = logit_scale * torch.matmul(all_eeg_features, all_img_features.T)
        
        # Modified: Normalize similarity to [0,1] range
        # Based on L2-normalized features, cosine similarity range [-1, 1], after multiplying by logit_scale range becomes [-logit_scale, logit_scale]
        min_val = -logit_scale
        max_val = logit_scale
        similarity_matrix = (similarity_matrix - min_val) / (max_val - min_val)
        # Ensure within [0,1] range (handle possible numerical errors)
        similarity_matrix = torch.clamp(similarity_matrix, 0.0, 1.0)
        
        # Modified: Directly assign heatmap_data, do not use sum_similarity_matrix
        if USE_aveage_eeg_signal:
            # Pre-averaged, use directly
            heatmap_data = similarity_matrix.cpu().numpy()
            aggregation_info = "pre-averaged EEG"
            is_averaging_mode = True  # Not averaging mode
        else:
            # Need to average multiple trials per image
            heatmap_data = similarity_matrix.reshape(num_images, trials_per_image, num_images).mean(dim=1).cpu().numpy()
            aggregation_info = f"avg of {trials_per_image} trials"
            is_averaging_mode = False  # Averaging mode
        
        # New: Create custom colormap (white->blue)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('white_blue', ['white', 'blue'])
        
        # Plot heatmap
        plt.figure(figsize=(12, 12))
        ax = sns.heatmap(
            heatmap_data,
            cmap=cmap,           # Use custom colormap
            vmin=0,              # Fix min to 0 (white)
            vmax=1,              # Fix max to 1 (blue)
            square=True,
            xticklabels=20,
            yticklabels=20,
            cbar_kws={'label': f'Similarity [0,1] ({aggregation_info}, τ={temperature:.4f})'}
        )
        
        # New: Add black border to diagonal in averaging mode
        if is_averaging_mode:
            from matplotlib.patches import Rectangle
            for i in range(num_images):
                # Find max value index in row i
                max_idx = np.argmax(heatmap_data[i, :])
                # Add black border to max value cell, linewidth 1, no fill
                rect = Rectangle((max_idx, i), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rect)
        
        avg_status = "ON" if USE_aveage_eeg_signal else "OFF"
        mode_info = f"EEG averaging: {avg_status} | {aggregation_info} | τ={temperature:.4f}"
        if is_averaging_mode:
            mode_info += " | Max per row highlighted"
        
        plt.title(f'Similarity Heatmap - Fold {fold_idx}\n' + mode_info, 
                  fontsize=16, pad=20)
        plt.xlabel('Image ID', fontsize=12)
        plt.ylabel('Image ID (EEG trials group)', fontsize=12)
        
        save_dir = './heatmap'
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f'heatmap_fold_{fold_idx}_{timestamp}.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"fold{fold_idx}: Heatmap saved to {save_path}")
        
        # Cleanup memory
        plt.close()
        del similarity_matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Modified data loading part in train_one_fold function
def train_one_fold(train_subs, val_sub, args, fold_idx=0):
    logger = get_logger("train")
    log_args(args, logger)
    
    # Mode recognition (unchanged)
    if len(train_subs) == 1 and val_sub is None:
        mode_str = "Intra-subject"
    elif val_sub is not None:
        mode_str = "Leave-one-subject-out (loso)"
        logger.info(f"fold {fold_idx} -> Validation subject: {val_sub}")
    else:
        mode_str = "Mixed training (inter_mix)"
    
    logger.info(f"fold {fold_idx} -> Mode: {mode_str}")
    logger.info(f"Training subjects: {sorted(train_subs)} (total {len(train_subs)})")
    
    early_stopper = EarlyStopping(patience=args.patience, delta=args.delta, logger=logger)
    
    # Step 1: Load image data first (each split loaded only once)
    logger.info("Loading training image data (global shared)...")
    img_train = get_or_load_images("training", logger=logger)
    
    logger.info("Loading validation image data (global shared)...")
    img_val = get_or_load_images("test", logger=logger)
    
    # Step 2: Only load EEG data (no images included)
    logger.info("Loading training EEG data...")
    train_eeg_dict = load_eeg_by_subjects(train_subs, "training", logger=logger)
    
    # Concatenate all training subjects' EEG data
    all_eeg_train = []
    sub_num = len(train_subs)
    for sub in train_subs:
        eeg = train_eeg_dict[sub]
        if USE_aveage_eeg_signal:
            eeg = eeg.mean(dim=1, keepdim=True)
        all_eeg_train.append(eeg)
    
    eeg_train = torch.cat(all_eeg_train, dim=0)
    logger.info(f"Training data: {img_train.shape[0]} images, {sub_num}*{eeg_train.shape[0] / sub_num}*{eeg_train.shape[1]} EEG samples")
    
    # Step 3: Load validation EEG data
    logger.info("Loading validation EEG data...")
    if val_sub is not None:
        # LOSO mode: Only load validation subject's test data
        val_eeg_dict = load_eeg_by_subjects([val_sub], "test", logger=logger)
        eeg_val = val_eeg_dict[val_sub]
        if USE_aveage_eeg_signal:
            eeg_val = eeg_val.mean(dim=1, keepdim=True)
    else:
        # Intra/inter_mix mode: Load training subjects' test data
        val_eeg_dict = load_eeg_by_subjects(train_subs, "test", logger=logger)
        all_eeg_val = []
        for sub in train_subs:
            eeg = val_eeg_dict[sub]
            if USE_aveage_eeg_signal:
                eeg = eeg.mean(dim=1, keepdim=True)
            all_eeg_val.append(eeg)
        eeg_val = torch.cat(all_eeg_val, dim=0)
    
    logger.info(f"Validation data: {img_val.shape[0]} images, {eeg_val.shape[0]}*{eeg_val.shape[1]} EEG samples")
    
    # Print data shape information
    N_train, M, C, T = eeg_train.shape
    N_val, M_v, _, _ = eeg_val.shape
    
    # Compute subject count for Dataset index mapping
    num_train_subs = len(train_subs)
    num_val_subs = 1 if val_sub is not None else num_train_subs
    
    logger.info(f'Training set: {num_train_subs} subjects, images not copied, accessed via index mapping')
    logger.info(f'Validation set: {num_val_subs} subjects, images not copied, accessed via index mapping')
    
    # Create data loaders - pass num_subjects parameter, do not copy images
    train_ds = EEGImagePairDataset(eeg_train, img_train, is_train=True, num_subjects=num_train_subs)
    val_ds = EEGImagePairDataset(eeg_val, img_val, is_train=False, num_subjects=num_val_subs)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    device = args.device
    model = EmotionAligner(C=C, T=T, H=img_train.shape[2], W=img_train.shape[3],
                           img_split=args.img_split, device=device)
    
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        logger.info(f"Using {n_gpu} GPUs for DataParallel training")
    else:
        logger.info("Single GPU mode")
    
    model = model.to(device)
    
    # Modified: Initialize loss function (using learnable temperature)
    criterion = InfoNCELoss(init_temp=args.init_temp, lambda_sup=args.lambda_sup, 
                           hard_negative_weight=args.hard_negative_weight)
    criterion = criterion.to(device)
    
    # Modified: Add criterion parameters to optimizer (temperature parameter uses smaller learning rate)
    
    opt = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': [criterion.logit_scale], 'lr': args.lr * args.temp_lr_ratio}  # Temperature parameter uses smaller learning rate
    ])
    
    logger.info(f"Learnable temperature initialization: τ={args.init_temp}, temperature parameter lr: {args.lr * args.temp_lr_ratio:.6f}")
    
    # Initialize best model tracking
    best_train_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    best_top1_acc = -1.0
    best_top5_acc = -1.0  
    best_metric_type = ""
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Training phase...
        model.train()
        epoch_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"fold{fold_idx} epoch{epoch} training", dynamic_ncols=True)
        
        for eeg, img, category in train_pbar:
            eeg = eeg.to(device, non_blocking=True)
            img = img.to(device, non_blocking=True)
            category = category.to(device, non_blocking=True)
            
            opt.zero_grad()
            
            eeg_out, img_out = model(eeg, img)
            
            loss = criterion(eeg_out, img_out, category).mean()
            loss.backward()
            
            # Compute average gradient update magnitude
            grad_norms = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_norms.append(p.grad.norm(2).item())
            avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
            
            # Also monitor temperature parameter gradient
            temp_grad = criterion.logit_scale.grad.item() if criterion.logit_scale.grad is not None else 0.0
            
            opt.step()
            #scheduler.step() #new
            epoch_loss += loss.item()
            train_pbar.set_postfix({
                "l": f"{loss.item():.3f}",
                "τ": f"{criterion.get_temperature().item():.4f}",  # Display current temperature
                "g": f"{avg_grad_norm:.5f}",
                "τ_g": f"{temp_grad:.4f}",  # Temperature parameter gradient
            })
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Record temperature info at epoch end
        current_temp = criterion.get_temperature().item()
        current_logit_scale = criterion.get_logit_scale().item()
        logger.info(f"fold{fold_idx} epoch{epoch}: Current temperature τ={current_temp:.6f}, logit_scale={current_logit_scale:.4f}")
        
        # Validation phase...
        # Modified: Pass criterion to get dynamic temperature
        acc_top1, acc_top5 = validate_model(
            model, val_loader, device, logger, epoch, fold_idx, criterion
        )

        current_top1 = acc_top1[200]
        current_top5 = acc_top5.get(200, 0)
        
        # Logging...
        logger.info(
            f"fold{fold_idx} epoch{epoch}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_acc_top1={acc_top1[200]:.4f}%, "
            f"val_acc_top5={acc_top5.get(200, 0):.4f}%, "
            f"τ={current_temp:.6f}"  # Add temperature info to log
        )
        
        # Update best model by priority
        is_best = False
        save_reason = ""
        
        if current_top1 > best_top1_acc:
            # Priority 1: top1 accuracy improvement
            is_best = True
            best_top1_acc = current_top1
            best_top5_acc = current_top5      # Also update, keep synchronized
            best_train_loss = avg_train_loss  # Also update
            save_reason = f"New best top1 acc: {current_top1:.4f}% (top5={current_top5:.4f}%, loss={avg_train_loss:.4f}, τ={current_temp:.6f})"
            best_metric_type = "top1_acc"
            
        elif current_top1 == best_top1_acc and current_top5 > best_top5_acc:
            # Priority 2: Same top1, but top5 improved
            is_best = True
            best_top1_acc = current_top1      # Keep synchronized (value same)
            best_top5_acc = current_top5
            best_train_loss = avg_train_loss  # Also update
            save_reason = f"New best top5 acc: {current_top5:.4f}% (top1={current_top1:.4f}%, loss={avg_train_loss:.4f}, τ={current_temp:.6f})"
            best_metric_type = "top5_acc"
            
        elif current_top1 == best_top1_acc and current_top5 == best_top5_acc and avg_train_loss < best_train_loss:
            # Priority 3: Same top1 and top5, but lower train loss
            is_best = True
            best_top1_acc = current_top1      # Keep synchronized
            best_top5_acc = best_top5_acc      # Keep synchronized
            best_train_loss = avg_train_loss
            save_reason = f"New best train loss: {avg_train_loss:.4f} (top1={current_top1:.4f}%, top5={current_top5:.4f}%, τ={current_temp:.6f})"
            best_metric_type = "train_loss"
        
        if is_best:
            best_epoch = epoch
            best_model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            logger.info(f"fold{fold_idx} epoch{epoch}: {save_reason}")
        
        # Early stopping check...
        if early_stopper.step(avg_train_loss):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Training finished, save best model...
    end_time = datetime.now().strftime("%Y%m%d%H%M%S")
    final_path = MODEL_DIR / f"{end_time}_fold{fold_idx}.pth"

    # [DEBUG] Temporarily skip model saving
    save_model = False  # Set to True to resume saving #False
    
    if save_model and best_model_state is not None:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
        
        torch.save({
            "model_state": best_model_state,
            "opt_state": opt.state_dict(),
            "epoch": best_epoch,
            "best_val_loss": best_train_loss,
            "best_val_acc": best_val_acc,
            "n_gpu": n_gpu,
            "train_subs": train_subs,
            "val_sub": val_sub,
            "mode": mode_str,
            "final_temperature": current_temp,  # Save final temperature
            "final_logit_scale": current_logit_scale,  # Save final logit_scale
        }, final_path)
        logger.info(f"Training finished. Best model (epoch={best_epoch}, val_acc={best_top1_acc:.4f}) saved -> {final_path}")
    elif save_model:
        logger.warning("No best model found. Saving last state.")
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            "model_state": state_dict,
            "opt_state": opt.state_dict(),
            "epoch": epoch,
            "n_gpu": n_gpu,
            "train_subs": train_subs,
            "val_sub": val_sub,
            "mode": mode_str,
            "final_temperature": current_temp,
            "final_logit_scale": current_logit_scale,
        }, final_path)

    logger.info(f"Training finished. Best model (epoch={best_epoch}, val_acc={best_top1_acc:.4f})")
    logger.info(f"Training finished. Model saving {'enabled' if save_model else 'DISABLED'}.")

    # Modified: Pass criterion
    validate_model_heatmap(model, val_loader, device, logger, fold_idx, criterion)
    
    return final_path


def main(args):
    # Build subject list according to mode
    if args.mode == "intra":
        assert args.sub is not None, "Intra mode requires --sub"
        train_subs = [args.sub]
        logger = get_logger("train")
        logger.info(f"==== Intra-subject mode: Only training subject {args.sub} ====")
        final_ckpt = train_one_fold(train_subs, None, args, fold_idx=args.sub)

    elif args.mode == "inter_mix":
        all_subs = list(range(1, 11))  # Fixed range 1-10
        logger = get_logger("train")
        logger.info(f"==== Inter-mix mode: Mixed training subjects 1-10 (total {len(all_subs)}) ====")
        final_ckpt = train_one_fold(all_subs, None, args, fold_idx=0)

    elif args.mode == "loso":
        # New LOSO logic: --sub specifies test subject, remaining 1-10 as training set
        assert args.sub is not None, "LOSO mode requires --sub as test subject"
        assert 1 <= args.sub <= 10, f"--sub must be in range 1-10, current: {args.sub}"
        
        all_subs = list(range(1, 11))  # Fixed range 1-10
        val_sub = args.sub
        train_subs = [s for s in all_subs if s != val_sub]
        
        logger = get_logger("train")
        logger.info(f"==== LOSO mode: Test subject {val_sub} | Training subjects {train_subs} (total {len(train_subs)}) ====")
        final_ckpt = train_one_fold(train_subs, val_sub, args, fold_idx=val_sub)
    else:
        raise ValueError("mode must be intra / inter_mix / loso")
    
    return final_ckpt

def auto_test(args, ckpt_path):
    """After training, automatically run test with exactly the same parameters as training"""
    from test import main as test_main
    args.ckpt = str(ckpt_path)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    args.win_len = ckpt.get("win_len", args.win_len)
    test_main(args)

class EarlyStopping:
    """
    Early stopping controller
    Call .step(val_loss) returns True means should stop training
    """
    def __init__(self, patience=10, delta=0.0, logger=None):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best = None
        self.logger = logger

    def step(self, val_loss):
        if self.patience == 0:
            return False

        if self.best is None or val_loss < self.best - self.delta:
            self.best = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.logger:
                self.logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            return self.counter >= self.patience

if __name__ == "__main__":
    import sys, shlex
    parser = argparse.ArgumentParser(
        description="EmotionAligner training (supports intra/inter_mix/loso mode, auto-temperature version)\n"
                    "  - intra mode: --sub specifies training subject\n"
                    "  - loso mode: --sub specifies test subject, remaining 9 subjects (1-10 excluding sub) as training set\n"
                    "  - inter_mix mode: Use subjects 1-10 mixed training")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["intra", "inter_mix", "loso"])
    parser.add_argument("--sub", type=int, 
                        help="Specify subject ID: intra mode as training subject, loso mode as test subject (range: 1-10)")
    parser.add_argument("--sub_lo", type=int, default=1, help="Deprecated, for backward compatibility only")
    parser.add_argument("--sub_hi", type=int, default=10, help="Deprecated, for backward compatibility only")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_split", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--auto_test", action="store_true",
                        help="Immediately test with same parameters after training")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stop if validation loss does not decrease for patience consecutive epochs; set to 0 to disable")
    parser.add_argument("--delta", type=float, default=0.000001,
                        help="Validation loss decrease less than delta considered no improvement")
    
    # New: Auto-temperature related parameters
    parser.add_argument("--init_temp", type=float, default=0.07,
                        help="Initial temperature value (default 0.07, consistent with CLIP)")
    parser.add_argument("--temp_lr_ratio", type=float, default=0.5,
                        help="Temperature parameter learning rate ratio relative to main lr (default 0.1)")
    parser.add_argument("--lambda_sup", type=float, default=0.5,
                        help="Weight of supervised contrastive loss")
    parser.add_argument("--hard_negative_weight", type=float, default=1.0,
                        help="Hard negative weighting coefficient (0=not used, recommended 0.5-1.0)")
    
    args = parser.parse_args()

    cmd = " ".join(shlex.quote(arg) for arg in sys.argv)
    print(f"\n==========  Training Command  ==========")
    print(f"{cmd}")
    print(f"========================================\n")

    final_ckpt = main(args)
    if args.auto_test:
        auto_test(args, final_ckpt)