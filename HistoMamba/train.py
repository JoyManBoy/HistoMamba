# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os
import sys
from collections import Counter

# --- Import from our source modules ---
from src.data.dataset import HistoPatchDataset
from src.data.utils import build_global_label_map
from src.models.histomamba import HistoMamba
from src.engine.trainer import train_epoch, validate_epoch

# Basic setup
torch.backends.cudnn.benchmark = True

def main(args):
    # --- Basic Setup ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_amp = not args.no_amp and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | AMP Enabled: {use_amp}")

    manifest_cache_dir = os.path.join(args.output_dir, "manifest_cache")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(manifest_cache_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    writer = SummaryWriter(log_dir=args.output_dir)

    # --- Data ID Loading and Splitting ---
    if args.id_list is None:
        try:
            all_dirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
            args.id_list = sorted([d for d in all_dirs if d != args.patch_dir_name and not d.startswith('.') and d != os.path.basename(manifest_cache_dir)])
            if not args.id_list: raise FileNotFoundError("No sample subdirectories found.")
            print(f"Found {len(args.id_list)} potential sample IDs.")
        except Exception as e:
            writer.close(); sys.exit(f"Error inferring ID list: {e}")

    global_label_map, label_col, metadata = build_global_label_map(args.id_list, args.metadata_csv)
    num_classes = len(global_label_map)
    if num_classes == 0: writer.close(); sys.exit("Error: Could not determine classes.")

    np.random.shuffle(args.id_list)
    val_count = int(args.val_split * len(args.id_list)) if args.val_split > 0 else 0
    train_ids, val_ids = args.id_list[val_count:], args.id_list[:val_count]
    print(f"\nSplitting Sample IDs: Train={len(train_ids)}, Validation={len(val_ids)}")

    # --- Create Datasets & DataLoaders ---
    try:
        train_dataset = HistoPatchDataset(args.data_dir, train_ids, img_size=args.img_size, augment=(not args.no_augment),
                                          global_label_map=global_label_map, global_num_classes=num_classes,
                                          label_source_column=label_col, loaded_metadata=metadata,
                                          patch_dir_name=args.patch_dir_name, cache_dir=manifest_cache_dir)
        val_dataset = HistoPatchDataset(args.data_dir, val_ids, img_size=args.img_size, augment=False,
                                        global_label_map=global_label_map, global_num_classes=num_classes,
                                        label_source_column=label_col, loaded_metadata=metadata,
                                        patch_dir_name=args.patch_dir_name, cache_dir=manifest_cache_dir) if val_ids else None
    except Exception as e:
        writer.close(); sys.exit(f"Error creating datasets: {e}")
    if len(train_dataset) == 0: writer.close(); sys.exit("Error: Training dataset has 0 patches.")

    train_sampler = None
    if args.use_weighted_sampler and num_classes > 1:
        train_labels_str = train_dataset.get_labels()
        if train_labels_str:
            train_labels_idx = [global_label_map.get(s, global_label_map['unknown']) for s in train_labels_str]
            class_counts = Counter(train_labels_idx)
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label_idx] for label_idx in train_labels_idx]
            train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            print("WeightedRandomSampler enabled for training.")

    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'), drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == 'cuda'), persistent_workers=persistent_workers) if val_dataset and len(val_dataset) > 0 else None

    # --- Model, Loss, Optimizer ---
    try:
        model = HistoMamba(num_classes=num_classes, img_size=args.img_size, use_lpu=args.use_lpu,
                               drop_path_rate=args.drop_path_rate, d_state=args.mamba_d_state,
                               d_conv=args.mamba_d_conv, expand=args.mamba_expand).to(device)
        print(f"\nModel params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except Exception as e:
        writer.close(); sys.exit(f"Error initializing model: {e}")

    criterion_cls = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler(enabled=use_amp)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    best_monitor_value = -float('inf') if args.monitor_metric in ['val_accuracy', 'val_f1'] else float('inf')
    best_epoch, epochs_no_improve = -1, 0

    for epoch in range(args.num_epochs):
        epoch_num = epoch + 1
        print(f"\n--- Epoch {epoch_num}/{args.num_epochs} --- LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_dataset.reset_load_error_count()
        train_loss, train_acc = train_epoch(model, train_loader, criterion_cls, optimizer, device, scaler,
                                            args.lambda_recon, epoch_num, args.num_epochs, writer)
        print(f"Train => Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

        if val_loader:
            val_dataset.reset_load_error_count()
            val_loss, val_acc, val_f1, _, _ = validate_epoch(model, val_loader, criterion_cls, device,
                                                            args.lambda_recon, epoch_num, num_classes, writer)
            metric_str = f" | F1: {val_f1:.4f}" if num_classes > 1 else ""
            print(f"Valid => Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%{metric_str}")
            
            current_monitor_val = val_loss if args.monitor_metric == 'val_loss' else (val_f1 if args.monitor_metric == 'val_f1' else val_acc)
            is_better = (current_monitor_val > best_monitor_value + args.early_stopping_min_delta) if args.monitor_metric != 'val_loss' else (current_monitor_val < best_monitor_value - args.early_stopping_min_delta)

            if is_better:
                best_monitor_value, best_epoch, epochs_no_improve = current_monitor_val, epoch_num, 0
                checkpoint_path = os.path.join(args.output_dir, "best_model_patch.pth")
                torch.save({'epoch': epoch_num, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(), 'global_label_map': global_label_map, 'args': args}, checkpoint_path)
                print(f"Checkpoint saved (Best {args.monitor_metric}): {best_monitor_value:.4f}")
            elif args.early_stopping_patience > 0:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch_num} epochs.")
                    break
        
        scheduler.step()

    print("\n--- Training Finished ---")
    writer.close()
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train HistoMamba patch classifier.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Add all your arguments here, they are unchanged from the original script
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--id_list", type=str, nargs='+', default=None)
    parser.add_argument("--output_dir", type=str, default="./histomamba_output")
    parser.add_argument("--patch_dir_name", type=str, default="patches")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--use_lpu", action='store_true')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--use_weighted_sampler", action='store_true')
    parser.add_argument("--no_augment", action='store_true')
    parser.add_argument("--lambda_recon", type=float, default=0.05)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--monitor_metric", type=str, default="val_loss", choices=["val_loss", "val_accuracy", "val_f1"])
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_gument("--early_stopping_min_delta", type=float, default=0.001)
    parser.add_argument("--no_amp", action='store_true')
    
    args = parser.parse_args()
    main(args)