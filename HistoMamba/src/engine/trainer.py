# src/engine/trainer.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
import warnings

# Added for enhanced metrics
try:
    from sklearn.metrics import f1_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
    print("Imported metrics from scikit-learn.")
except ImportError:
    print("Warning: scikit-learn not found. Validation metrics will be limited to accuracy.")
    SKLEARN_AVAILABLE = False
    f1_score, precision_score, recall_score = None, None, None


def train_epoch(model, dataloader, criterion_cls, optimizer, device, scaler,
                lambda_recon, epoch_num, total_epochs, writer):
    model.train()
    total_loss, total_cls_loss, total_rec_loss = 0.0, 0.0, 0.0
    correct_predictions, total_samples = 0, 0
    criterion_rec = nn.MSELoss()
    num_batches = len(dataloader)
    print_interval = max(1, num_batches // 10)

    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            cls_loss = criterion_cls(outputs['classification'], labels)
            loss = cls_loss
            rec_loss_val = 0.0
            if lambda_recon > 0 and 'reconstruction' in outputs:
                rec_loss = criterion_rec(outputs['reconstruction'], images)
                loss = loss + lambda_recon * rec_loss
                rec_loss_val = rec_loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        if lambda_recon > 0: total_rec_loss += rec_loss_val * batch_size

        _, predicted = torch.max(outputs['classification'], 1)
        total_samples += batch_size
        correct_predictions += (predicted == labels).sum().item()

        if (i + 1) % print_interval == 0 or (i + 1) == num_batches:
            current_avg_loss = total_loss / total_samples if total_samples > 0 else 0
            current_acc = (100.0 * correct_predictions / total_samples) if total_samples > 0 else 0.0
            print(f"  Epoch {epoch_num}/{total_epochs} | Batch {i+1}/{num_batches} | Loss: {current_avg_loss:.4f} | Acc: {current_acc:.2f}%", end='\r')
    print() # Newline

    avg_loss = total_loss / total_samples if total_samples else 0
    avg_cls_loss = total_cls_loss / total_samples if total_samples else 0
    avg_rec_loss = total_rec_loss / total_samples if total_samples and lambda_recon > 0 else 0
    accuracy = 100.0 * correct_predictions / total_samples if total_samples else 0

    if writer:
        writer.add_scalar('Loss/Train_Total', avg_loss, epoch_num)
        writer.add_scalar('Loss/Train_Classification', avg_cls_loss, epoch_num)
        if lambda_recon > 0: writer.add_scalar('Loss/Train_Reconstruction', avg_rec_loss, epoch_num)
        writer.add_scalar('Accuracy/Train', accuracy, epoch_num)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch_num)

    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion_cls, device, lambda_recon,
                   epoch_num, num_classes, writer):
    model.eval()
    total_loss, total_cls_loss, total_rec_loss = 0.0, 0.0, 0.0
    correct_predictions, total_samples = 0, 0
    criterion_rec = nn.MSELoss()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                cls_loss = criterion_cls(outputs['classification'], labels)
                loss = cls_loss
                rec_loss_val = 0.0
                if lambda_recon > 0 and 'reconstruction' in outputs:
                    rec_loss = criterion_rec(outputs['reconstruction'], images)
                    loss = loss + lambda_recon * rec_loss
                    rec_loss_val = rec_loss.item()
            
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            if lambda_recon > 0: total_rec_loss += rec_loss_val * batch_size

            _, predicted = torch.max(outputs['classification'], 1)
            total_samples += batch_size
            correct_predictions += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if total_samples == 0:
        print("Warning: No samples in validation dataloader.")
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct_predictions / total_samples
    
    f1, precision, recall = 0.0, 0.0, 0.0
    if SKLEARN_AVAILABLE and num_classes > 1:
        avg_mode = 'macro'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                f1 = f1_score(all_labels, all_preds, average=avg_mode, zero_division=0)
                precision = precision_score(all_labels, all_preds, average=avg_mode, zero_division=0)
                recall = recall_score(all_labels, all_preds, average=avg_mode, zero_division=0)
            except ValueError as e: print(f"Warning: sklearn metrics error: {e}")

    if writer:
        writer.add_scalar('Loss/Validation_Total', avg_loss, epoch_num)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch_num)
        if SKLEARN_AVAILABLE and num_classes > 1:
            writer.add_scalar('Metrics/Validation_F1_Macro', f1, epoch_num)
            writer.add_scalar('Metrics/Validation_Precision_Macro', precision, epoch_num)
            writer.add_scalar('Metrics/Validation_Recall_Macro', recall, epoch_num)

    return avg_loss, accuracy, f1, precision, recall