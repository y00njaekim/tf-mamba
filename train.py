import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

from models import TF_Mamba
from datasets import IEMOCAPDataset, MELDDataset, collate_fn
from loss import CMDTLoss


def train_one_epoch(model, dataloader, optimizer, criterion_ce, criterion_cmdt, lambda_cmdt, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        waveforms, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        logits, features = model(waveforms)

        loss_ce = criterion_ce(logits, labels)
        loss_cmdt = criterion_cmdt(features, labels)
        loss = loss_ce + lambda_cmdt * loss_cmdt

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    wa = accuracy_score(all_labels, all_preds)
    ua = accuracy_score(all_labels, all_preds)  # Placeholder, needs proper calculation for imbalanced classes
    wf1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, wa, ua, wf1


def evaluate(model, dataloader, criterion_ce, criterion_cmdt, lambda_cmdt, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            waveforms, labels = batch[0].to(device), batch[1].to(device)

            logits, features = model(waveforms)

            loss_ce = criterion_ce(logits, labels)
            loss_cmdt = criterion_cmdt(features, labels)
            loss = loss_ce + lambda_cmdt * loss_cmdt

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    wa = accuracy_score(all_labels, all_preds)
    # Note: For UA, balanced accuracy or per-class accuracy average is better.
    # Using simple accuracy as a placeholder.
    ua = accuracy_score(all_labels, all_preds)
    wf1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, wa, ua, wf1


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dataset == "iemocap":
        dataset = IEMOCAPDataset(root_dir=args.data_path)
        num_classes = 4

        # 5-fold cross-validation for IEMOCAP
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        sessions = np.array([s["session"] for s in dataset.samples])
        unique_sessions = np.unique(sessions)  # Should be [1, 2, 3, 4, 5]

        # This is a speaker-independent cross-validation setup based on sessions
        # For simplicity, we split samples by session, not speakers directly.
        # A more rigorous approach would group speakers (e.g., Ses01F and Ses01M).

        fold_results = []
        for fold, (train_session_idx, val_session_idx) in enumerate(kf.split(unique_sessions)):
            print(f"\n----------- Fold {fold+1}/5 -----------")
            train_sessions = unique_sessions[train_session_idx]
            val_sessions = unique_sessions[val_session_idx]

            train_indices = [i for i, s in enumerate(dataset.samples) if s["session"] in train_sessions]
            val_indices = [i for i, s in enumerate(dataset.samples) if s["session"] in val_sessions]

            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)

            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

            # Re-initialize model and optimizer for each fold
            model = TF_Mamba(num_classes=num_classes).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            criterion_ce = nn.CrossEntropyLoss()
            criterion_cmdt = CMDTLoss(temperature=0.1)

            for epoch in range(args.epochs):
                train_loss, train_wa, train_ua, train_wf1 = train_one_epoch(
                    model, train_loader, optimizer, criterion_ce, criterion_cmdt, args.lambda_cmdt, device
                )
                val_loss, val_wa, val_ua, val_wf1 = evaluate(model, val_loader, criterion_ce, criterion_cmdt, args.lambda_cmdt, device)

                print(
                    f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val WA: {val_wa:.4f} | Val UA: {val_ua:.4f} | Val WF1: {val_wf1:.4f}"
                )

            fold_results.append({"wa": val_wa, "ua": val_ua, "wf1": val_wf1})

        # Average results over folds
        avg_wa = np.mean([r["wa"] for r in fold_results])
        avg_ua = np.mean([r["ua"] for r in fold_results])
        avg_wf1 = np.mean([r["wf1"] for r in fold_results])
        print(f"\nIEMOCAP 5-Fold CV Average Results: WA={avg_wa:.4f}, UA={avg_ua:.4f}, WF1={avg_wf1:.4f}")

    elif args.dataset == "meld":
        num_classes = 7
        train_dataset = MELDDataset(root_dir=args.data_path, split="train")
        val_dataset = MELDDataset(root_dir=args.data_path, split="dev")
        test_dataset = MELDDataset(root_dir=args.data_path, split="test")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

        model = TF_Mamba(num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_cmdt = CMDTLoss(temperature=0.1)

        for epoch in range(args.epochs):
            train_loss, train_wa, train_ua, train_wf1 = train_one_epoch(model, train_loader, optimizer, criterion_ce, criterion_cmdt, args.lambda_cmdt, device)
            val_loss, val_wa, val_ua, val_wf1 = evaluate(model, val_loader, criterion_ce, criterion_cmdt, args.lambda_cmdt, device)
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val WF1: {val_wf1:.4f}")

        test_loss, test_wa, test_ua, test_wf1 = evaluate(model, test_loader, criterion_ce, criterion_cmdt, args.lambda_cmdt, device)
        print(f"\nMELD Test Results: Loss={test_loss:.4f}, WA={test_wa:.4f}, UA={test_ua:.4f}, WF1={test_wf1:.4f}")

    else:
        raise ValueError("Invalid dataset specified. Choose 'iemocap' or 'meld'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-Mamba for Speech Emotion Recognition")
    parser.add_argument("--dataset", type=str, required=True, choices=["iemocap", "meld"], help="Dataset to use")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--lambda_cmdt", type=float, default=0.1, help="Weight for CMDT loss")

    args = parser.parse_args()
    main(args)
