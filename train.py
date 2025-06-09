import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from models import TF_Mamba
from datasets import IEMOCAPDataset, MELDDataset, collate_fn
from loss import CMDTLoss

# Set multiprocessing start method to 'spawn' for CUDA safety
try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass


def save_checkpoint(state, is_best, checkpoint_dir, filename="last_checkpoint.pth.tar"):
    """모델 체크포인트를 저장하고, 최고 성능 모델을 별도로 복사합니다."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, "model_best.pth.tar"))


def log_to_file(message, log_file):
    """지정된 파일에 로그 메시지를 기록합니다."""
    with open(log_file, "a") as f:
        f.write(message + "\n")


def train_one_epoch(model, dataloader, optimizer, criterion_ce, criterion_cmdt, lambda_cmdt, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training"):
        if batch is None or batch[0] is None:
            continue
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
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None or batch[0] is None:
                continue
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

    # Setup logging
    log_to_file(f"실험 시작: {args}", args.log_file)

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
            fold_num = fold + 1
            print(f"\n----------- Fold {fold_num}/5 -----------")
            log_to_file(f"\n----------- Fold {fold_num}/5 -----------", args.log_file)
            train_sessions = unique_sessions[train_session_idx]
            val_sessions = unique_sessions[val_session_idx]

            train_indices = [i for i, s in enumerate(dataset.samples) if s["session"] in train_sessions]
            val_indices = [i for i, s in enumerate(dataset.samples) if s["session"] in val_sessions]

            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)

            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
            
            fold_checkpoint_dir = os.path.join(args.checkpoint_dir, f"iemocap_fold_{fold_num}")

            model = TF_Mamba(num_classes=num_classes).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            criterion_ce = nn.CrossEntropyLoss()
            criterion_cmdt = CMDTLoss(temperature=0.1)
            
            best_val_loss = float("inf")

            for epoch in range(args.epochs):
                epoch_num = epoch + 1
                train_loss, train_wa, train_ua, train_wf1 = train_one_epoch(
                    model, train_loader, optimizer, criterion_ce, criterion_cmdt, args.lambda_cmdt, device
                )
                val_loss, val_wa, val_ua, val_wf1 = evaluate(model, val_loader, criterion_ce, criterion_cmdt, args.lambda_cmdt, device)

                log_message = f"Epoch {epoch_num}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val WA: {val_wa:.4f} | Val UA: {val_ua:.4f} | Val WF1: {val_wf1:.4f}"
                print(log_message)
                log_to_file(log_message, args.log_file)
                
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                save_checkpoint({
                    'epoch': epoch_num,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, is_best, fold_checkpoint_dir)

            # Load best model for final evaluation
            best_model_path = os.path.join(fold_checkpoint_dir, "model_best.pth.tar")
            if os.path.isfile(best_model_path):
                print(f"\nFold {fold_num} 최고 성능 모델 로딩 중 (val_loss: {best_val_loss:.4f})...")
                checkpoint = torch.load(best_model_path)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print("\n최고 성능 모델을 찾지 못해, 마지막 시점의 모델로 평가합니다.")

            final_val_loss, final_val_wa, final_val_ua, final_val_wf1 = evaluate(model, val_loader, criterion_ce, criterion_cmdt, args.lambda_cmdt, device)
            final_log_message = f"Fold {fold_num} 최종 성능: Val WA={final_val_wa:.4f}, Val UA={final_val_ua:.4f}, Val WF1={final_val_wf1:.4f}"
            print(final_log_message)
            log_to_file(final_log_message, args.log_file)

            fold_results.append({"wa": final_val_wa, "ua": final_val_ua, "wf1": final_val_wf1})

        # Average results over folds
        avg_wa = np.mean([r["wa"] for r in fold_results])
        avg_ua = np.mean([r["ua"] for r in fold_results])
        avg_wf1 = np.mean([r["wf1"] for r in fold_results])
        avg_log_message = f"\nIEMOCAP 5-Fold CV 평균 결과: WA={avg_wa:.4f}, UA={avg_ua:.4f}, WF1={avg_wf1:.4f}"
        print(avg_log_message)
        log_to_file(avg_log_message, args.log_file)

    elif args.dataset == "meld":
        num_classes = 7
        train_dataset = MELDDataset(root_dir=args.data_path, split="train")
        val_dataset = MELDDataset(root_dir=args.data_path, split="dev")
        test_dataset = MELDDataset(root_dir=args.data_path, split="test")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        
        meld_checkpoint_dir = os.path.join(args.checkpoint_dir, "meld")

        model = TF_Mamba(num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_cmdt = CMDTLoss(temperature=0.1)
        
        best_val_loss = float("inf")

        for epoch in range(args.epochs):
            epoch_num = epoch + 1
            train_loss, train_wa, train_ua, train_wf1 = train_one_epoch(model, train_loader, optimizer, criterion_ce, criterion_cmdt, args.lambda_cmdt, device)
            val_loss, val_wa, val_ua, val_wf1 = evaluate(model, val_loader, criterion_ce, criterion_cmdt, args.lambda_cmdt, device)
            
            log_message = f"Epoch {epoch_num}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val WF1: {val_wf1:.4f}"
            print(log_message)
            log_to_file(log_message, args.log_file)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            save_checkpoint({
                'epoch': epoch_num,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, is_best, meld_checkpoint_dir)

        # Load best model for testing
        print("\n최고 성능 모델을 로드하여 테스트를 진행합니다...")
        best_model_path = os.path.join(meld_checkpoint_dir, "model_best.pth.tar")
        if os.path.isfile(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"'{best_model_path}' 에서 최고 성능 모델 로드 완료")
            log_to_file(f"테스트를 위해 최고 성능 모델 로드: {best_model_path}", args.log_file)
        else:
            print("최고 성능 모델을 찾지 못했습니다. 마지막 시점의 모델로 테스트를 진행합니다.")
            log_to_file("최고 성능 모델을 찾지 못했습니다. 마지막 시점의 모델로 테스트를 진행합니다.", args.log_file)


        test_loss, test_wa, test_ua, test_wf1 = evaluate(model, test_loader, criterion_ce, criterion_cmdt, args.lambda_cmdt, device)
        test_log_message = f"\nMELD Test 결과: Loss={test_loss:.4f}, WA={test_wa:.4f}, UA={test_ua:.4f}, WF1={test_wf1:.4f}"
        print(test_log_message)
        log_to_file(test_log_message, args.log_file)

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
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='모델 체크포인트를 저장할 디렉토리')
    parser.add_argument('--log_file', type=str, default='training.log', help='로그를 저장할 텍스트 파일')

    args = parser.parse_args()
    main(args)
