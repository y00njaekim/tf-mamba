import os
import re
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

# ==========================================================================================
# README: 필수 라이브러리 설치
#
# 이 파일을 실행하기 전에 다음 라이브러리를 설치해야 합니다.
# pip install torch torchaudio pandas
# ==========================================================================================

# ==========================================================================================
# README: 데이터셋 구조
#
# 이 코드에서는 IEMOCAP과 MELD 데이터셋을 사용합니다.
# 각 데이터셋은 아래와 같은 디렉토리 구조를 따라야 합니다.
#
# 1. IEMOCAP 데이터셋 구조:
#
# /path/to/IEMOCAP_full_release/
# ├── Session1/
# │   ├── dialog/
# │   │   ├── wav/
# │   │   │   ├── Ses01F_impro01.wav
# │   │   │   ├── Ses01F_impro01_F000.wav
# │   │   │   └── ...
# │   │   └── EmoEvaluation/
# │   │       ├── Ses01F_impro01.txt
# │   │       └── ...
# │   └── ...
# ├── Session2/
# │   └── ...
# ├── Session3/
# │   └── ...
# ├── Session4/
# │   └── ...
# └── Session5/
#     └── ...
#
# 2. MELD 데이터셋 구조:
#
# /path/to/MELD.Raw/
# ├── train_samples/
# │   ├── dia0_utt0.wav
# │   └── ...
# ├── dev_samples/
# │   ├── dia0_utt0.wav
# │   └── ...
# ├── test_samples/
# │   ├── dia0_utt0.wav
# │   └── ...
# ├── train_sent_emo.csv
# ├── dev_sent_emo.csv
# └── test_sent_emo.csv
#
# ==========================================================================================


class IEMOCAPDataset(Dataset):
    """IEMOCAP 데이터셋을 위한 Dataset 클래스"""

    def __init__(self, root_dir, target_sample_rate=16000):
        self.root_dir = root_dir
        self.target_sample_rate = target_sample_rate
        self.samples = []
        self.emotion_map = {"neu": 0, "hap": 1, "ang": 2, "sad": 3, "exc": 1}  # 'excited' is mapped to 'happy'
        self.emotions_to_use = ["neu", "hap", "ang", "sad", "exc"]

        self._prepare_data()

    def _prepare_data(self):
        for session_id in range(1, 6):
            session_dir = os.path.join(self.root_dir, f"Session{session_id}")
            emo_eval_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
            wav_dir = os.path.join(session_dir, "dialog", "wav")

            for emo_file in os.listdir(emo_eval_dir):
                if not emo_file.endswith(".txt"):
                    continue

                with open(os.path.join(emo_eval_dir, emo_file), "r") as f:
                    lines = f.readlines()

                for line in lines:
                    if line.startswith("[") and line.strip().endswith("]"):
                        parts = re.split(r"\s+", line.strip())
                        if len(parts) > 4:
                            utt_id = parts[3]
                            emotion = parts[4]

                            if emotion in self.emotions_to_use:
                                wav_file = os.path.join(wav_dir, f"{utt_id}.wav")
                                if os.path.exists(wav_file):
                                    self.samples.append({"path": wav_file, "emotion": self.emotion_map[emotion], "session": session_id})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["path"]
        emotion = sample["emotion"]

        waveform, sample_rate = torchaudio.load(path)

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0), emotion, sample["session"]


class MELDDataset(Dataset):
    """MELD 데이터셋을 위한 Dataset 클래스"""

    def __init__(self, root_dir, split, target_sample_rate=16000):
        self.root_dir = root_dir
        self.split = split
        self.target_sample_rate = target_sample_rate
        self.samples = []
        # MELD 7가지 감정: anger, disgust, fear, joy, neutral, sadness, surprise
        self.emotion_map = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}

        self._prepare_data()

    def _prepare_data(self):
        csv_file = os.path.join(self.root_dir, f"{self.split}_sent_emo.csv")
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            dialogue_id = row["Dialogue_ID"]
            utterance_id = row["Utterance_ID"]
            emotion = row["Emotion"]

            if emotion in self.emotion_map:
                wav_file = os.path.join(self.root_dir, f"{self.split}_samples", f"dia{dialogue_id}_utt{utterance_id}.wav")
                if os.path.exists(wav_file):
                    self.samples.append({"path": wav_file, "emotion": self.emotion_map[emotion]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["path"]
        emotion = sample["emotion"]

        waveform, sample_rate = torchaudio.load(path)

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0), emotion


def collate_fn(batch):
    """
    DataLoader를 위한 collate_fn.
    가변 길이의 오디오 시퀀스를 패딩하여 동일한 길이로 만듭니다.
    """
    waveforms, emotions, sessions = [], [], []
    is_iemocap = len(batch[0]) == 3

    for item in batch:
        waveforms.append(item[0])
        emotions.append(item[1])
        if is_iemocap:
            sessions.append(item[2])

    # Pad waveforms to the same length
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0)
    emotions = torch.tensor(emotions, dtype=torch.long)

    if is_iemocap:
        sessions = torch.tensor(sessions, dtype=torch.long)
        return padded_waveforms, emotions, sessions
    else:
        return padded_waveforms, emotions


if __name__ == "__main__":
    # 사용 예시 (데이터셋 경로를 실제 경로로 수정해야 합니다)
    iemocap_path = "/path/to/IEMOCAP_full_release"
    meld_path = "/path/to/MELD.Raw"

    print("IEMOCAP 데이터셋 테스트...")
    if os.path.exists(iemocap_path) and os.path.isdir(os.path.join(iemocap_path, "Session1")):
        try:
            iemocap_dataset = IEMOCAPDataset(root_dir=iemocap_path)
            print(f"IEMOCAP 샘플 수: {len(iemocap_dataset)}")
            waveform, emotion, session = iemocap_dataset[0]
            print(f"첫 번째 샘플: waveform shape={waveform.shape}, emotion={emotion}, session={session}")
        except Exception as e:
            print(f"IEMOCAP 데이터셋 로딩 중 오류 발생: {e}")
    else:
        print(f"IEMOCAP 경로를 찾을 수 없습니다: {iemocap_path}")

    print("\nMELD 데이터셋 테스트...")
    if os.path.exists(meld_path) and os.path.exists(os.path.join(meld_path, "train_sent_emo.csv")):
        try:
            meld_dataset = MELDDataset(root_dir=meld_path, split="train")
            print(f"MELD (train) 샘플 수: {len(meld_dataset)}")
            waveform, emotion = meld_dataset[0]
            print(f"첫 번째 샘플: waveform shape={waveform.shape}, emotion={emotion}")

            from torch.utils.data import DataLoader

            loader = DataLoader(meld_dataset, batch_size=4, collate_fn=collate_fn)
            batch_waveforms, batch_emotions = next(iter(loader))
            print(f"DataLoader 배치 테스트: waveforms shape={batch_waveforms.shape}, emotions shape={batch_emotions.shape}")

        except Exception as e:
            print(f"MELD 데이터셋 로딩 중 오류 발생: {e}")
    else:
        print(f"MELD 경로를 찾을 수 없습니다: {meld_path}")
