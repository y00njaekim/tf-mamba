import os
import re
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset

# ==========================================================================================
# README: 필수 라이브러리 설치
#
# 이 파일을 실행하기 전에 다음 라이브러리를 설치해야 합니다.
# pip install torch torchaudio pandas soundfile
#
# FFmpeg 지원이 필요합니다. 시스템에 FFmpeg을 설치하세요.
# (예: Ubuntu/Debian에서 `sudo apt-get install ffmpeg`)
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
            wav_dir = os.path.join(session_dir, "sentences", "wav")

            if not os.path.isdir(emo_eval_dir):
                continue

            for dirpath, _, filenames in os.walk(emo_eval_dir):
                for emo_file in filenames:
                    if not emo_file.endswith(".txt"):
                        continue

                    file_path = os.path.join(dirpath, emo_file)
                    with open(file_path, "r", encoding="latin-1") as f:
                        lines = f.readlines()

                    for line in lines:
                        if line.startswith("[") and line.strip().endswith("]"):
                            parts = re.split(r"\s+", line.strip())
                            if len(parts) > 4:
                                utt_id = parts[3]
                                emotion = parts[4]

                                if emotion in self.emotions_to_use:
                                    # utt_id (e.g., Ses01F_impro01_F000)에서 대화 이름(e.g., Ses01F_impro01) 추출
                                    dialog_id = "_".join(utt_id.split("_")[:-1])
                                    wav_file = os.path.join(wav_dir, dialog_id, f"{utt_id}.wav")
                                    if os.path.exists(wav_file):
                                        self.samples.append({"path": wav_file, "emotion": self.emotion_map[emotion], "session": session_id})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["path"]
        emotion = sample["emotion"]

        waveform, sample_rate = torchaudio.load(path)

        # 다중 채널 오디오를 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

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
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"'{csv_file}' 파일을 찾을 수 없습니다. 데이터셋 경로를 확인하세요.")
        df = pd.read_csv(csv_file)

        # 데이터 분할(split)에 따라 실제 오디오 파일이 담긴 디렉토리 경로를 설정합니다.
        # 데이터셋의 실제 구조가 README에 명시된 구조와 다를 수 있는 점을 감안합니다.
        if self.split == 'train':
            # 'train_splits' 또는 'train_samples' 디렉토리 확인
            samples_dir_options = [os.path.join(self.root_dir, 'train_splits'), os.path.join(self.root_dir, 'train_samples')]
        elif self.split == 'dev':
            # 'dev_splits_complete' 또는 'dev_samples' 디렉토리 확인
            samples_dir_options = [os.path.join(self.root_dir, 'dev_splits_complete'), os.path.join(self.root_dir, 'dev_samples')]
        elif self.split == 'test':
            # 'test_splits', 'test_samples', 'output_repeated_splits_test' 디렉토리 확인
            samples_dir_options = [os.path.join(self.root_dir, 'test_splits'), os.path.join(self.root_dir, 'test_samples'), os.path.join(self.root_dir, 'output_repeated_splits_test')]
        else:
            samples_dir_options = [os.path.join(self.root_dir, f'{self.split}_samples')]

        samples_dir = None
        for path in samples_dir_options:
            if os.path.exists(path):
                samples_dir = path
                break

        # 오디오 샘플 디렉토리를 찾지 못한 경우 오류 처리
        if samples_dir is None:
            tar_file_path = os.path.join(self.root_dir, f"{self.split}.tar.gz")
            if os.path.exists(tar_file_path):
                raise FileNotFoundError(
                    f"오디오 샘플 디렉토리를 찾을 수 없습니다. 확인한 경로: {samples_dir_options}. "
                    f"'{tar_file_path}' 파일이 존재합니다. 압축을 먼저 해제해야 할 수 있습니다. "
                    f"예: `tar -zxvf {tar_file_path} -C {self.root_dir}`"
                )
            else:
                raise FileNotFoundError(f"오디오 샘플 디렉토리를 찾을 수 없습니다. 확인한 경로: {samples_dir_options}")


        for _, row in df.iterrows():
            dialogue_id = row["Dialogue_ID"]
            utterance_id = row["Utterance_ID"]
            emotion = row["Emotion"]

            if emotion in self.emotion_map:
                # MELD 데이터셋은 .mp4 확장자를 사용하므로, 파일 경로를 .mp4로 우선 탐색
                audio_file_path = os.path.join(samples_dir, f"dia{dialogue_id}_utt{utterance_id}.mp4")
                if not os.path.exists(audio_file_path):
                    # .wav 파일도 확인 (Fallback)
                    audio_file_path = os.path.join(samples_dir, f"dia{dialogue_id}_utt{utterance_id}.wav")

                if os.path.exists(audio_file_path):
                    self.samples.append({"path": audio_file_path, "emotion": self.emotion_map[emotion]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["path"]
        emotion = sample["emotion"]

        try:
            # FFmpeg 백엔드를 사용하여 오디오 로드
            waveform, sample_rate = torchaudio.load(path, backend="ffmpeg")
        except Exception as e:
            print(f"'{path}' 파일 로딩 중 오류 발생: {e}")
            # 오류 발생 시 빈 텐서 반환 또는 다른 예외 처리
            return torch.zeros(1), -1

        # 다중 채널 오디오를 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0), emotion


def collate_fn(batch):
    """
    DataLoader를 위한 collate_fn.
    가변 길이의 오디오 시퀀스를 패딩하여 동일한 길이로 만듭니다.
    """
    # Filter out samples that failed to load
    batch = [item for item in batch if item[1] != -1]
    if not batch:
        # If all samples in the batch failed, return None or empty tensors
        return None, None, None if len(batch) > 0 and len(batch[0]) == 3 else None

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
