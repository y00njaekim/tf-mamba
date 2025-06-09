import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("Warning: mamba-ssm is not installed. Please install it with 'pip install mamba-ssm causal-conv1d'.")

# ==========================================================================================
# README: 필수 라이브러리 설치
#
# 이 파일을 실행하기 전에 다음 라이브러리를 설치해야 합니다.
# pip install torch transformers
# pip install mamba-ssm causal-conv1d
# ==========================================================================================


class TemporalFrequencyMambaBlock(nn.Module):
    """
    논문의 핵심 구성 요소인 Temporal-Frequency Mamba Block.
    Temporal-Aware 모듈과 Frequency Filter 모듈을 포함합니다.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba-ssm is not installed. Please install it to use this model.")

        self.d_model = d_model

        # Temporal-Aware Module
        self.temporal_mamba = Mamba(
            d_model=d_model // 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Frequency Filter Module
        self.frequency_mamba = Mamba(
            d_model=d_model // 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # 채널별(feature-wise) 학습 가능한 주파수 필터 임계값 omega
        # FFT 결과의 절반(실수부)만 사용하므로 d_model // 2 크기
        self.omega = nn.Parameter(torch.randn(d_model // 2))

        # Concat 후 차원을 원래대로 복원하기 위한 프로젝션 레이어
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, L, D)

        # 잔차 연결을 위해 입력을 저장
        residual = x

        # 1. Temporal-Aware Module
        # 입력을 시간/주파수 브랜치로 나누기 위해 절반으로 분할
        x_temporal = x[:, :, : self.d_model // 2]
        x_frequency = x[:, :, self.d_model // 2 :]

        temporal_out = self.temporal_mamba(x_temporal)

        # 2. Frequency Filter Module
        # 2-1. FFT
        freq_domain_features = torch.fft.rfft(x_frequency, dim=1)  # (B, L/2+1, D/2)

        # 2-2. Adaptive Low-pass Filter
        power_spectrum = freq_domain_features.abs().pow(2)
        # omega를 (1, 1, D/2)로 브로드캐스팅 가능하게 만듦
        threshold = self.omega.unsqueeze(0).unsqueeze(0)
        mask = (power_spectrum > threshold).to(x.dtype)
        filtered_freq_domain = freq_domain_features * mask

        # 2-3. IFFT
        filtered_time_domain = torch.fft.irfft(filtered_freq_domain, n=x.size(1), dim=1)

        frequency_out = self.frequency_mamba(filtered_time_domain)

        # 3. 출력 결합
        fused_out = torch.cat([temporal_out, frequency_out], dim=-1)

        # 4. 출력 프로젝션 및 잔차 연결
        output = self.out_proj(fused_out) + residual

        return output


class TF_Mamba(nn.Module):
    """
    Zhao et al. 2025 논문의 전체 TF-Mamba 모델 아키텍처
    """

    def __init__(self, num_classes, d_model=1024, n_head=8, num_tf_mamba_blocks=4, d_state=16, d_conv=4, expand=2):
        super().__init__()

        # 1. WavLM-Large 특징 추출기
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        # WavLM의 특징은 학습되지 않도록 고정 (논문에서 명시적으로 언급되지는 않았지만 일반적인 접근 방식)
        for param in self.wavlm.parameters():
            param.requires_grad = False

        # 2. Multi-Head Self-Attention Encoder
        self.self_attention_encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model * 2, dropout=0.1, activation="relu", batch_first=True  # 일반적인 설정
        )

        # 3. Temporal-Frequency Mamba Blocks
        self.tf_mamba_blocks = nn.ModuleList([TemporalFrequencyMambaBlock(d_model, d_state, d_conv, expand) for _ in range(num_tf_mamba_blocks)])

        # 4. Emotion Classifier (MLP)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model // 2, num_classes))

    def forward(self, x_wav, attention_mask=None):
        # x_wav: (B, L_wav)

        # 1. WavLM 특징 추출
        # WavLM은 입력을 받은 후 (B, L, D) 형태의 hidden_states 튜플을 반환. 마지막 레이어 사용.
        with torch.no_grad():
            wavlm_out = self.wavlm(x_wav, attention_mask=attention_mask, output_hidden_states=True)
            x = wavlm_out.last_hidden_state  # (B, L_feat, 1024)

        # 2. Self-Attention Encoder
        x = self.self_attention_encoder(x)

        # 3. TF-Mamba Blocks
        for block in self.tf_mamba_blocks:
            x = block(x)

        # 4. Pooling (Classifier 입력 전)
        # 시퀀스 길이에 대해 평균 풀링을 수행하여 (B, D) 텐서 생성
        pooled_features = x.mean(dim=1)

        # 5. Classifier
        logits = self.classifier(pooled_features)

        # CMDT Loss를 위해 풀링된 특징도 함께 반환
        return logits, pooled_features


if __name__ == "__main__":
    # 모델 테스트
    num_iemocap_classes = 4
    num_meld_classes = 7
    d_model = 1024

    # 더미 입력 데이터
    batch_size = 4
    sequence_length_wav = 16000 * 5  # 5초 길이의 오디오
    dummy_wav = torch.randn(batch_size, sequence_length_wav)

    print("TF-Mamba for IEMOCAP 테스트...")
    model_iemocap = TF_Mamba(num_classes=num_iemocap_classes, d_model=d_model)

    # 모델을 float32로 변환 (WavLM이 기본적으로 float32)
    model_iemocap.float()
    dummy_wav = dummy_wav.float()

    logits, features = model_iemocap(dummy_wav)
    print(f"Logits shape: {logits.shape}")  # (B, num_classes)
    print(f"Features shape: {features.shape}")  # (B, d_model)

    print("\nTF-Mamba for MELD 테스트...")
    model_meld = TF_Mamba(num_classes=num_meld_classes, d_model=d_model)
    model_meld.float()
    logits, features = model_meld(dummy_wav)
    print(f"Logits shape: {logits.shape}")  # (B, num_classes)
    print(f"Features shape: {features.shape}")  # (B, d_model)

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model_iemocap.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params / 1e6:.2f} M")
