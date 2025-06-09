import torch
import torch.nn as nn
import torch.nn.functional as F


class CMDTLoss(nn.Module):
    """
    Complex Metric-Distance Triplet (CMDT) Loss.
    논문에 기술된 CMDT Loss를 Supervised Contrastive Learning 방식으로 구현합니다.
    """

    def __init__(self, temperature=0.1):
        """
        Args:
            temperature (float): 점수 분포를 정규화하기 위한 온도 계수(τ).
        """
        super(CMDTLoss, self).__init__()
        self.temperature = temperature

    def _vec_sim(self, u, v):
        """
        논문 수식 (7)의 Vector Cosine Similarity를 계산합니다.
        u, v: (..., N) 형태의 복소수 텐서.
        """
        # u * v.conj()
        uv_dot = torch.sum(u * v.conj(), dim=-1)
        # |u| * |v|
        u_norm = torch.norm(u, p=2, dim=-1)
        v_norm = torch.norm(v, p=2, dim=-1)

        # Re(u * v.conj()) / (|u| * |v|)
        return uv_dot.real / (u_norm * v_norm)

    def forward(self, features, labels):
        """
        Args:
            features (torch.Tensor): 모델로부터 나온 임베딩. (N, D) 형태의 실수 텐서.
            labels (torch.Tensor): 각 임베딩에 해당하는 레이블. (N,) 형태.
        """
        # 1. 실수 특징을 복소수 주파수 도메인으로 변환
        # torch.fft.fft는 복소수 텐서를 반환. (N, D_complex)
        complex_features = torch.fft.fft(features)

        # 2. 유사도 행렬 계산
        batch_size = complex_features.size(0)

        # 각 샘플을 다른 모든 샘플과 비교하기 위해 텐서 확장
        anchor_features = complex_features.unsqueeze(1)  # (N, 1, D_complex)
        contrast_features = complex_features.unsqueeze(0)  # (1, N, D_complex)

        # 모든 쌍에 대한 유사도 계산
        sim_matrix = self._vec_sim(anchor_features, contrast_features)  # (N, N)
        sim_matrix = sim_matrix / self.temperature

        # 3. Supervised Contrastive Loss 마스크 생성
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # 대각선 (자기 자신과의 유사도)은 제외
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask

        # 4. 손실 계산
        # log_prob 계산을 위해 log_softmax 사용
        # log_softmax(x) = x - log(sum(exp(x)))
        # 분모: 모든 contrast 샘플에 대한 exp 합
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        # 분자: 포지티브 샘플에 대한 log_prob의 평균을 계산
        # mask.sum(1)은 각 앵커에 대한 포지티브 샘플의 수
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # 최종 손실: 모든 앵커에 대한 손실의 평균
        loss = -mean_log_prob_pos.mean()

        return loss


if __name__ == "__main__":
    # 테스트를 위한 더미 데이터 생성
    batch_size = 32
    feature_dim = 1024
    num_classes = 4

    dummy_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))

    # CMDT Loss 객체 생성 및 손실 계산
    cmdt_loss_fn = CMDTLoss(temperature=0.1)
    loss = cmdt_loss_fn(dummy_features, dummy_labels)

    print(f"Dummy Features Shape: {dummy_features.shape}")
    print(f"Dummy Labels Shape: {dummy_labels.shape}")
    print(f"Calculated CMDT Loss: {loss.item()}")

    # 레이블이 두 그룹으로만 나뉘는 간단한 케이스
    simple_labels = torch.tensor([0, 0, 1, 1] * (batch_size // 4))
    simple_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    loss_simple = cmdt_loss_fn(simple_features, simple_labels)
    print(f"\nSimple Case CMDT Loss: {loss_simple.item()}")
