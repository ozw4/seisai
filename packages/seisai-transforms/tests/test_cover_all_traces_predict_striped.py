# test_cover_all_traces_predict_striped.py
# - cover_all_traces_predict_striped が「完全被覆(全トレースが各オフセットにつき1回はマスクされる)」を
#   実際の入出力経路で満たしているかを検証します。
# - モデルは恒等写像だが、フォワード入力から「全ゼロ行=マスクされた行」を検出・カウントします。
# - noise_std=0, mask_noise_mode='replace' とすることで、マスク行が 0 に置換される前提で検出します。

import torch
from seisai_transforms.mask_inference import cover_all_traces_predict_striped
from torch import nn

# from your_module import cover_all_traces_predict_striped
# ここでは同一スコープに定義済みであることを想定


class ProbeIdentity(nn.Module):
    """恒等出力しつつ、バッチ内の各サンプル・各トレース行が
    「マスク(=全ゼロ行)」として出現した回数を seen[b,h] に累積する。
    """

    def __init__(self, B: int, H: int) -> None:
        super().__init__()
        self.B = int(B)
        self.H = int(H)
        self.register_buffer('seen', torch.zeros((B, H), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,1,H,W)  ※N = B * (#並列パス)
        assert x.dim() == 4
        assert x.size(2) == self.H
        N = x.size(0)
        # 行が完全に 0 か判定(noise_std=0 & replace 前提)
        row_zero = x[:, 0].abs().sum(dim=-1) == 0  # (N,H) bool
        b_idx = (torch.arange(N, device=x.device) % self.B).to(torch.long)  # (N,)
        add = row_zero.to(torch.float32)  # (N,H)
        # (N,H) を dim=0 で B×H に集約
        buf = torch.zeros_like(self.seen)  # (B,H)
        buf.index_add_(0, b_idx, add)  # 各サンプルへ行方向を加算
        self.seen += buf
        return x  # 恒等


def _run_one_case(H, W, band_width, mask_ratio, offsets, B=2, device='cpu') -> None:
    # 入力を非ゼロにしておく(0埋めマスク検出の前提)
    torch.manual_seed(0)
    x = torch.randn(B, 1, H, W, device=device) + 1.0

    model = ProbeIdentity(B=B, H=H).to(device)
    y = cover_all_traces_predict_striped(
        model,
        x,
        mask_ratio=float(mask_ratio),
        band_width=int(band_width),
        noise_std=0.0,  # 0置換で「マスク=0行」を検出
        mask_noise_mode='replace',
        use_amp=False,
        device=device,
        offsets=tuple(offsets),  # TTAオフセットの数だけ各行がマスクされるはず
        passes_batch=4,
    )
    assert y.shape == x.shape
    # 期待値：各行のヒット回数 == len(offsets)
    expected = float(len(offsets))
    ok = (model.seen == expected).all().item()
    if not ok:
        # 失敗時はどの行が欠落かを簡易表示
        miss = (model.seen != expected).nonzero(as_tuple=False)
        msg = (
            f'coverage failed: H={H},W={W},band_width={band_width},ratio={mask_ratio},offsets={offsets}; '
            f'mismatch rows: {miss.tolist()}'
        )
        raise AssertionError(
            msg
        )


def test_complete_coverage() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cases = [
        # band_width=1(従来相当)
        {'H': 17, 'W': 64, 'band_width': 1, 'mask_ratio': 0.5, 'offsets': (0,)},
        {'H': 33, 'W': 40, 'band_width': 1, 'mask_ratio': 0.3, 'offsets': (0, 1, 2)},
        # 幅付きバンド
        {'H': 64, 'W': 128, 'band_width': 4, 'mask_ratio': 0.25, 'offsets': (0,)},
        {'H': 37, 'W': 96, 'band_width': 5, 'mask_ratio': 0.4, 'offsets': (0, 1)},
        {'H': 13, 'W': 80, 'band_width': 5, 'mask_ratio': 1.0, 'offsets': (0,)},  # 全行一括
        {'H': 48, 'W': 72, 'band_width': 7, 'mask_ratio': 0.6, 'offsets': (0, 1, 2, 3)},
    ]
    for cfg in cases:
        _run_one_case(**cfg, device=device)
    print('✅ complete coverage confirmed for all test cases.')


if __name__ == '__main__':
    test_complete_coverage()
