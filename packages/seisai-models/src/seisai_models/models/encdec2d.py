from dataclasses import dataclass
from typing import Literal

import timm
import torch
import torch.nn.functional as F
from seisai_models.nn.anti_alias import BlurPool2d
from seisai_models.nn.decoder import UnetDecoder2d
from seisai_models.nn.heads import SegmentationHead2d
from seisai_models.ops.stride_pad import override_stage_strides
from torch import nn


@dataclass(frozen=True)
class _SkipLayoutEntry:
    source: Literal['backbone', 'prestage']
    source_shallow_index: int
    decoder_skip_index: int
    channels: int
    enabled: bool


@dataclass(frozen=True)
class _DecoderSkipSlot:
    source: Literal['extra', 'backbone', 'prestage']
    decoder_skip_index: int
    channels: int
    source_shallow_index: int | None
    enabled: bool


class EncDec2D(nn.Module):
    """timm バックボーン + U-Net デコーダの汎用ネットワーク。.

    timm バックボーンの前に Conv+BN+ReLU の前段ステージを任意段数挿入できる。
    SAME などの動的パディングはモデル外(データローダ側)で行うこと。

    """

    def __init__(
        self,
        backbone: str,
        in_chans: int = 1,
        out_chans: int = 1,
        pretrained: bool = True,
        stage_strides: list[tuple[int, int]] | None = None,
        extra_stages: int = 0,
        extra_stage_strides: tuple[tuple[int, int], ...] | None = None,
        extra_stage_channels: tuple[int, ...] | None = None,
        extra_stage_use_bn: bool = True,
        pre_stages: int = 0,
        pre_stage_strides: tuple[tuple[int, int], ...] | None = None,
        pre_stage_kernels: tuple[int, ...] | None = None,
        pre_stage_channels: tuple[int, ...] | None = None,
        pre_stage_use_bn: bool = True,
        pre_stage_antialias: bool = False,
        pre_stage_aa_taps: int = 3,
        pre_stage_aa_pad_mode: str = 'zeros',
        disable_prestage_skip_indices: tuple[int, ...] | list[int] = (),
        disable_backbone_skip_indices: tuple[int, ...] | list[int] = (),
        # decoder オプション
        decoder_channels: tuple = (256, 128, 64, 32),
        decoder_scales: tuple = (2, 2, 2, 2),
        upsample_mode: str = 'bilinear',
        attention_type: str = 'scse',
        intermediate_conv: bool = True,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.disable_prestage_skip_indices = self._normalize_disable_skip_indices(
            disable_prestage_skip_indices,
            name='disable_prestage_skip_indices',
        )
        self.disable_backbone_skip_indices = self._normalize_disable_skip_indices(
            disable_backbone_skip_indices,
            name='disable_backbone_skip_indices',
        )
        if pre_stage_antialias and pre_stage_aa_pad_mode != 'zeros':
            raise ValueError(
                f'Unsupported pre_stage_aa_pad_mode: {pre_stage_aa_pad_mode}. '
                'Use "zeros".'
            )
        # 前段のダウンサンプル
        self.pre_down = nn.ModuleList()
        self.pre_out_channels = []  # ★追加：各pre段の出力chリスト
        c_in = in_chans
        kernels = list(pre_stage_kernels or [])
        strides = list(pre_stage_strides or [])
        channels = list(pre_stage_channels or [])
        for i in range(pre_stages):
            k = kernels[i] if i < len(kernels) else 3
            s = strides[i] if i < len(strides) else (1, 1)
            stride_h = None
            stride_w = None
            if isinstance(s, int):
                stride_h, stride_w = s, s
            elif isinstance(s, (tuple, list)) and len(s) == 2:
                stride_h, stride_w = s
            if stride_h is not None and stride_w is not None:
                stride_h = int(stride_h)
                stride_w = int(stride_w)
                s_norm = (stride_h, stride_w)
            else:
                s_norm = s
            p = k // 2
            c_out = channels[i] if i < len(channels) else c_in
            block = []
            if pre_stage_antialias:
                if stride_h is None or stride_w is None:
                    msg = f'Invalid pre_stage stride: {s}'
                    raise ValueError(msg)
                if stride_w > 1:
                    if stride_h != 1:
                        msg = f'pre_stage stride_h must be 1, got {stride_h}'
                        raise ValueError(msg)
                    block.append(
                        BlurPool2d(
                            taps=pre_stage_aa_taps,
                            stride_w=stride_w,
                            pad_mode=pre_stage_aa_pad_mode,
                        )
                    )
                    conv_stride = (1, 1)
                else:
                    conv_stride = s_norm
            else:
                conv_stride = s_norm
            block.append(
                nn.Conv2d(c_in, c_out, k, stride=conv_stride, padding=p, bias=False)
            )
            if pre_stage_use_bn:
                block.append(nn.BatchNorm2d(c_out))
            block.append(nn.ReLU(inplace=True))
            self.pre_down.append(nn.Sequential(*block))
            self.pre_out_channels.append(c_out)  # ★控える
            c_in = c_out
        pre_out_ch = c_in
        # Encoder (timm features_only)
        self.backbone = timm.create_model(
            backbone,
            in_chans=pre_out_ch,
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=0.0,
        )
        if stage_strides is not None:
            override_stage_strides(self.backbone, stage_strides)

        # 追加のダウンサンプル段
        self.extra_down = nn.ModuleList()
        # 1) backbone のチャンネル列(深い→浅い)
        ecs_base = [fi['num_chs'] for fi in self.backbone.feature_info][::-1]

        # 2) extra_down を作る(最深の上に積む)
        self.extra_down = nn.ModuleList()
        c_in = ecs_base[0] if ecs_base else 0
        extra_out_channels: list[int] = []

        extra_strides = list(extra_stage_strides or [])
        if len(extra_strides) < extra_stages:
            extra_strides += [(2, 2)] * (extra_stages - len(extra_strides))
        extra_channels = list(extra_stage_channels or [])

        for i in range(extra_stages):
            stride = extra_strides[i]
            c_out = extra_channels[i] if i < len(extra_channels) else c_in
            block = [
                nn.Conv2d(
                    c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False
                )
            ]
            if extra_stage_use_bn:
                block.append(nn.BatchNorm2d(c_out))
            block.append(nn.ReLU(inplace=True))
            self.extra_down.append(nn.Sequential(*block))
            extra_out_channels.append(c_out)
            c_in = c_out

        # 3) ecs を最終確定：extra↓ を先頭に、pre↓ を末尾に
        extra_encoder_channels = tuple(reversed(extra_out_channels))
        backbone_encoder_channels = tuple(ecs_base)
        prestage_encoder_channels = tuple(self.pre_out_channels[::-1])
        self._encoder_channels = (
            extra_encoder_channels + backbone_encoder_channels + prestage_encoder_channels
        )
        (
            extra_skip_channels,
            backbone_skip_channels,
            prestage_skip_channels,
        ) = self._collect_skip_channel_groups(
            extra_out_channels=extra_out_channels,
            backbone_encoder_channels=backbone_encoder_channels,
            prestage_encoder_channels=prestage_encoder_channels,
        )
        self._decoder_skip_slots, self._skip_layout = self._build_skip_layout(
            extra_skip_channels=extra_skip_channels,
            backbone_skip_channels=backbone_skip_channels,
            prestage_skip_channels=prestage_skip_channels,
        )
        self._decoder_skip_channels = self._build_decoder_skip_channels(
            self._decoder_skip_slots
        )
        # Decoder
        self.decoder = UnetDecoder2d(
            encoder_channels=self._encoder_channels,
            skip_channels=self._decoder_skip_channels,
            decoder_channels=decoder_channels,
            scale_factors=decoder_scales,
            upsample_mode=upsample_mode,
            attention_type=attention_type,
            intermediate_conv=intermediate_conv,
        )
        self.seg_head = SegmentationHead2d(
            in_channels=self.decoder.decoder_channels[-1],
            out_channels=out_chans,
        )

        # 推論時の TTA(flip)を使うか
        self.use_tta = True

    @staticmethod
    def _normalize_disable_skip_indices(
        indices: tuple[int, ...] | list[int],
        *,
        name: str,
    ) -> tuple[int, ...]:
        if not isinstance(indices, (list, tuple)):
            msg = f'{name} must be list[int]'
            raise TypeError(msg)
        out: list[int] = []
        seen: set[int] = set()
        for idx, item in enumerate(indices):
            if isinstance(item, bool) or not isinstance(item, int):
                msg = f'{name}[{idx}] must be int'
                raise TypeError(msg)
            item_int = int(item)
            if item_int < 0:
                msg = f'{name}[{idx}] must be >= 0'
                raise ValueError(msg)
            if item_int in seen:
                msg = f'{name} contains duplicate index {item_int}'
                raise ValueError(msg)
            seen.add(item_int)
            out.append(item_int)
        return tuple(out)

    @staticmethod
    def _collect_skip_channel_groups(
        *,
        extra_out_channels: list[int],
        backbone_encoder_channels: tuple[int, ...],
        prestage_encoder_channels: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        extra_encoder_channels = tuple(reversed(extra_out_channels))
        extra_skip_channels = extra_encoder_channels[1:]
        if extra_encoder_channels:
            backbone_skip_channels = backbone_encoder_channels
        else:
            backbone_skip_channels = backbone_encoder_channels[1:]
        return extra_skip_channels, backbone_skip_channels, prestage_encoder_channels

    @staticmethod
    def _validate_disable_skip_indices(
        indices: tuple[int, ...],
        *,
        source: Literal['backbone', 'prestage'],
        available_count: int,
    ) -> None:
        key = f'disable_{source}_skip_indices'
        for source_index in indices:
            if source_index >= available_count:
                msg = (
                    f'{key} contains out-of-range {source} shallow index '
                    f'{source_index}; available {source} skip count is {available_count}'
                )
                raise ValueError(msg)

    def _build_skip_layout(
        self,
        *,
        extra_skip_channels: tuple[int, ...],
        backbone_skip_channels: tuple[int, ...],
        prestage_skip_channels: tuple[int, ...],
    ) -> tuple[tuple[_DecoderSkipSlot, ...], tuple[_SkipLayoutEntry, ...]]:
        self._validate_disable_skip_indices(
            self.disable_backbone_skip_indices,
            source='backbone',
            available_count=len(backbone_skip_channels),
        )
        self._validate_disable_skip_indices(
            self.disable_prestage_skip_indices,
            source='prestage',
            available_count=len(prestage_skip_channels),
        )

        decoder_skip_slots: list[_DecoderSkipSlot] = []
        skip_layout: list[_SkipLayoutEntry] = []
        decoder_skip_index = 0

        for channels in extra_skip_channels:
            decoder_skip_slots.append(
                _DecoderSkipSlot(
                    source='extra',
                    source_shallow_index=None,
                    decoder_skip_index=decoder_skip_index,
                    channels=int(channels),
                    enabled=True,
                )
            )
            decoder_skip_index += 1

        for source, source_channels, disabled_indices in (
            ('backbone', backbone_skip_channels, self.disable_backbone_skip_indices),
            ('prestage', prestage_skip_channels, self.disable_prestage_skip_indices),
        ):
            disabled_set = set(disabled_indices)
            count = len(source_channels)
            for local_deep_index, channels in enumerate(source_channels):
                source_shallow_index = count - 1 - local_deep_index
                enabled = source_shallow_index not in disabled_set
                slot = _DecoderSkipSlot(
                    source=source,
                    source_shallow_index=source_shallow_index,
                    decoder_skip_index=decoder_skip_index,
                    channels=int(channels),
                    enabled=enabled,
                )
                decoder_skip_slots.append(slot)
                skip_layout.append(
                    _SkipLayoutEntry(
                        source=source,
                        source_shallow_index=source_shallow_index,
                        decoder_skip_index=decoder_skip_index,
                        channels=int(channels),
                        enabled=enabled,
                    )
                )
                decoder_skip_index += 1

        return tuple(decoder_skip_slots), tuple(skip_layout)

    @staticmethod
    def _build_decoder_skip_channels(
        decoder_skip_slots: tuple[_DecoderSkipSlot, ...]
    ) -> tuple[int, ...]:
        return tuple(
            slot.channels if slot.enabled else 0 for slot in decoder_skip_slots
        )

    def _disable_skip_tensors(
        self, feats: list[torch.Tensor]
    ) -> list[torch.Tensor | None]:
        if len(feats) != len(self._encoder_channels):
            msg = (
                'encoder feature count mismatch: '
                f'expected {len(self._encoder_channels)}, got {len(feats)}'
            )
            raise ValueError(msg)
        out: list[torch.Tensor | None] = [feats[0]]
        for slot, feat in zip(self._decoder_skip_slots, feats[1:], strict=True):
            out.append(feat if slot.enabled else None)
        return out

    def _encode(self, x) -> list[torch.Tensor | None]:
        # ★各pre段の出力を控える
        pre_feats = []
        for b in self.pre_down:
            x = b(x)
            pre_feats.append(x)
            if getattr(self, 'print_shapes', False):
                print(f'[pre] {tuple(x.shape)}')

        # backbone → deepest-first
        feats = self.backbone(x)[::-1]

        # extra_down(最深側を前に積む)
        top = feats[0]
        for b in self.extra_down:
            top = b(top)
            feats = [top, *feats]

        # ★pre_down 出力を浅い側(末尾)に積む
        return self._disable_skip_tensors(feats + pre_feats[::-1])

    @torch.inference_mode()
    def _proc_flip(self, x_in):
        x_flip = torch.flip(x_in, dims=[-2])

        feats = self._encode(x_flip)

        dec = self.decoder(feats)
        y = self.seg_head(dec[-1])
        return torch.flip(y, dims=[-2])

    def forward(self, x):
        """入力: x=(B,C,H,W)
        出力: y=(B,out_chans,H,W)  ※入力サイズに合わせて補間して返す.
        """
        H, W = x.shape[-2:]
        feats = self._encode(x)

        if getattr(self, 'print_shapes', False):
            for i, f in enumerate(feats):
                if f is None:
                    print(f'Encoder feature {i} shape: None')
                else:
                    print(f'Encoder feature {i} shape:', f.shape)
        dec = self.decoder(feats)

        y = self.seg_head(dec[-1])  # 低解像度 → 後段で補間
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)

        if self.training or not self.use_tta:
            return y

        # eval 時のみ簡易 TTA(左右反転)
        p1 = self._proc_flip(x)
        p1 = F.interpolate(p1, size=(H, W), mode='bilinear', align_corners=False)
        return torch.quantile(torch.stack([y, p1]), q=0.5, dim=0)
