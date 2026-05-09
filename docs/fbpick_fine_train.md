# FBPick Fine Training

This document is the current contract for `fbpick-fine` training local windows.
Fine inference behavior is documented separately in `docs/fbpick_fine_infer.md`.

## Window Center

Fine training uses the physics / robust output as the local-window center source.
The code default remains `robust_pick_i` for compatibility with old robust
artifacts, but the recommended config uses `fine_center_i` first and falls back
to `robust_pick_i`:

```yaml
window_center:
  npz_key: fine_center_i
  fallback_npz_key: robust_pick_i
```

When the robust npz contains `fine_center_i`, the fine stage uses it as the
center of each local window. Older robust artifacts without `fine_center_i` can
still be used by setting `fallback_npz_key: robust_pick_i`.

Physics output writes `fine_center_i` even when physical trend centering is not
used. In that case `fine_center_i` is still the fine-stage center contract and
may equal the robust center depending on the physics configuration.

## Center Jitter

`center_augment` adds training-only perturbations to the selected fine window center.
It simulates realistic error in the upstream coarse / physics center while
keeping the current fine target and loss unchanged.

Default config preserves the previous behavior:

```yaml
center_augment:
  enabled: false
  train_only: true
  p_no_jitter: 1.0
  uniform_jitter_samples: []
  clip_to_record: true
  require_fb_inside: true
```

Recommended site54 training config:

```yaml
center_augment:
  enabled: true
  train_only: true
  p_no_jitter: 0.7
  uniform_jitter_samples:
    - {prob: 0.20, lo: -32, hi: 32}
    - {prob: 0.08, lo: -64, hi: 64}
    - {prob: 0.02, lo: -128, hi: 128}
  clip_to_record: true
  require_fb_inside: true
```

Sampling behavior:

- `p_no_jitter` selects `jitter = 0`.
- Each `uniform_jitter_samples` entry samples an integer offset from `[lo, hi]`,
  inclusive.
- The probabilities are normalized internally and may be specified as relative
  weights, but their sum must be greater than zero.
- With `clip_to_record: true`, the jittered center is clipped to
  `[0, n_samples_orig - 1]`.
- `require_fb_inside: false` is unsupported and rejected.

## Supervision

The target remains the local first-break index:

```text
fb_idx_view = fb_i - window_start_i
```

Only windows where the ground-truth first break remains inside the local window
are used for supervised training. If jitter moves the window so the ground truth
falls outside the local time range, the sample is rejected and redrawn.

Out-of-window jittered samples are not used as negative / no-pick / noise
examples. No-pick and noise supervision are out of scope for this fine-training
contract.

## Validation And Inference

Center jitter is applied only by the fine training dataset. Validation and
labeled-infer datasets use the configured `window_center` value directly, and
raw fine inference also uses the configured window center without jitter.

For deterministic validation and inference, keep:

```yaml
window_center:
  npz_key: fine_center_i
  fallback_npz_key: robust_pick_i
```

`center_augment` is not part of the fine inference config.

## Rejection Monitoring

Fine training retries sample construction up to `dataset.max_trials`. Rejection
accounting includes both non-jitter local-window failures and jitter-induced
local-window failures:

```text
rejections: empty=..., min_pick=..., fblc=..., local_window=..., center_jitter=...
```

Monitor `center_jitter` relative to `local_window` when tuning jitter ranges.
A high `center_jitter` count means the configured jitter often places the ground
truth outside the local window and reduces useful supervised samples.
