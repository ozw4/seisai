from typing import Literal

import numpy as np

from .config import FirstBreakGateConfig


class FirstBreakGate:
    """First-break based gating utilities for quality control.

    This class provides simple acceptance criteria (gates) based on first-break
    pick indices within a gather/window. Two gates are supported:

    - Min-pick ratio gate: requires a minimum fraction of valid first-break picks.
    - First-break local consistency (FBLC) gate: checks local consistency of
    adjacent first-break picks using a percentile of absolute index differences.

    Notes:
            - A pick index is treated as **valid** when it is strictly greater than 0.
            - All thresholds are validated at initialization and again when overridden
            per-call (e.g., in `fblc_accept`).

    Attributes:
            cfg: Configuration object controlling gate behavior and thresholds.

    """

    def __init__(self, cfg: FirstBreakGateConfig) -> None:
        """Initialize the gate with configuration.

        Args:
                cfg: Gate configuration parameters.

        Raises:
                ValueError: If any configuration value is out of the allowed range.

        """
        self._validate_config(
            percentile=cfg.percentile,
            thresh_ms=cfg.thresh_ms,
            min_pairs=cfg.min_pairs,
            apply_on=cfg.apply_on,
        )
        self.cfg = cfg

    @staticmethod
    def _validate_config(
        *,
        percentile: float,
        thresh_ms: float,
        min_pairs: int,
        apply_on: Literal['any', 'super_only', 'off'] | None = None,
    ) -> None:
        """Validate gate parameters.

        Args:
                percentile: Percentile in (0, 100) used for the FBLC statistic.
                thresh_ms: Threshold (milliseconds) for FBLC acceptance.
                min_pairs: Minimum number of valid adjacent pick pairs required to
                        compute FBLC.
                apply_on: Whether to apply the FBLC gate. If provided, must be one of
                        `'any'`, `'super_only'`, or `'off'`.

        Raises:
                ValueError: If parameters are out of range or `apply_on` is invalid.

        """
        if not (0.0 < float(percentile) < 100.0):
            msg = 'percentile must be in (0, 100)'
            raise ValueError(msg)
        if not (float(thresh_ms) > 0.0):
            msg = 'thresh_ms must be positive'
            raise ValueError(msg)
        if int(min_pairs) < 0:
            msg = 'min_pairs must be non-negative'
            raise ValueError(msg)
        if apply_on is not None and apply_on not in ('any', 'super_only', 'off'):
            msg = "apply_on must be 'any', 'super_only', or 'off'"
            raise ValueError(msg)

    def should_apply(
        self,
        *,
        did_super: bool,
        apply_on: Literal['any', 'super_only', 'off'] | None = None,
    ) -> bool:
        """Determine whether the gate should be applied for this sample.

        The effective `apply_on` mode is taken from `self.cfg.apply_on` unless an
        override is provided.

        Args:
                did_super: Whether a "super" processing stage was applied upstream.
                apply_on: Optional override for application mode:
                        - `'off'`: never apply the gate.
                        - `'any'`: always apply the gate.
                        - `'super_only'`: apply only if `did_super` is True.

        Returns:
                True if the gate should be applied, otherwise False.

        Raises:
                ValueError: If the effective `apply_on` is not a supported value.

        """
        ap = self.cfg.apply_on if apply_on is None else apply_on
        if ap == 'off':
            return False
        if ap == 'any':
            return True
        if ap == 'super_only':
            return bool(did_super)
        msg = "apply_on must be 'any', 'super_only', or 'off'"
        raise ValueError(msg)

    def min_pick_accept(self, fb_idx_win: np.ndarray) -> tuple[bool, int, float]:
        """Apply the minimum pick ratio gate.

        A pick index is treated as valid when it is strictly greater than 0.
        The ratio is computed as:

                `ratio = n_valid / H`

        where `H` is the number of samples (elements) in the window.

        Args:
                fb_idx_win: 1D array of first-break pick indices for a window/gather.

        Returns:
                A tuple of:
                        - accept: True if `ratio >= cfg.min_pick_ratio` (or gate disabled),
                        otherwise False.
                        - n_valid: Number of valid picks (indices > 0).
                        - ratio: Fraction of valid picks in the window.

        Notes:
                - If `cfg.min_pick_ratio` is None or 0.0, the gate is disabled and
                returns `(True, 0, 0.0)`.
                - If the input window is empty, returns `(False, 0, 0.0)`.

        """
        r = self.cfg.min_pick_ratio
        if r is None or float(r) == 0.0:
            return True, 0, 0.0
        v = fb_idx_win.astype(np.int64, copy=False)
        H = int(v.size)
        if H == 0:
            return False, 0, 0.0
        valid = v > 0
        n_valid = int(valid.sum())
        ratio = n_valid / H
        return (ratio >= float(r), n_valid, ratio)

    def fblc_accept(
        self,
        fb_idx_win: np.ndarray,
        dt_eff_sec: float,
        *,
        did_super: bool = False,
        percentile: float | None = None,
        thresh_ms: float | None = None,
        min_pairs: int | None = None,
        apply_on: Literal['any', 'super_only', 'off'] | None = None,
    ) -> tuple[bool, float | None, int]:
        """Apply the First-Break Local Consistency (FBLC) gate.

        This gate measures local consistency by looking at adjacent pick index
        differences. Only adjacent pairs where both indices are valid (> 0) are
        considered. Let:

                `diffs = |v[i+1] - v[i]|` for valid adjacent pairs

        The FBLC statistic is defined as the `percentile`-th percentile of `diffs`
        (measured in **index** units), then converted to milliseconds via
        `dt_eff_sec`.

        Acceptance criterion:

                `p_ms <= thresh_ms`

        where `p_ms = percentile(diffs) * dt_eff_sec * 1000`.

        Args:
                fb_idx_win: 1D array of first-break pick indices for a window/gather.
                dt_eff_sec: Effective sample interval (seconds) used to convert index
                        differences into time.
                did_super: Whether a "super" processing stage was applied upstream.
                        Used only when `apply_on` is `'super_only'`.
                percentile: Optional override for the percentile in (0, 100).
                thresh_ms: Optional override for the threshold in milliseconds (> 0).
                min_pairs: Optional override for minimum required valid adjacent pairs
                        (non-negative).
                apply_on: Optional override for application mode (`'any'`,
                        `'super_only'`, `'off'`). If `'off'` or gate is not applicable, the
                        function returns an accepted result without computing FBLC.

        Returns:
                A tuple of:
                        - accept: True if accepted, otherwise False.
                        - p_ms: Computed percentile in milliseconds, or None if gate did not
                        run (e.g., not applicable) or insufficient valid pairs.
                        - valid_pairs: Number of valid adjacent pairs used (or available).

        Raises:
                ValueError: If overridden parameters are invalid.

        """
        # 無効化 or 適用不要なら素通し
        if not self.should_apply(did_super=did_super, apply_on=apply_on):
            return True, None, 0

        p = self.cfg.percentile if percentile is None else float(percentile)
        th = self.cfg.thresh_ms if thresh_ms is None else float(thresh_ms)
        mp = self.cfg.min_pairs if min_pairs is None else int(min_pairs)
        self._validate_config(
            percentile=p,
            thresh_ms=th,
            min_pairs=mp,
            apply_on=apply_on,
        )

        v = fb_idx_win.astype(np.float64, copy=False)
        valid = v > 0
        m = valid[1:] & valid[:-1]
        valid_pairs = int(m.sum())
        if valid_pairs < mp:
            return False, None, valid_pairs

        diffs = np.abs(v[1:] - v[:-1])[m]
        q = float(np.percentile(diffs, p))
        p_ms = q * float(dt_eff_sec) * 1000.0
        return (p_ms <= th), p_ms, valid_pairs
