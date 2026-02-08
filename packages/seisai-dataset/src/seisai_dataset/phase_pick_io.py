from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class PhasePickCSR:
    """Phase picks stored as two per-trace CSR lists (P and S).

    - indptr: (n_traces+1,)
    - data: (nnz,)
    - pick values: <=0 are treated as invalid; >0 are valid
    """

    p_indptr: np.ndarray
    p_data: np.ndarray
    s_indptr: np.ndarray
    s_data: np.ndarray

    @property
    def n_traces(self) -> int:
        return int(self.p_indptr.size) - 1


@dataclass(frozen=True, slots=True)
class PhasePickWindowCSR:
    """Phase picks after subsetting/padding, with extracted first picks."""

    p_indptr: np.ndarray
    p_data: np.ndarray
    s_indptr: np.ndarray
    s_data: np.ndarray
    p_first: np.ndarray
    s_first: np.ndarray

    @property
    def H(self) -> int:  # noqa: N802 (keep dataset naming convention)
        return int(self.p_first.size)


def validate_csr(
    *,
    indptr: np.ndarray,
    data: np.ndarray,
    n_traces: int | None = None,
    name: str = 'csr',
) -> int:
    """Validate a 1D CSR representation.

    Parameters
    ----------
    indptr, data
            CSR arrays where `indptr` has length `n_traces+1` and
            `indptr[-1] == len(data)`.
    n_traces
            Optional expected number of traces. If None, inferred from indptr length.
    name
            Name used in error messages.

    Returns
    -------
    int
            The number of traces (rows) in the CSR.

    Raises
    ------
    ValueError
            If any contract violation is detected.

    """
    ip = np.asarray(indptr)
    if ip.ndim != 1:
        msg = f'{name}: indptr must be 1D, got shape={ip.shape}'
        raise ValueError(msg)
    if ip.size == 0:
        msg = f'{name}: indptr must be non-empty'
        raise ValueError(msg)
    if not np.issubdtype(ip.dtype, np.integer):
        msg = f'{name}: indptr must be integer dtype, got {ip.dtype}'
        raise ValueError(msg)

    if n_traces is None:
        n_traces = int(ip.size) - 1
    else:
        n_traces = int(n_traces)
        if n_traces < 0:
            msg = f'{name}: n_traces must be >= 0, got {n_traces}'
            raise ValueError(msg)

    if int(ip.size) != n_traces + 1:
        msg = f'{name}: indptr length must be n_traces+1={n_traces + 1}, got {ip.size}'
        raise ValueError(
            msg
        )

    if int(ip[0]) != 0:
        msg = f'{name}: indptr[0] must be 0, got {int(ip[0])}'
        raise ValueError(msg)
    if np.any(ip[1:] < ip[:-1]):
        msg = f'{name}: indptr must be monotonic non-decreasing'
        raise ValueError(msg)

    d = np.asarray(data)
    if d.ndim != 1:
        msg = f'{name}: data must be 1D, got shape={d.shape}'
        raise ValueError(msg)
    if not np.issubdtype(d.dtype, np.integer):
        msg = f'{name}: data must be integer dtype, got {d.dtype}'
        raise ValueError(msg)
    if int(ip[-1]) != int(d.size):
        msg = f'{name}: indptr[-1] must equal len(data)={d.size}, got {int(ip[-1])}'
        raise ValueError(
            msg
        )

    return n_traces


def validate_phase_pick_csr(
    *,
    p_indptr: np.ndarray,
    p_data: np.ndarray,
    s_indptr: np.ndarray,
    s_data: np.ndarray,
) -> int:
    """Validate a pair of CSR lists (P and S) that must share the same n_traces."""
    n_traces = validate_csr(indptr=p_indptr, data=p_data, name='p')
    _ = validate_csr(indptr=s_indptr, data=s_data, n_traces=n_traces, name='s')
    return n_traces


def load_phase_pick_csr_npz(path: str | Path) -> PhasePickCSR:
    """Load and validate CSR phase picks from a .npz file."""
    with np.load(path, allow_pickle=False) as z:
        required = ('p_indptr', 'p_data', 's_indptr', 's_data')
        missing = [k for k in required if k not in z.files]
        if missing:
            msg = f'missing key(s) in npz: {missing}'
            raise ValueError(msg)

        p_indptr = np.asarray(z['p_indptr'])
        p_data = np.asarray(z['p_data'])
        s_indptr = np.asarray(z['s_indptr'])
        s_data = np.asarray(z['s_data'])

    validate_phase_pick_csr(
        p_indptr=p_indptr,
        p_data=p_data,
        s_indptr=s_indptr,
        s_data=s_data,
    )

    # Normalize dtypes (avoid copies when possible).
    return PhasePickCSR(
        p_indptr=p_indptr.astype(np.int64, copy=False),
        p_data=p_data.astype(np.int64, copy=False),
        s_indptr=s_indptr.astype(np.int64, copy=False),
        s_data=s_data.astype(np.int64, copy=False),
    )


def subset_csr(
    *,
    indptr: np.ndarray,
    data: np.ndarray,
    indices: np.ndarray,
    name: str = 'csr',
) -> tuple[np.ndarray, np.ndarray]:
    """Subset a CSR list by trace indices (rows).

    - Output row order follows `indices`.
    - Duplicate indices are allowed.
    """
    ip_in = np.asarray(indptr)
    d_in = np.asarray(data)
    n_traces = validate_csr(indptr=ip_in, data=d_in, name=name)
    ip = ip_in.astype(np.int64, copy=False)
    d = d_in.astype(np.int64, copy=False)

    ii_in = np.asarray(indices)
    if not np.issubdtype(ii_in.dtype, np.integer):
        msg = f'{name}: indices must be integer dtype, got {ii_in.dtype}'
        raise ValueError(msg)
    ii = ii_in.astype(np.int64, copy=False)
    if ii.ndim != 1:
        msg = f'{name}: indices must be 1D, got shape={ii.shape}'
        raise ValueError(msg)
    if ii.size == 0:
        return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64)

    if int(ii.min()) < 0 or int(ii.max()) >= n_traces:
        msg = f'{name}: indices out of range [0,{n_traces}), got min={int(ii.min())}, max={int(ii.max())}'
        raise ValueError(
            msg
        )

    starts = ip[ii]
    ends = ip[ii + 1]
    lengths = (ends - starts).astype(np.int64, copy=False)

    out_indptr = np.zeros(int(ii.size) + 1, dtype=np.int64)
    np.cumsum(lengths, out=out_indptr[1:])

    out_data = np.empty(int(out_indptr[-1]), dtype=np.int64)
    pos = 0
    for st, en, ln in zip(starts, ends, lengths, strict=True):
        ln_i = int(ln)
        if ln_i > 0:
            out_data[pos : pos + ln_i] = d[int(st) : int(en)]
            pos += ln_i
    return out_indptr, out_data


def pad_csr(
    *,
    indptr: np.ndarray,
    data: np.ndarray,
    n_traces: int,
    name: str = 'csr',
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a CSR list with empty traces up to `n_traces`."""
    ip_in = np.asarray(indptr)
    d_in = np.asarray(data)
    n0 = validate_csr(indptr=ip_in, data=d_in, name=name)
    ip = ip_in.astype(np.int64, copy=False)
    d = d_in.astype(np.int64, copy=False)
    n_traces = int(n_traces)
    if n_traces < n0:
        msg = f'{name}: cannot pad to smaller n_traces={n_traces} < {n0}'
        raise ValueError(msg)
    pad = n_traces - n0
    if pad == 0:
        return ip, d

    last = int(ip[-1])
    out_indptr = np.concatenate(
        [ip, np.full(pad, last, dtype=np.int64)],
        axis=0,
    )
    return out_indptr, d


def csr_first_positive(*, indptr: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Compute per-trace first pick as min(picks>0), or 0 if none."""
    ip_in = np.asarray(indptr)
    d_in = np.asarray(data)
    n_traces = validate_csr(indptr=ip_in, data=d_in, name='csr')
    ip = ip_in.astype(np.int64, copy=False)
    d = d_in.astype(np.int64, copy=False)
    out = np.zeros(n_traces, dtype=np.int64)
    for t in range(n_traces):
        st = int(ip[t])
        en = int(ip[t + 1])
        if st == en:
            continue
        v = d[st:en]
        vp = v[v > 0]
        if vp.size == 0:
            continue
        out[t] = int(vp.min())
    return out


def invalidate_s_by_first(
    *,
    p_first: np.ndarray,
    s_indptr: np.ndarray,
    s_data: np.ndarray,
    s_first: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invalidate S picks per trace if s_first < p_first by emptying that S slice."""
    pf_in = np.asarray(p_first)
    if pf_in.ndim != 1:
        msg = f'p_first must be 1D, got shape={pf_in.shape}'
        raise ValueError(msg)
    if not np.issubdtype(pf_in.dtype, np.integer):
        msg = f'p_first must be integer dtype, got {pf_in.dtype}'
        raise ValueError(msg)
    pf = pf_in.astype(np.int64, copy=False)

    sip_in = np.asarray(s_indptr)
    sd_in = np.asarray(s_data)
    H = int(pf.size)
    _ = validate_csr(indptr=sip_in, data=sd_in, n_traces=H, name='s')
    sip = sip_in.astype(np.int64, copy=False)
    sd = sd_in.astype(np.int64, copy=False)

    if s_first is None:
        sf = csr_first_positive(indptr=sip, data=sd)
    else:
        sf_in = np.asarray(s_first)
        if sf_in.ndim != 1 or int(sf_in.size) != H:
            msg = f's_first must have shape ({H},), got {sf_in.shape}'
            raise ValueError(msg)
        if not np.issubdtype(sf_in.dtype, np.integer):
            msg = f's_first must be integer dtype, got {sf_in.dtype}'
            raise ValueError(msg)
        sf = sf_in.astype(np.int64, copy=False)

    # NOTE: s_first==0 means "no valid S picks"; if p_first>0, we still empty the S slice
    # for consistency (it may contain only <=0 invalid values).
    mask = (pf > 0) & (sf < pf)
    if not np.any(mask):
        return sip, sd, sf

    # Rebuild CSR with selected traces zeroed.
    lengths = np.diff(sip).astype(np.int64, copy=False)
    out_lengths = lengths.copy()
    out_lengths[mask] = 0

    out_indptr = np.zeros(H + 1, dtype=np.int64)
    np.cumsum(out_lengths, out=out_indptr[1:])
    out_data = np.empty(int(out_indptr[-1]), dtype=np.int64)

    pos = 0
    for t, ln in enumerate(out_lengths):
        ln_i = int(ln)
        if ln_i == 0:
            continue
        st = int(sip[t])
        en = int(sip[t + 1])
        out_data[pos : pos + ln_i] = sd[st:en]
        pos += ln_i

    sf = sf.copy()
    sf[mask] = 0
    return out_indptr, out_data, sf


def subset_pad_first_invalidate(
    *,
    p_indptr: np.ndarray,
    p_data: np.ndarray,
    s_indptr: np.ndarray,
    s_data: np.ndarray,
    indices: np.ndarray,
    subset_traces: int,
) -> PhasePickWindowCSR:
    """Subset/pad P & S CSR, extract first picks, and apply the S<P invalidation rule."""
    validate_phase_pick_csr(
        p_indptr=p_indptr,
        p_data=p_data,
        s_indptr=s_indptr,
        s_data=s_data,
    )

    ii_in = np.asarray(indices)
    if ii_in.ndim != 1:
        msg = f'indices must be 1D, got shape={ii_in.shape}'
        raise ValueError(msg)
    if not np.issubdtype(ii_in.dtype, np.integer):
        msg = f'indices must be integer dtype, got {ii_in.dtype}'
        raise ValueError(msg)
    ii = ii_in.astype(np.int64, copy=False)
    H0 = int(ii.size)
    H = int(subset_traces)
    if H < H0:
        msg = f'subset_traces={H} must be >= len(indices)={H0}'
        raise ValueError(msg)

    p_ip, p_d = subset_csr(indptr=p_indptr, data=p_data, indices=ii, name='p')
    s_ip, s_d = subset_csr(indptr=s_indptr, data=s_data, indices=ii, name='s')

    p_ip, p_d = pad_csr(indptr=p_ip, data=p_d, n_traces=H, name='p')
    s_ip, s_d = pad_csr(indptr=s_ip, data=s_d, n_traces=H, name='s')

    p_first = csr_first_positive(indptr=p_ip, data=p_d)
    s_first = csr_first_positive(indptr=s_ip, data=s_d)
    s_ip, s_d, s_first = invalidate_s_by_first(
        p_first=p_first,
        s_indptr=s_ip,
        s_data=s_d,
        s_first=s_first,
    )

    return PhasePickWindowCSR(
        p_indptr=p_ip,
        p_data=p_d,
        s_indptr=s_ip,
        s_data=s_d,
        p_first=p_first,
        s_first=s_first,
    )
