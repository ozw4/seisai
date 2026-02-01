## SegyGatherPipelineDataset: Input assumptions (v0)

### segy_files
- list[str], empty not allowed.
- Each file is readable by segyio.
- Within a file, dt and n_samples are constant.
- Offsets may be missing or broken; in that case offsets are treated as all-zero and processing continues.

### fb_files
- list[str], same length as segy_files (1:1 pairing).
- np.load() must succeed.
- fb is per-trace aligned: fb[trace_index] corresponds to the SEG-Y trace order (length matches tracecount).

### fb definition
- fb is integer sample index (fixed).
- Raw fb is 0-based, but for data-quality reasons fb<=0 is invalid.
- After view projection, fb_view<=0 is also invalid.
- Invalid fb is represented as -1.

### short gather padding
- If a sampled gather has fewer than subset_traces, the waveform is padded (already done by loader).
- fb/offsets are also padded to H=subset_traces:
  - fb pad value: -1
  - offsets pad value: 0
  - indices pad value: -1
- trace_valid: bool (H,), True for real traces and False for padded traces.
