# YAML Config Loading Contract (v0)

This document specifies the **exact** behavior of YAML config loading in this repo
(SeisAI monorepo), including:

- YAML parsing requirements
- `base:` inheritance (recursive deep-merge)
- automatic path resolution for specific `*_files` keys
- listfile expansion (`1 line = 1 path`, optional per-line JSON metadata)
- additional path/URI resolution performed by pipeline utilities

This contract is implemented primarily by:

- `seisai_utils.config_yaml.load_yaml`
- `seisai_utils.listfiles.expand_cfg_listfiles`
- `seisai_engine.pipelines.common.config_io.resolve_cfg_paths` / `resolve_relpath`
- `seisai_engine.pipelines.common.skeleton_helpers.resolve_out_dir`
- `seisai_engine.tracking.config.resolve_tracking_uri`

---

## Terminology

### base_dir
`base_dir` is the directory that contains the **entry config YAML**.
In most entrypoints, this is computed as:

- `base_dir = Path(config_path).resolve().parent`

### listfile
A **listfile** is a plain text file containing one path per line (with optional
metadata). It is used to specify large file lists without writing them directly
in YAML.

---

## 1) YAML parsing (root requirements)

### Loader
Configs are loaded with `yaml.safe_load`.

### Root type
The YAML root **must** be a mapping (Python `dict`).

- If the YAML root is not a dict (including the empty file which loads as `None`),
  loading **fails immediately** with `TypeError('config root must be a dict')`.

---

## 2) `base:` inheritance

### 2.1 Syntax
A config may define a top-level `base:` key.

- `base: <str>`
- `base: [<str>, <str>, ...]`

Constraints:

- `base` must be a non-empty `str` or a non-empty `list[str]`.
- Empty strings and empty lists are rejected.

### 2.2 Path resolution for base files
Each base entry is treated as a file path.

- `~` is expanded.
- If the base path is **relative**, it is resolved **relative to the YAML file
  that declares it** (i.e., `current_yaml.parent / base`).
- The base path must exist and must be a file.

If the base file is missing, loading fails with:

- `ValueError('base config file not found: <abs_path>')`

### 2.3 Recursion and cycle detection
Base configs may themselves contain `base:`.

- Loading is recursive.
- Cycles are rejected.

If a circular reference is detected, loading fails with a message like:

- `ValueError('circular base reference: a.yaml -> b.yaml -> a.yaml')`

### 2.4 Merge semantics (deep merge)
Inheritance is implemented as a **deep merge** on dicts.

- If `base[key]` and `override[key]` are both dicts:
  - merge recursively
- Otherwise:
  - the override value **replaces** the base value

Important implications:

- Lists are **not concatenated**. A list in the override replaces the base list.
- Scalars always replace.

### 2.5 Multiple base files (precedence)
If `base` is a list, each base is loaded in order and merged into an accumulated
base config.

- Later base entries override earlier ones for conflicting keys.
- Then the current YAML overrides the merged base.

In other words:

1. `merged_base = merge({}, base1)`
2. `merged_base = merge(merged_base, base2)`
3. `final = merge(merged_base, current_cfg_without_base)`

### 2.6 `base` key is removed
The `base` key does **not** appear in the final returned config.

---

## 3) Automatic path resolution in `load_yaml`

### 3.1 Target keys
During YAML loading, values of specific keys are resolved into absolute paths.
The key match is by **key name only** (not by location), and it is applied
recursively across all dict/list nodes.

Target key set:

- `segy_files`
- `phase_pick_files`
- `infer_segy_files`
- `infer_phase_pick_files`
- `input_segy_files`
- `target_segy_files`
- `infer_input_segy_files`
- `infer_target_segy_files`

### 3.2 Allowed value types
For each target key, the value must be:

- `str`  (single path, often a listfile path)
- `list[str]` (explicit file list)

Otherwise loading fails with `TypeError`.

Notes:

- For `list[str]`, every item must be a `str`.

### 3.3 Resolution rules
Each target value is resolved with:

- `~` expansion (`Path(value).expanduser()`)
- If not absolute, prepend `base_dir_of_that_yaml` (`yaml_path.parent / value`)
- `resolve()` to an absolute normalized path

Absolute paths remain absolute.

### 3.4 Environment variables are NOT expanded here
`load_yaml` does **not** perform `os.path.expandvars`.

- Do not write `$VAR/...` in YAML path values expecting expansion.
- If you need env var expansion, use **listfile** lines (see §4.2), where
  expansion is supported.

---

## 4) Listfile expansion (`expand_cfg_listfiles`)

`expand_cfg_listfiles(cfg, keys=...)` transforms configured listfile references
into `list[str]` values **in-place**.

### 4.1 Key selection (`keys`)
- `keys` is a list of key paths.
- Each key path may be:
  - dot-path string, e.g. `"paths.segy_files"`
  - or `Sequence[str]`, e.g. `["paths", "segy_files"]`

If the key path cannot be traversed (missing key, wrong parent type), it fails
immediately (no fallback).

### 4.2 Value types
For each selected key, the current config value must be:

- `list[str]`  → kept as-is (no file existence validation here)
- `str` or `Path` → treated as a listfile path, loaded and expanded

Any other type raises `TypeError`.

### 4.3 Listfile path normalization
When loading a listfile path:

- `os.path.expandvars` is applied to the listfile path string
- `~` is expanded
- the path is `resolve()`-ed

**Recommendation:** In typical pipelines, the listfile path itself should already
be absolute (or at least resolved against the YAML directory) before expansion.
This is naturally satisfied when the key is one of the auto-resolved `*_files`
keys in §3.

### 4.4 Listfile line format
Each non-empty, non-comment line produces one path entry.

Ignored lines:

- empty / whitespace-only lines
- lines starting with `#` after stripping

Path token rules:

- Environment variables are expanded (`$VAR`)
- `~` is expanded
- If the resulting path is relative, it is resolved relative to the **listfile
  directory** (not the YAML directory)
- Finally, the path is `resolve()`-ed into an absolute path

### 4.5 File existence validation
After loading all lines:

- The list must be non-empty, otherwise `ValueError('listfile is empty: ...')`.
- Every resolved path must exist and be a **file**.
  - missing → `FileNotFoundError(<path>)`
  - exists but not a file → `ValueError('expected file: <path>')`

### 4.6 Optional per-line JSON metadata
A listfile line may attach a JSON object (metadata) to a path.

Supported syntaxes:

1) TAB-separated:

```
<path>\t<json object>
```

2) Whitespace + JSON object:

```
<path><space>{...}
```

Parsing details:

- If a `{` appears in the path **without whitespace before it** (e.g.
  `a{meta}.sgy`), it is treated as part of the path (no metadata).
- Metadata must be valid JSON and must decode to a JSON object (dict).
- Invalid JSON or non-object JSON causes a `ValueError`.

### 4.7 Metadata storage in config (`_listfile_meta`)
When a selected key is expanded from a listfile, its per-line metadata list is
stored into:

- `cfg['_listfile_meta'][<key_path_str>] = metas`

where:

- `<key_path_str>` is the normalized dot-path (e.g. `paths.segy_files`)
- `metas` is `list[dict | None]` aligned to the expanded path list

If a key is provided as `list[str]` (not a listfile), metadata for that key path
is removed from `_listfile_meta` if present.

### 4.8 Reading metadata
Use:

- `seisai_utils.listfiles.get_cfg_listfile_meta(cfg, key_path=...)`

It returns:

- `None` if no metadata exists for the key
- `list[dict | None]` otherwise (a shallow copy)

---

## 5) Additional path/URI resolution (pipeline-level)

Beyond the automatic `*_files` resolution in §3, pipelines resolve other
path-like values explicitly.

### 5.1 `paths.out_dir`
`paths.out_dir` is resolved relative to the entry YAML directory (`base_dir`) by
`resolve_out_dir(cfg, base_dir)`.

- value must be `str`
- `~` is expanded
- if relative, it is resolved against `base_dir`

### 5.2 `train.init_ckpt` (optional)
If present and non-empty, `train.init_ckpt` is resolved relative to `base_dir`.

- type must be `str` or `null`
- if resolved path does not exist as a file → `FileNotFoundError`

(Resolution and validation are performed by
`seisai_engine.pipelines.common.init_weights._resolve_init_ckpt_path`.)

### 5.3 `tracking.tracking_uri` for `file:` URIs
Tracking config supports relative `file:` URIs, resolved against `base_dir`.

Examples:

- `file:./mlruns` → `file:/abs/path/to/mlruns`

(Implemented by `seisai_engine.tracking.config.resolve_tracking_uri`.)

### 5.4 Explicit resolution for arbitrary keys
For additional keys not covered by the heuristics above, use:

- `resolve_cfg_paths(cfg, base_dir, keys=[...])`

It resolves only the provided keys, and supports values of:

- `str`
- `list[str]`

Note: this function expands `~` but does not expand environment variables.

---

## 6) Recommended loading sequence (entrypoint)

Typical usage in `*/train.py` and `*/infer_*.py`:

1) Load config + compute base_dir:

- `cfg, base_dir = load_cfg_with_base_dir(Path(args.config))`

2) Expand listfiles for the relevant keys:

- `expand_cfg_listfiles(cfg, keys=[...])`

3) Resolve additional paths/URIs as needed:

- `out_dir = resolve_out_dir(cfg, base_dir)`
- `tracking_cfg = load_tracking_config(cfg, base_dir)` (handles file: URI)
- `maybe_load_init_weights(cfg=cfg, base_dir=base_dir, ...)`

---

## 7) Examples

### 7.1 Base inheritance

`train.yaml`:

```yaml
base:
  - ./base.yaml
  - ./override.yaml

paths:
  segy_files: ./lists/train_segy.txt
  out_dir: ./out

train:
  lr: 1.0e-4
```

- `base.yaml` is resolved relative to `train.yaml`.
- `override.yaml` overrides `base.yaml` on conflicts.
- `train.yaml` overrides both.

### 7.2 Listfile with metadata

`lists/train_segy.txt`:

```text
/data/a.sgy\t{"primary_keys":["ffid"],"primary_ranges":{"ffid":[[1,100]]}}
/data/b.sgy
```

After `expand_cfg_listfiles(cfg, keys=['paths.segy_files'])`:

- `cfg['paths']['segy_files']` becomes `['/data/a.sgy', '/data/b.sgy']`
- `cfg['_listfile_meta']['paths.segy_files']` stores metadata aligned to the list

---

## 8) Non-goals / limitations

- Environment variable expansion is **not** performed for YAML path values.
  Use listfiles if you need `$VAR` expansion.
- Existence checks for direct `list[str]` values are **not** performed by the
  loader; pipelines/datasets may fail later when opening files.
