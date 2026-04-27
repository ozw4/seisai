#!/usr/bin/env bash
set -euo pipefail

readonly WORK_EXCLUDE_PATHSPEC=':(exclude).work'
readonly VENDOR_EXCLUDE_PATHSPEC=':(exclude)vendor/issue_forge'

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required check command: $1"
}

collect_changed_files() {
  local base_ref="$1"

  {
    git diff --name-only "$base_ref" -- . "$WORK_EXCLUDE_PATHSPEC" "$VENDOR_EXCLUDE_PATHSPEC"
    git diff --name-only --cached -- . "$WORK_EXCLUDE_PATHSPEC" "$VENDOR_EXCLUDE_PATHSPEC"
    git diff --name-only -- . "$WORK_EXCLUDE_PATHSPEC" "$VENDOR_EXCLUDE_PATHSPEC"
    git ls-files --others --exclude-standard -- . "$WORK_EXCLUDE_PATHSPEC" "$VENDOR_EXCLUDE_PATHSPEC"
  } | awk 'NF && !seen[$0]++'
}

run_shellcheck_if_needed() {
  local -a shell_targets=("$@")

  if [[ "${#shell_targets[@]}" -eq 0 ]]; then
    printf 'shellcheck: skipped (no shell targets changed)\n'
    return 0
  fi

  require_command shellcheck
  printf 'shellcheck: %s target(s)\n' "${#shell_targets[@]}"
  shellcheck -x "${shell_targets[@]}"
}

run_pytest_targets_if_needed() {
  local -a pytest_targets=("$@")

  if [[ "${#pytest_targets[@]}" -eq 0 ]]; then
    printf 'pytest: skipped (no changed pytest targets)\n'
    return 0
  fi

  require_command pytest
  printf 'pytest: pytest -q'
  printf ' %s' "${pytest_targets[@]}"
  printf '\n'
  pytest -q "${pytest_targets[@]}"
}

main() {
  local base_ref
  local path
  local -a changed_files=()
  local -a pytest_targets=()
  local -a shell_targets=()

  if [[ "$#" -ne 1 ]]; then
    fail "Usage: $0 <base-ref>"
  fi

  base_ref="$1"

  git rev-parse --verify "$base_ref" >/dev/null 2>&1 || fail "Missing base ref for checks: $base_ref"

  mapfile -t changed_files < <(collect_changed_files "$base_ref")

  if [[ "${#changed_files[@]}" -eq 0 ]]; then
    printf 'No changes detected relative to %s\n' "$base_ref"
    return 0
  fi

  printf 'Changed files relative to %s:\n' "$base_ref"
  printf ' - %s\n' "${changed_files[@]}"

  for path in "${changed_files[@]}"; do
    case "$path" in
      *.sh)
        [[ -f "$path" ]] && shell_targets+=("$path")
        ;;
    esac

    if [[ -f "$path" && "$path" == *.py ]]; then
      case "$path" in
        test_*.py|*_test.py|tests/*|*/tests/*)
          pytest_targets+=("$path")
          ;;
      esac
    fi
  done

  run_shellcheck_if_needed "${shell_targets[@]}"
  run_pytest_targets_if_needed "${pytest_targets[@]}"
}

main "$@"
