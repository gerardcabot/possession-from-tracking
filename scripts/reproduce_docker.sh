#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -gt 2 ]; then
  echo "usage: scripts/reproduce_docker.sh [data_dir] [out_dir]" >&2
  echo "env: MTAH_DATA_DIR (default ../data), OUT_DIR (default outputs), OUT_NAME (default learned_final_docker.csv)," >&2
  echo "     DOCKER_PLATFORM (default: the host's), IMAGE_TAG (default possession-repro)" >&2
  exit 2
fi

data_dir="${1:-${MTAH_DATA_DIR:-../data}}"
out_dir="${2:-${OUT_DIR:-outputs}}"
out_name="${OUT_NAME:-learned_final_docker.csv}"
image="${IMAGE_TAG:-possession-repro}"
platform_args=()
if [ -n "${DOCKER_PLATFORM:-}" ]; then
  platform_args=(--platform "$DOCKER_PLATFORM")
fi

resolve() {
  perl -MCwd=realpath -e 'print realpath(shift)' "$1"
}

meta="$(resolve "$data_dir/tactical/BAR-ATH_tactical_atd.xml")"
raw="$(resolve "$data_dir/tactical/BAR-ATH_tactical_atd.txt")"
mkdir -p "$out_dir"
out_dir="$(resolve "$out_dir")"

if [ "${#platform_args[@]}" -gt 0 ]; then
  docker build "${platform_args[@]}" -t "$image" .
else
  docker build -t "$image" .
fi

start=$(date +%s)
if [ "${#platform_args[@]}" -gt 0 ]; then
  run=(docker run --rm "${platform_args[@]}")
else
  run=(docker run --rm)
fi
"${run[@]}" \
  -v "$meta":/data/tactical/BAR-ATH_tactical_atd.xml:ro \
  -v "$raw":/data/tactical/BAR-ATH_tactical_atd.txt:ro \
  -v "$out_dir":/out \
  "$image" \
  python -m possession.run --config configs/learned.json --out "/out/$out_name"
echo "run seconds $(( $(date +%s) - start ))"

if command -v sha256sum > /dev/null; then
  sha256sum "$out_dir/$out_name"
else
  shasum -a 256 "$out_dir/$out_name"
fi
