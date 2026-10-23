#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <archive_dir> <output_dir>" >&2
    exit 2
fi

archive_dir=$1
output_dir=$2
mkdir -p "$output_dir"

mapfile -t archives < <(
    find "$archive_dir" -maxdepth 1 -type f \
        -name 'scannet_scans_part_*.tar.gz' | sort
)

if [[ ${#archives[@]} -ne 16 ]]; then
    echo "Expected 16 scan archives, found ${#archives[@]}" >&2
    exit 1
fi

for archive in "${archives[@]}"; do
    name=$(basename "$archive" .tar.gz)
    marker="$output_dir/.${name}.complete"
    if [[ -f "$marker" ]]; then
        echo "SKIP $name (already complete)"
        continue
    fi

    echo "START $name $(date --iso-8601=seconds)"
    tar -xzf "$archive" -C "$output_dir" \
        --wildcards \
        'scans/scene*/*_vh_clean_2.ply' \
        'scans/scene*/*_vh_clean_2.0.010000.segs.json' \
        'scans/scene*/*.aggregation.json'
    touch "$marker"
    echo "DONE  $name $(date --iso-8601=seconds)"
done

echo "EXTRACTION_COMPLETE $(date --iso-8601=seconds)"
