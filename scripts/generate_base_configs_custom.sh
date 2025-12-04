#!/usr/bin/env bash

# This script generates base_config.yaml for every URDF folder under
# data/URDF_with_affordance/URDF. It copies the assets to
# data/custom_objects/<dataset-name> and writes a base_config.yaml.

conda activate articubot
source prepare.sh

ASSET_ROOT="$PROJECT_DIR/data/URDF_with_affordance/URDF"
OUTPUT_ROOT="$PROJECT_DIR/data/custom_objects"

for asset in "$ASSET_ROOT"/*; do
  if [ ! -d "$asset" ]; then
    continue
  fi
  # skip hidden/temporary directories
  base="$(basename "$asset")"
  if [[ "$base" == .* ]]; then
    continue
  fi

  # Check for annotation and urdf
  if [ ! -f "$asset/annotation.json" ]; then
    echo "Skipping $asset, annotation.json missing"
    continue
  fi
  shopt -s nullglob
  urdfs=("$asset"/*.urdf)
  shopt -u nullglob
  if [ ${#urdfs[@]} -eq 0 ]; then
    echo "Skipping $asset, no URDF found"
    continue
  fi

  echo "Generating base config for: $asset"
  python manipulation/generate_base_config_from_urdf_custom.py \
    --asset-folder "$asset" \
    --output-root "$OUTPUT_ROOT" \
    --overwrite-assets \
    --overwrite-support || echo "Failed for $asset" && continue
  echo "Done: $asset"
  echo
done

echo "All done"
