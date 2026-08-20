#!/usr/bin/env bash

# Local deployment paths for this workstation.
if [[ ! -d '/media/sajio/New Volume/CodexDeployments/Greenwich' ]]; then
  udisksctl mount -b /dev/disk/by-uuid/5EEA38C5EA389B6B >/dev/null 2>&1 || true
fi
export GREENWICH_DEPLOY_ROOT='/media/sajio/New Volume/CodexDeployments/Greenwich'
export ALPHAMOTION_CACHE="$GREENWICH_DEPLOY_ROOT/cache/alphamotion"
export ALPHAMOTION_DATA="$GREENWICH_DEPLOY_ROOT/data/alphamotion"
export ALPHAMOTION_IMPORTED_LIBRARY="$ALPHAMOTION_DATA/imported_smpl"
export ALPHAMOTION_INCLUDE_CURATED=0
export ALPHAMOTION_DATA_STUDIO_REPO='/media/sajio/New Volume/BodyDataStudio'
export ALPHAMOTION_DATA_STUDIO_ROOT='/media/sajio/New Volume/body_data'
export ALPHAMOTION_DATA_STUDIO_CACHE="$GREENWICH_DEPLOY_ROOT/data/data_studio"
export ALPHAMOTION_DATA_STUDIO_PORT=8765
export ALPHAMOTION_GENMO_PYTHON="$GREENWICH_DEPLOY_ROOT/envs/genmo/bin/python"
export ALPHAMOTION_GENMO_REPO="$GREENWICH_DEPLOY_ROOT/sources/GENMO"
export HF_HOME="$GREENWICH_DEPLOY_ROOT/cache/huggingface"
export UV_CACHE_DIR="$GREENWICH_DEPLOY_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$GREENWICH_DEPLOY_ROOT/tools/python"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export ALPHAMOTION_ENV="$GREENWICH_DEPLOY_ROOT/envs/alphamotion"
export MOTION_PERCEPTION_ENV="$GREENWICH_DEPLOY_ROOT/envs/genmo"
export PATH="$ALPHAMOTION_ENV/bin:$GREENWICH_DEPLOY_ROOT/tools/bin:$PATH"
