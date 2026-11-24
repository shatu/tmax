#!/usr/bin/env bash
# Server setup for Olmo 3.5 MoE (Olmo3MoeForCausalLM, e.g. the A4B "Small" checkpoints) on vLLM, which no PyPI vLLM
# serves (checked through 0.34.0). Recreates what olmo-eval's TB Lite / SWE-bench cells install at job start
# (ledger H259's h259-a4b-*-tblite-*-keepthink-b cells, olmo-eval image ai2-tylerm/olmo-eval-cu1281-trc2100-amd64-sandbox):
# vllm 0.19.1 in /opt/vllm-venv on the image's own torch 2.10 + CUDA libraries (symlinked from /opt/venv), plus
# flash-linear-attention 0.5.2, transformers 5.16.1, OLMo-core 0c85d71c and the olmoe3 vLLM/transformers plugins
# from the ladders tarball the cells pin by sha256. Needs an image built on olmo-eval's (its /opt/venv) and
# /weka/oe-adapt-default. Run by run_eval_in_job.sh (launch_eval.sh --server-setup setup_olmoe3_vllm.sh)
# before the server starts; serve with --vllm-bin /opt/vllm-venv/bin/vllm.
set -euo pipefail
VENV=/opt/vllm-venv
LADDERS_TAR=/weka/oe-adapt-default/abhishekr/handoff/termination-2x2/ladders-3542d99.tar
LADDERS_SHA256=12117833da3cc502df5be6bef511165e7a4536090e7956bf4414a88b1a5a822f
OLMO_CORE_COMMIT=0c85d71c9ff2b8bf9aa34b279639d6514e1715a8

if [ -x "$VENV/bin/vllm" ] && "$VENV/bin/python" -c 'import olmoe3_vllm_plugin' 2>/dev/null; then
    echo "setup_olmoe3_vllm: $VENV already set up"; exit 0
fi
[ -x /opt/venv/bin/python ] || { echo "setup_olmoe3_vllm: no /opt/venv (needs an image built on olmo-eval's)" >&2; exit 1; }
echo "$LADDERS_SHA256  $LADDERS_TAR" | sha256sum -c -

uv venv --python /opt/venv/bin/python "$VENV"
for pkg in /opt/venv/lib/python*/site-packages/torch* /opt/venv/lib/python*/site-packages/nvidia*; do
    ln -sf "$pkg" "$VENV"/lib/python*/site-packages/
done
# fastapi < 0.137: newer fastapi breaks vLLM's prometheus middleware (see run_eval_in_job.sh).
uv pip install --python "$VENV/bin/python" 'vllm==0.19.1' 'fastapi<0.137'

mkdir -p /tmp/hero-ladder
tar -xf "$LADDERS_TAR" -C /tmp/hero-ladder
[ -d /tmp/hero-core/.git ] || git clone --quiet https://github.com/allenai/OLMo-core.git /tmp/hero-core
git -C /tmp/hero-core checkout --quiet "$OLMO_CORE_COMMIT"
uv pip install --python "$VENV/bin/python" 'flash-linear-attention==0.5.2' 'transformers==5.16.1' 'huggingface_hub==1.16.1' \
    -e /tmp/hero-core -e /tmp/hero-ladder/ladders/olmoe3/vllm_plugin -e /tmp/hero-ladder/ladders/olmoe3/transformers_plugin
"$VENV/bin/python" -c 'import vllm, transformers, fla, torch; print("setup_olmoe3_vllm: vllm", vllm.__version__, "transformers", transformers.__version__, "torch", torch.__version__)'
