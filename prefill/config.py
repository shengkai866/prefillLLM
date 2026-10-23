"""Central paths and model names. Everything lives under SCRATCH_ROOT."""
import os

SCRATCH_ROOT = os.environ.get("PREFILL_ROOT", "/scratch/drjieliu_root/drjieliu/shengkai/prefillLLM")

MODELS_DIR = os.path.join(SCRATCH_ROOT, "models")
DATA_DIR = os.path.join(SCRATCH_ROOT, "data")
CKPT_DIR = os.path.join(SCRATCH_ROOT, "checkpoints")
# Small human-readable outputs (summary.md, results.json) live with the code; big artifacts stay on scratch.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
HF_CACHE_DIR = os.path.join(SCRATCH_ROOT, "hf_cache")

SMALL_REPO = "Qwen/Qwen3-0.6B"
LARGE_REPO = "Qwen/Qwen3-8B"
# The small (prefill) model can be swapped per job, e.g. PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Qwen3-1.7B
SMALL_DIR = os.environ.get("PREFILL_SMALL_DIR", os.path.join(MODELS_DIR, "Qwen3-0.6B"))
SMALL_NAME = os.path.basename(SMALL_DIR.rstrip("/")).replace("Qwen3-", "")  # "0.6B", "1.7B", ... for labels
LARGE_DIR = os.environ.get("PREFILL_LARGE_DIR", os.path.join(MODELS_DIR, "Qwen3-8B"))
LARGE_NAME = os.path.basename(LARGE_DIR.rstrip("/")).replace("Qwen3-", "")
# Split the large model over several GPUs for training (e.g. 32B): PREFILL_LARGE_MAXMEM="24GiB,76GiB" loads it with
# device_map="auto" under these per-GPU caps; the small model, projector and losses stay on cuda:0.
LARGE_MAXMEM = os.environ.get("PREFILL_LARGE_MAXMEM", "")

# dataset name -> (hf repo, config name, text field)
DATASETS = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    # cerebras/SlimPajama-627B now returns 401 on the Hub; DKYoon/SlimPajama-6B is a public 6B-token subsample of it
    "slimpajama": ("DKYoon/SlimPajama-6B", None, "text"),
}
GSM8K_DIR = os.path.join(DATA_DIR, "gsm8k")
SELFDISTILL_DIR = os.path.join(DATA_DIR, "selfdistill")

# Sequence budget for the first round of experiments (4k prompt + 256 continuation).
MAX_PROMPT_LEN = 4096
MAX_SEQ_LEN = MAX_PROMPT_LEN + 256
