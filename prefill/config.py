"""Central paths and model names. Everything lives under SCRATCH_ROOT."""
import os

SCRATCH_ROOT = os.environ.get("PREFILL_ROOT", "/scratch/drjieliu_root/drjieliu/shengkai/prefillLLM")

MODELS_DIR = os.path.join(SCRATCH_ROOT, "models")
# Token pools depend on the tokenizer: another model family (Llama) uses its own tree, e.g. PREFILL_DATA_DIR=$PREFILL_ROOT/data_llama
DATA_DIR = os.environ.get("PREFILL_DATA_DIR", os.path.join(SCRATCH_ROOT, "data"))
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
# Whose outputs the projected cache is trained and scored against: "large" (the model that decodes from the cache; the
# default) or "small" (the model that prefills). "small" is for reverse pairs, e.g. PREFILL_SMALL_DIR=Qwen3-8B,
# PREFILL_LARGE_DIR=Qwen3-0.6B, PREFILL_TEACHER=small: the 8B prefills, the 0.6B decodes and should answer like the 8B.
# "small" / "large" keep their code meaning (source of hidden states / owner of the KV cache), not their size.
TEACHER = os.environ.get("PREFILL_TEACHER", "large")
assert TEACHER in ("large", "small"), TEACHER
TEACHER_NAME = SMALL_NAME if TEACHER == "small" else LARGE_NAME
# Split the large model over several GPUs for training (e.g. 32B): PREFILL_LARGE_MAXMEM="24GiB,76GiB" loads it with
# device_map="auto" under these per-GPU caps; the small model, projector and losses stay on cuda:0.
LARGE_MAXMEM = os.environ.get("PREFILL_LARGE_MAXMEM", "")
# Same for the small (prefilling) model, e.g. a reverse pair whose source is Qwen3-30B-A3B (61 GB, > one 48 GB L40S):
# PREFILL_SMALL_MAXMEM="22GiB,40GiB" splits it layer-wise over GPUs 0..n-1; the large model (decoder), projector and
# losses stay on cuda:0. Hidden states are gathered on cuda:0 (kv.small_hidden_states, fast.FastQwen3).
SMALL_MAXMEM = os.environ.get("PREFILL_SMALL_MAXMEM", "")

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
