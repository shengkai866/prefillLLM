# Llama-3.2-1B-Instruct -> Llama-3.1-8B-Instruct pair (sourced after slurm/env.sh). Token pools depend on the tokenizer,
# so this pair has its own data tree on scratch; raw (untokenised) data is shared with the Qwen tree via symlinks.
export PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Llama-3.2-1B-Instruct
export PREFILL_LARGE_DIR=$PREFILL_ROOT/models/Llama-3.1-8B-Instruct
export PREFILL_DATA_DIR=$PREFILL_ROOT/data_llama
mkdir -p $PREFILL_DATA_DIR
for d in gsm8k longbench quality; do [ -e $PREFILL_DATA_DIR/$d ] || ln -s $PREFILL_ROOT/data/$d $PREFILL_DATA_DIR/$d; done
