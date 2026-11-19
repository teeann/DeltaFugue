set shell := ["bash", "-uc"]
set positional-arguments
set dotenv-load

python:='python3'
tokenizer:='meta-llama/Llama-2-7b-hf'
# Hugging Face cache for the pretraining data (fineweb-edu).
hf:='~/.cache/huggingface'

# Pretraining run selected by `just flame_train`.
config:='configs/delta_fugue_340M_ts16.json'
dump_folder:='exp/delta_fugue_340M_ts16'

lr:='1e-3'
train_step:='28672'
warmup_steps:='1024'
context_len:='4096'
seqlen:='32768'
grad_accum:='4'  # seqlen * grad_accum * NGPU(4) = 0.5M tokens per step

install:
    cd fla-latest && {{python}} -m pip install -e .
    {{python}} -m pip install -r requirements.txt

install_flame:
    cd flame && {{python}} -m pip install .
    {{python}} -m pip uninstall torchtitan -y
    {{python}} -m pip install git+https://github.com/pytorch/torchtitan.git@0b44d4c
    {{python}} -m pip install tyro
    {{python}} -m pip install -U datasets

# Formal-language experiments (experiment_configs/lab.json).
lab:
    {{python}} run_experiment.py -c lab

lab_multi:
    {{python}} run_experiment.py -c lab --multi

test:
    cd fla-latest && pytest -s tests/models/test_modeling_delta_fugue.py tests/models/test_modeling_gated_deltafugue.py

# LM pretraining on fineweb-edu with flame / torchtitan (4 GPUs).
[no-cd]
flame_train:
    cd flame && HF_HOME={{hf}} NNODE=1 NGPU=4 bash train.sh \
        --job.config_file flame/models/fla.toml \
        --job.dump_folder {{dump_folder}} \
        --model.config {{config}} \
        --model.tokenizer_path {{tokenizer}} \
        --optimizer.name AdamW \
        --optimizer.eps 1e-15 \
        --optimizer.lr {{lr}} \
        --optimizer.weight_decay 0.01 \
        --lr_scheduler.warmup_steps {{warmup_steps}} \
        --lr_scheduler.lr_min 0.03 \
        --lr_scheduler.decay_type cosine \
        --training.batch_size 1 \
        --training.seq_len {{seqlen}} \
        --training.context_len {{context_len}} \
        --training.varlen \
        --training.gradient_accumulation_steps {{grad_accum}} \
        --training.steps {{train_step}} \
        --training.max_norm 1.0 \
        --training.skip_nan_inf \
        --training.dataset HuggingFaceFW/fineweb-edu \
        --training.dataset_name sample-100BT \
        --training.dataset_split train \
        --training.num_workers 32 \
        --training.prefetch_factor 2 \
        --training.seed 22 \
        --training.compile \
        --training.tensor_parallel_degree 1 \
        --checkpoint.interval 1024 \
        --training.disable_loss_parallel \
        --checkpoint.keep_latest_k 2 \
        --checkpoint.load_step -1 \
        --metrics.log_freq 1

# Convert a torchtitan DCP checkpoint to a Hugging Face checkpoint.
[no-cd]
convert path:
    cd flame && {{python}} -m flame.utils.convert_dcp_to_hf \
        --path {{path}} \
        --step {{train_step}} \
        --config {{config}} \
        --tokenizer {{tokenizer}}

# Zero-shot evaluation with lm-evaluation-harness (pip install lm-eval).
lm_eval path:
    {{python}} run_lm_eval.py --model hf \
        --model_args pretrained={{path}},trust_remote_code=True,dtype=bfloat16 \
        --tasks arc_easy,arc_challenge,winogrande,piqa,hellaswag,wikitext,lambada_openai
