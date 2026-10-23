_base_ = ["./semseg-litept-small-v1m1.py"]

# Preliminary 54-train / 10-validation mini-ScanNet experiment only.
# This is not a full ScanNet benchmark configuration.
batch_size = 1
num_worker = 4
mix_prob = 0
epoch = 5
eval_epoch = 5
seed = 0
enable_wandb = False

# Ten times lower than the training-from-scratch schedule for short fine-tuning.
optimizer = dict(type="AdamW", lr=0.0006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0006, 0.00006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.00006)]

save_path = "exp/scannet/semseg-litept-small-subset-finetune-v1m1"
