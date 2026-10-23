_base_ = ["./semseg-litept-small-relflex-v1m1.py"]

# Identical mini-ScanNet protocol to the baseline subset smoke config.
batch_size = 1
num_worker = 4
mix_prob = 0
epoch = 1
eval_epoch = 1
seed = 0
enable_wandb = False

save_path = "exp/scannet/semseg-litept-small-relflex-subset-smoke-v1m1"
