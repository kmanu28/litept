_base_ = ["./semseg-litept-small-v1m1.py"]

# Mini-ScanNet pipeline smoke test only: 54 train / 10 validation scenes.
# This is not a full ScanNet benchmark configuration.
batch_size = 1
num_worker = 4
mix_prob = 0
epoch = 1
eval_epoch = 1
seed = 0
enable_wandb = False

save_path = "exp/scannet/semseg-litept-small-subset-smoke-v1m1"
