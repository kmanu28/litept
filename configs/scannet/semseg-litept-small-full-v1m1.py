_base_ = ["./semseg-litept-small-v1m1.py"]

# Full official ScanNet v2: 1201 train / 312 validation scenes.
# Batch size and workers are reduced for the 8 GiB laptop GPU.
batch_size = 1
num_worker = 4
seed = 0
enable_wandb = False

data_root = "data/scannet_full"
data = dict(
    train=dict(data_root=data_root),
    val=dict(data_root=data_root),
    test=dict(data_root=data_root),
)

save_path = "exp/scannet/semseg-litept-small-full-v1m1"
