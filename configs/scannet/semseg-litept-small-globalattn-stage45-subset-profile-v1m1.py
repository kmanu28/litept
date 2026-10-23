_base_ = ["./semseg-litept-small-globalattn-stage45-v1m1.py"]

# Real-loop profile on the 54-scene mini-ScanNet training subset.
batch_size = 1
num_worker = 4
mix_prob = 0
epoch = 1
eval_epoch = 1
seed = 0
enable_wandb = False
evaluate = False

save_path = "exp/scannet/semseg-litept-small-globalattn-stage45-subset-profile-v1m1"

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=3),
    dict(type="InformationWriter"),
    dict(type="TorchProfilerTraining", wait=1, warmup=2, active=10, row_limit=30),
]
