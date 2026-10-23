_base_ = ["./semseg-litept-small-relflex-subset-smoke-v1m1.py"]

evaluate = False
save_path = "exp/scannet/semseg-litept-small-relflex-subset-profile-v1m1"

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=3),
    dict(type="InformationWriter"),
    dict(type="TorchProfilerTraining", wait=1, warmup=2, active=10, row_limit=30),
]
