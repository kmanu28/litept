_base_ = ["./semseg-litept-small-full-globalattn-v1m1.py"]

# Four deterministic yaw views; the test pipeline averages per-point logits.
save_path = "exp/scannet/semseg-litept-small-full-globalattn-tta-v1m1"
data = dict(
    test=dict(
        test_cfg=dict(
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1)],
            ]
        )
    )
)
