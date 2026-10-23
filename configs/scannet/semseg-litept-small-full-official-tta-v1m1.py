_base_ = ["./semseg-litept-small-full-v1m1.py"]
save_path = "exp/scannet/official-litept-s-tta-v1m1"
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
