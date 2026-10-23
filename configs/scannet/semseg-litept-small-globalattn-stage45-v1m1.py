_base_ = ["./semseg-litept-small-globalattn-v1m1.py"]

# Replace both attention-enabled encoder stages (stages 4 and 5).
save_path = "exp/scannet/semseg-litept-small-globalattn-stage45-v1m1"

model = dict(
    backbone=dict(
        enc_global_attn_stages=2,
    )
)
