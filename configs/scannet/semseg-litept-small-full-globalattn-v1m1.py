_base_ = ["./semseg-litept-small-full-v1m1.py"]

save_path = "exp/scannet/semseg-litept-small-full-globalattn-v1m1"

# Replace local serialized-window attention at the final encoder stage.
model = dict(
    backbone=dict(
        enc_attention_type="global",
        enc_global_attn_stages=1,
    )
)
