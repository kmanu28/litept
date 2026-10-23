_base_ = ["./semseg-litept-small-v1m1.py"]

# Replace local serialized-window attention at the final encoder stage only.
# Increase this to 2 to target both attention-enabled LitePT-S encoder stages.
save_path = "exp/scannet/semseg-litept-small-globalattn-v1m1"

model = dict(
    backbone=dict(
        enc_attention_type="global",
        enc_global_attn_stages=1,
    )
)
