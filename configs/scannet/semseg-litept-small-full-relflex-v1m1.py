_base_ = ["./semseg-litept-small-full-v1m1.py"]

save_path = "exp/scannet/semseg-litept-small-full-relflex-v1m1"

# Replace PointROPE + local FlashAttention with PointROPE + RelFlexformer
# at the final encoder stage.
model = dict(
    backbone=dict(
        enc_attention_type="relflex",
        enc_global_attn_stages=1,
        relflex_num_frequencies=8,
        relflex_lambda=1.0,
        relflex_coord_scale=0.01,
        relflex_eps=0.01,
        relflex_seed=0,
    )
)
