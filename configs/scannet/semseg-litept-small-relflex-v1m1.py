_base_ = ["./semseg-litept-small-v1m1.py"]

# Replace PointROPE + local FlashAttention with PointROPE + RelFlexformer
# attention at the final encoder stage only. Increase
# enc_global_attn_stages to 2 for stages 4+5.
save_path = "exp/scannet/semseg-litept-small-relflex-v1m1"

model = dict(
    backbone=dict(
        enc_attention_type="relflex",
        enc_global_attn_stages=1,
        relflex_num_frequencies=8,
        relflex_lambda=1.0,
        # Coordinates are in metres. Scaling controls the Fourier phase range.
        relflex_coord_scale=0.01,
        # Relative floor for stochastic-mask row normalization.
        relflex_eps=0.01,
        relflex_seed=0,
    )
)
