_base_ = ["./semseg-litept-small-full-globalattn-v1m1.py"]

save_path = "exp/scannet/semseg-litept-small-full-globalattn-pretrained-ft-v1m1"

# Conservative fine-tuning schedule for the official ScanNet LitePT-S weights.
optimizer = dict(type="AdamW", lr=0.0006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0006, 0.00006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
