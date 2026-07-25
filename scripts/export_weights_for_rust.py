#!/usr/bin/env python3
"""Export trained MetaMLModel weights to JSON for the Rust geolane_twin.

Usage:
    python scripts/export_weights_for_rust.py checkpoint.pth output.json

Layer order below is the order the Rust model reads parameters in. Each entry is
(rust_layer_name, python_state_dict_prefix_or_None, [in_dim, out_dim]):
  * a real prefix  -> weight+bias copied from the Python checkpoint;
  * a None prefix  -> the Rust model has this layer but the Python model does not,
                      so it is exported as zeros (placeholder).

NOTE: `head_prominence` (theta_heads.peak_prominence) was added to the Python
MetaMLModel after the initial Rust port. It is appended LAST so existing Rust
layer indices (0..6) stay stable; the Rust geolane_twin model must add a matching
trailing head before it can consume this parameter (not wired up yet).
"""

import json
import sys
import torch


# (rust_layer_name, python_prefix_or_None, [in, out])
LAYERS = [
    ("fc1",              "feature_extractor.0",            [5, 128]),
    ("fc2",              "feature_extractor.3",            [128, 128]),
    ("head_width_scale", "theta_heads.width_scale",        [128, 1]),
    ("head_consistency", "theta_heads.consistency_weight", [128, 1]),
    ("head_triplet",     "theta_heads.triplet_margin",     [128, 1]),
    ("head_smoothing",   "theta_heads.smoothing_factor",   [128, 1]),
    ("head_sigma",       None,                             [128, 1]),  # not in Python — zeros
    ("head_prominence",  "theta_heads.peak_prominence",    [128, 1]),  # added post-port (see note)
    ("head_min_evidence","theta_heads.min_lane_evidence",  [128, 1]),  # recall lever, added last
]

LAYER_DIMS = [dims for (_, _, dims) in LAYERS]


def export_weights(checkpoint_path: str, output_path: str):
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in cp:
        sd = cp["model_state_dict"]
    elif "state_dict" in cp:
        sd = cp["state_dict"]
    else:
        sd = cp

    print(f"State dict keys: {list(sd.keys())}")

    data = []

    for name, prefix, (in_dim, out_dim) in LAYERS:
        if prefix is None:
            # Layer present in the Rust model but absent in Python — export zeros.
            n = in_dim * out_dim + out_dim  # weight + bias
            data.extend([0.0] * n)
            print(f"  {name}: zeros ({in_dim * out_dim} + {out_dim} = {n} params)")
            continue

        w_key = f"{prefix}.weight"
        b_key = f"{prefix}.bias"
        if w_key not in sd or b_key not in sd:
            raise KeyError(
                f"{prefix} missing from checkpoint (keys: {list(sd.keys())}). "
                f"Checkpoint predates this head? Retrain or adjust LAYERS."
            )

        w = sd[w_key].detach().float().numpy().flatten()
        b = sd[b_key].detach().float().numpy().flatten()
        data.extend(w.tolist())
        data.extend(b.tolist())
        print(f"  {name} <- {prefix}: weight{list(sd[w_key].shape)} + bias{list(sd[b_key].shape)} = {len(w) + len(b)} params")

    output = {
        "data": data,
        "layer_dims": LAYER_DIMS,
    }

    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"\nExported {len(data)} parameters to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <checkpoint.pth> <output.json>")
        sys.exit(1)

    export_weights(sys.argv[1], sys.argv[2])
