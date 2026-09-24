#!/usr/bin/env python3
import argparse
import numpy as np
import zarr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zarr",
        default="data/place_empty_cup-dp_test_demo-50.zarr",
        help="Path to DP zarr dataset"
    )
    args = ap.parse_args()

    root = zarr.open(args.zarr, mode="r")
    action = np.asarray(root["data/action"][:], dtype=np.float64)
    state = np.asarray(root["data/state"][:], dtype=np.float64)
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)

    print("zarr:", args.zarr)
    print("action:", action.shape)
    print("state :", state.shape)
    print("episodes:", len(ends))
    print()

    print("===== per-dimension ACTION stats =====")
    for i in range(action.shape[1]):
        print(
            f"dim {i:2d}: "
            f"min={action[:, i].min(): .6f} "
            f"max={action[:, i].max(): .6f} "
            f"mean={action[:, i].mean(): .6f} "
            f"std={action[:, i].std(): .6f}"
        )

    print("\n===== per-episode arm activity =====")
    start = 0
    rows = []
    counts = {"left": 0, "right": 0, "both/ambiguous": 0}

    for ep, end in enumerate(ends):
        a = action[start:end]

        L = a[:, 0:6]
        LG = a[:, 6]
        R = a[:, 7:13]
        RG = a[:, 13]

        # Range captures how far each arm traverses over an episode.
        left_span = float(np.max(np.ptp(L, axis=0)))
        right_span = float(np.max(np.ptp(R, axis=0)))

        # Total variation captures amount of commanded motion.
        left_tv = float(np.abs(np.diff(L, axis=0)).sum()) if len(L) > 1 else 0.0
        right_tv = float(np.abs(np.diff(R, axis=0)).sum()) if len(R) > 1 else 0.0

        left_grip_span = float(np.ptp(LG))
        right_grip_span = float(np.ptp(RG))

        # Expert task is single-arm. Use a conservative ratio to classify.
        eps = 1e-9
        if left_tv > 3.0 * max(right_tv, eps):
            active = "left"
        elif right_tv > 3.0 * max(left_tv, eps):
            active = "right"
        else:
            active = "both/ambiguous"

        counts[active] += 1
        rows.append(
            (ep, start, int(end), active,
             left_span, right_span, left_tv, right_tv,
             left_grip_span, right_grip_span)
        )
        start = int(end)

    print(
        "ep  start   end   active          "
        "L_span    R_span    L_TV      R_TV      LG_span   RG_span"
    )
    for r in rows:
        print(
            f"{r[0]:2d} {r[1]:6d} {r[2]:6d} {r[3]:14s} "
            f"{r[4]:8.4f} {r[5]:8.4f} "
            f"{r[6]:9.4f} {r[7]:9.4f} "
            f"{r[8]:9.4f} {r[9]:9.4f}"
        )

    print("\n===== arm-count summary =====")
    for k, v in counts.items():
        print(f"{k:14s}: {v}")

    left_eps = [r for r in rows if r[3] == "left"]
    right_eps = [r for r in rows if r[3] == "right"]

    def med(vals):
        return float(np.median(vals)) if vals else float("nan")

    print("\n===== median movement =====")
    print("left-active episodes :", len(left_eps))
    print("  median L span:", med([r[4] for r in left_eps]))
    print("  median L TV  :", med([r[6] for r in left_eps]))
    print("right-active episodes:", len(right_eps))
    print("  median R span:", med([r[5] for r in right_eps]))
    print("  median R TV  :", med([r[7] for r in right_eps]))

    print("\n===== state/action alignment sanity =====")
    # process_data.py is intended to build state=q_t, action=q_{t+1}.
    # Quantify one-step delta magnitude globally.
    delta = action - state
    print("max |action-state| per dim:")
    print(np.max(np.abs(delta), axis=0))
    print("mean |action-state| per dim:")
    print(np.mean(np.abs(delta), axis=0))

if __name__ == "__main__":
    main()
