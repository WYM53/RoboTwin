#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import zarr

def classify_arm(action):
    L = action[:, 0:6]
    R = action[:, 7:13]
    ltv = float(np.abs(np.diff(L, axis=0)).sum()) if len(L) > 1 else 0.0
    rtv = float(np.abs(np.diff(R, axis=0)).sum()) if len(R) > 1 else 0.0
    eps = 1e-9
    if ltv > 3.0 * max(rtv, eps):
        return "left", ltv, rtv
    if rtv > 3.0 * max(ltv, eps):
        return "right", ltv, rtv
    return "ambiguous", ltv, rtv

def blue_centroid_rgb(img):
    # Zarr head_camera is expected as CHW uint8.
    hwc = np.moveaxis(img, 0, -1)
    hsv = cv2.cvtColor(hwc, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([80, 50, 40], dtype=np.uint8),
        np.array([130, 255, 255], dtype=np.uint8),
    )
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 30:
            comps.append((area, cents[i], stats[i]))
    if not comps:
        return None, 0, hwc
    comps.sort(key=lambda x: x[0], reverse=True)
    area, center, _ = comps[0]
    return (float(center[0]), float(center[1])), area, hwc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/place_empty_cup-dp_test_demo-50.zarr")
    ap.add_argument("--out", default="training_visibility")
    args = ap.parse_args()

    root = zarr.open(args.zarr, mode="r")
    imgs = root["data/head_camera"]
    actions = root["data/action"]
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    thumbs = []
    start = 0
    for ep, end in enumerate(ends):
        a = np.asarray(actions[start:end])
        arm, ltv, rtv = classify_arm(a)

        center, area, hwc = blue_centroid_rgb(np.asarray(imgs[start]))
        vis = "not_detected"
        if center is not None:
            x, y = center
            w = hwc.shape[1]
            if x < 15 or x > w - 16:
                vis = "edge"
            else:
                vis = "visible"
        else:
            x = y = float("nan")

        rows.append({
            "episode": ep,
            "arm": arm,
            "start_index": int(start),
            "end_index": int(end),
            "cup_px_x": x,
            "cup_px_y": y,
            "blue_area_px": area,
            "visibility": vis,
            "left_total_variation": ltv,
            "right_total_variation": rtv,
        })

        canvas = hwc.copy()
        text = f"ep{ep:02d} {arm[0].upper()} {vis}"
        cv2.rectangle(canvas, (0,0), (180,20), (0,0,0), -1)
        cv2.putText(canvas, text, (3,15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
        if center is not None:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 5, (255,0,0), 2)
        thumbs.append(canvas)
        start = int(end)

    csv_path = out / "training_initial_visibility.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    h, w = thumbs[0].shape[:2]
    cols = 5
    nrows = math.ceil(len(thumbs)/cols)
    sheet = np.full((nrows*h, cols*w, 3), 255, dtype=np.uint8)
    for i, im in enumerate(thumbs):
        rr, cc = divmod(i, cols)
        sheet[rr*h:(rr+1)*h, cc*w:(cc+1)*w] = im
    montage_path = out / "training_initial_montage.png"
    cv2.imwrite(str(montage_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    # Console summary
    print("zarr:", args.zarr)
    print("episodes:", len(rows))
    for arm in ["left", "right", "ambiguous"]:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        print(f"\n[{arm}] n={len(sub)}")
        vals = [r["cup_px_x"] for r in sub if np.isfinite(r["cup_px_x"])]
        if vals:
            print(f"cup_px_x min/median/max: {min(vals):.1f} / {np.median(vals):.1f} / {max(vals):.1f}")
        print("visibility:",
              {k: sum(r["visibility"] == k for r in sub)
               for k in ["visible","edge","not_detected"]})

    print("\nSaved:")
    print(csv_path)
    print(montage_path)

if __name__ == "__main__":
    main()
