#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser(description="Launcher for Approx Borrow Size Analysis (BVAP).")
    ap.add_argument("--script", default="approx_borrow_size_analysis.py",
                    help="Target analysis script to execute.")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--topN", type=int, default=9)
    ap.add_argument("--overview-rank", choices=["borrow","dev_cum","alpha"], default="dev_cum")
    ap.add_argument("--detail-order", choices=["overview","alpha","borrow","dev_cum"], default="overview")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--x-mode", choices=["time","index"], default="time")
    ap.add_argument("--borrow-view", choices=["absolute","delta"], default="absolute")
    ap.add_argument("--delta-baseline", choices=["first","median"], default="first")
    ap.add_argument("--borrow-zoom-frac", type=float, default=None)
    ap.add_argument("--borrow-floor-mode", choices=["none","p10","min"], default="none")

    args, rest = ap.parse_known_args()

    cmd = [
        sys.executable, args.script,
        "--data-dir", args.data_dir,
        "--days", str(args.days),
        "--topN", str(args.topN),
        "--overview-rank", args.overview_rank,
        "--detail-order", args.detail_order,
        "--x-mode", args.x_mode,
        "--borrow-view", args.borrow_view,
        "--delta-baseline", args.delta_baseline,
        "--borrow-floor-mode", args.borrow_floor_mode,
    ]
    if args.detail:
        cmd.append("--detail")
    if args.borrow_zoom_frac is not None:
        cmd += ["--borrow-zoom-frac", str(args.borrow_zoom_frac)]

    cmd += rest

    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
