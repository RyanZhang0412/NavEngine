#!/usr/bin/env python3
"""弯道控制日志分析（调参后对比用）。

用法：
    .venv/bin/python analyze_curve.py logs/xxx.log
    .venv/bin/python analyze_curve.py logs/xxx.log logs/yyy.log   # 多文件对比

输出：
    1. 执行层核对（指令 vs odom 推算）——可选，有 odom 行才出
    2. P/A 符号诊断：深弯帧按 lat·head 分组，看衰减是否打到该打的地方
    3. 每个弯道事件的 peak / 出弯段 yaw / demand / lat 收敛情况
    4. demand 塌陷帧（|head|>60 且 |demand|<0.03）列表
"""
from __future__ import annotations

import math
import re
import statistics as st
import sys
from pathlib import Path


def parse(path: Path) -> list[dict]:
    frames: list[dict] = []
    cur: dict = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            m = re.match(r"\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d\.\d+) --- frame ---", line)
            if m:
                cur = {"ts": m.group(1)}
                continue
            if not cur:
                # 顶部的 config 行
                m2 = re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+   CFG (.+)", line)
                if m2:
                    frames.append({"_config": m2.group(1)})
                continue
            m = re.match(r"run=(\w+)", line)
            if m:
                cur["run"] = m.group(1)
            m = re.match(r"(\w+) \| (.*)", line)
            if m and "mode" not in cur:
                cur["mode"] = m.group(1).lower()
            m = re.match(
                r"fwd=([+-][\d.]+) scale=([\d.]+) yaw=([+-][\d.]+) "
                r"\(demand=([+-][\d.]+) raw=([+-][\d.]+) sat=([YN])\)",
                line,
            )
            if m:
                cur.update(
                    fwd=float(m.group(1)),
                    scale=float(m.group(2)),
                    yaw=float(m.group(3)),
                    demand=float(m.group(4)),
                    raw=float(m.group(5)),
                    sat=m.group(6),
                )
            m = re.match(r"lat=([+-][\d.]+|nan) head=([+-][\d.]+|nan)", line)
            if m:
                try:
                    cur["lat"] = float(m.group(1))
                    cur["head"] = float(m.group(2))
                except ValueError:
                    cur["lat"] = cur["head"] = float("nan")
            m = re.match(
                r"P=([+-][\d.]+) D=([+-][\d.]+) A=([+-][\d.]+) dt=([\d.]+)s lat_f=([+-][\d.]+)",
                line,
            )
            if m:
                cur.update(
                    P=float(m.group(1)),
                    Dt=float(m.group(2)),
                    A=float(m.group(3)),
                    dt=float(m.group(4)),
                    lat_f=float(m.group(5)),
                )
                frames.append(cur)
                cur = {}
    return frames


def find_curves(run: list[dict]) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []
    in_e = False
    s = None
    for i, f in enumerate(run):
        if abs(f["head"]) > 30:
            if not in_e:
                in_e = True
                s = i
        else:
            if in_e and i - s >= 8:
                events.append((s, i - 1))
            in_e = False
    if in_e and len(run) - s >= 8:
        events.append((s, len(run) - 1))
    return events


def analyze(path: Path) -> None:
    print(f"\n{'='*70}\n{path.name}\n{'='*70}")
    fr = parse(path)
    cfg = next((f for f in fr if "_config" in f), {})
    if cfg:
        print(f"CFG: {cfg['_config']}")
    fr = [f for f in fr if "_config" not in f]
    run = [
        f for f in fr
        if f.get("run") == "True" and not math.isnan(f.get("head", float("nan")))
    ]
    if not run:
        print("(无 run=True 的有效帧)")
        return
    print(f"有效行驶帧: {len(run)}")

    # ── 1. P/A 符号诊断 ──
    deep = [f for f in run if abs(f["head"]) > 60]
    if deep:
        g_conflict = [f for f in deep if f["lat"] * f["head"] > 0]  # P 对抗 A，该衰减
        g_help = [f for f in deep if f["lat"] * f["head"] < 0]      # P 助攻 A，不该衰减
        print(f"\n[P/A 符号诊断] 深弯帧 |head|>60: {len(deep)}")
        for tag, g in [("lat·head>0 (P对抗A, 该衰减)", g_conflict),
                       ("lat·head<0 (P助攻A, 不该衰减)", g_help)]:
            if not g:
                print(f"  {tag}: 0帧")
                continue
            demand_abs = [abs(f["demand"]) for f in g]
            pa_over_a = [
                abs(f["P"] + f["A"]) / abs(f["A"])
                for f in g if abs(f["A"]) > 0.01
            ]
            print(f"  {tag}: {len(g)}帧")
            print(f"    |demand| mean={st.mean(demand_abs):.3f}  "
                  f"|P+A|/|A|={st.mean(pa_over_a):.2f} "
                  f"(<1=P抵消A, >1=P帮忙A)")

    # ── 2. 弯道事件 ──
    evts = find_curves(run)
    print(f"\n[弯道事件] 共 {len(evts)} 个")
    big = sorted(
        evts,
        key=lambda se: -max(abs(run[k]["head"]) for k in range(se[0], se[1] + 1)),
    )[:6]
    big.sort(key=lambda se: se[0])
    for ei, (s, e) in enumerate(big):
        seg = run[s : e + 1]
        peak = max(range(len(seg)), key=lambda k: abs(seg[k]["head"]))
        sign = "L" if seg[peak]["head"] < 0 else "R"
        tail = seg[peak + 2 :]
        if len(tail) < 5:
            print(f"  弯#{ei+1}({sign}) peak|head|={abs(seg[peak]['head']):.0f} "
                  f"(出弯段太短)")
            continue
        ayaw = [abs(f["yaw"]) for f in tail]
        ademand = [abs(f["demand"]) for f in tail]
        alat = [f["lat"] for f in tail]
        end_lat = seg[-1]["lat"]
        print(
            f"  弯#{ei+1}({sign}) peak|head|={abs(seg[peak]['head']):.0f} "
            f"出弯{len(tail)}帧: |yaw|={st.mean(ayaw):.3f} "
            f"|demand|={st.mean(ademand):.3f} "
            f"lat[{min(alat):+.0f},{max(alat):+.0f}] end={end_lat:+.0f}"
        )

    # ── 3. demand 塌陷帧 ──
    collapsed = [f for f in deep if abs(f["demand"]) < 0.03]
    print(f"\n[demand 塌陷帧] (|head|>60 且 |demand|<0.03): {len(collapsed)}帧")
    for f in collapsed[:6]:
        pa = (f["P"] + f["A"]) / abs(f["A"]) if abs(f["A"]) > 0.01 else 0
        print(
            f"  {f['ts']} lat={f['lat']:+.0f} head={f['head']:+.0f} "
            f"P={f['P']:+.3f} A={f['A']:+.3f} demand={f['demand']:+.3f} "
            f"(P+A)/|A|={pa:+.2f}"
        )


    # ── 4. 直道验收（Claude 建议的回归项）──
    straight = [f for f in run if abs(f["head"]) <= 15]
    print(f"\n[直道验收] |head|<=15: {len(straight)}帧")
    if straight:
        lats = [f["lat"] for f in straight]
        yaws = [f["yaw"] for f in straight]
        lat_std = st.pstdev(lats) if len(lats) > 1 else 0.0
        dyaw = [abs(yaws[i] - yaws[i - 1]) for i in range(1, len(yaws))]
        print(f"  lat std: {lat_std:.1f}px  (目标 <15)")
        print(f"  帧间 |Δyaw| mean/max: {st.mean(dyaw):.4f}/{max(dyaw):.4f}")
        print(f"  lat 范围: [{min(lats):+.0f}, {max(lats):+.0f}]")
    else:
        print("  (无足够直道帧)")

    # ── 5. 掉头捕获后收线（大偏移瞬态）──
    # 找 lat 突然从大值出现的段（TURN→FOLLOW 切换后）
    big_lat_straight = [
        f for f in straight if abs(f["lat"]) > 80 and abs(f["head"]) < 30
    ]
    print(f"\n[收线瞬态] 直道但 |lat|>80 且 |head|<30: {len(big_lat_straight)}帧")
    if big_lat_straight:
        # 这些帧的 P 衰减是否被误触发？看 yaw 是否正常修偏
        yaw_sign_ok = sum(
            1 for f in big_lat_straight
            if (f["lat"] < 0 and f["yaw"] > 0) or (f["lat"] > 0 and f["yaw"] < 0)
        )
        print(f"  yaw 方向正确（修偏）: {yaw_sign_ok}/{len(big_lat_straight)}")
        print(f"  |yaw| mean: {st.mean(abs(f['yaw']) for f in big_lat_straight):.3f}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        # 默认拿最新的
        log_dir = Path(__file__).resolve().parent / "logs"
        latest = sorted(log_dir.glob("follow_line_yolo2_*.log"))[-1]
        print(f"(未指定，用最新: {latest.name})")
        args = [str(latest)]
    for p in args:
        analyze(Path(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
