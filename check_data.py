#!/usr/bin/env python3
"""检查跑某个序列需要的全部数据是否齐全（登录节点用宿主机 python3 直接跑，不需要容器）。

    python3 check_data.py            # 默认 0103
    python3 check_data.py 0103 0106  # 多个序列
    python3 check_data.py --extract 0278 0331   # 缺的传感器文件直接从原始 zip 精确解压（只解需要的文件）

检查项：
  NCAP   : v1.0-trainval 元数据表、can_bus 的 pose / zoe_veh_info、场景 yaml
  NeuRAD : 该场景所有相机和激光雷达帧（关键帧 + sweeps）、地图、checkpoint
  UniAD  : 权重
缺文件时会去原始 blob zip 里找它在哪个包，并打印解压命令。
"""
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

BASE = Path("/nobackup/proj/disk/cvls/personal/shipengl/ad")
DST = BASE / "neurad-studio/data/nuscenes"
ORIG = Path("/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip")
EXTRACT = "--extract" in sys.argv
SEQS = [a for a in sys.argv[1:] if a != "--extract"] or ["0103"]
SCENARIOS = ["frontal", "side", "stationary"]

ok_all = True
def report(ok, msg):
    global ok_all
    print(("  ✓ " if ok else "  ✗ ") + msg)
    ok_all &= ok

def is_file(p):  # 失效软链也算缺失
    return os.path.isfile(p) and os.path.getsize(p) > 0

print("== 元数据表 v1.0-trainval ==")
TABLES = ["attribute", "calibrated_sensor", "category", "ego_pose", "instance", "log", "map",
          "sample", "sample_annotation", "sample_data", "scene", "sensor", "visibility"]
for t in TABLES:
    p = DST / "v1.0-trainval" / f"{t}.json"
    report(is_file(p), str(p.relative_to(DST)))
if not ok_all:
    sys.exit("元数据表不全，先补齐再检查")

print("  读取元数据（sample_data.json 较大，约需 1 分钟）...", flush=True)
load = lambda t: json.load(open(DST / "v1.0-trainval" / f"{t}.json"))
scenes = {s["name"]: s for s in load("scene")}
samples = load("sample")
logs = {l["token"]: l for l in load("log")}
maps = load("map")
calib = {c["token"]: c["sensor_token"] for c in load("calibrated_sensor")}
sensors = {s["token"]: s["channel"] for s in load("sensor")}
sample_data = load("sample_data")

missing_sensor_files = []
for seq in SEQS:
    name = f"scene-{int(seq):04d}"
    print(f"\n========== {name} ==========")
    if name not in scenes:
        report(False, f"{name} 不在 scene.json 里"); continue
    scene = scenes[name]
    log = logs[scene["log_token"]]
    print(f"  log: {log['logfile']}  location: {log['location']}")

    # ---- NCAP: can_bus ----
    print("-- NCAP --")
    for kind in ["pose", "zoe_veh_info"]:
        p = DST / "can_bus" / f"{name}_{kind}.json"
        report(is_file(p), str(p.relative_to(DST)))
    found = [c for c in SCENARIOS if (BASE / f"neuro-ncap/scenarios/{c}/{seq}.yaml").is_file()]
    report(bool(found), f"场景文件 scenarios/*/{seq}.yaml  ({', '.join(found) or '无'})")

    # ---- NeuRAD: 传感器文件 ----
    print("-- NeuRAD --")
    sample_tokens = {s["token"] for s in samples if s["scene_token"] == scene["token"]}
    per_channel = defaultdict(lambda: [0, 0])
    for sd in sample_data:
        if sd["sample_token"] not in sample_tokens:
            continue
        ch = sensors[calib[sd["calibrated_sensor_token"]]]
        ok = is_file(DST / sd["filename"])
        per_channel[ch][0] += 1
        if not ok:
            per_channel[ch][1] += 1
            if not ch.startswith("RADAR"):
                missing_sensor_files.append(sd["filename"])
    for ch in sorted(per_channel):
        total, miss = per_channel[ch]
        if ch.startswith("RADAR"):
            print(f"  · {ch:18s} {total - miss}/{total}（NeuRAD 不用雷达，缺了无所谓）")
        else:
            report(miss == 0, f"{ch:18s} {total - miss}/{total}")

    map_files = [m["filename"] for m in maps if scene["log_token"] in m["log_tokens"]]
    for mf in map_files:
        report(is_file(DST / mf), f"地图 {mf}")

    ck = BASE / f"neurad-studio/checkpoints/{seq}"
    report(is_file(ck / "config.yml"), f"checkpoints/{seq}/config.yml")
    ckpts = list(ck.rglob("*.ckpt"))
    report(bool(ckpts), f"checkpoints/{seq}/**/*.ckpt  ({len(ckpts)} 个)")

print("\n== UniAD ==")
report(is_file(BASE / "UniAD/checkpoints/uniad_base_e2e.pth"), "UniAD/checkpoints/uniad_base_e2e.pth")

# ---- 缺失的传感器文件去原始 zip 里找 ----
if missing_sensor_files:
    print(f"\n== 缺 {len(missing_sensor_files)} 个传感器文件，去原始 zip 里找 ==")
    want = set(missing_sensor_files)
    where = {}
    for z in sorted(ORIG.glob("*.zip")):
        if "blobs" not in z.name and "sweeps" not in z.name and "samples" not in z.name:
            continue
        try:
            names = zipfile.ZipFile(z).namelist()
        except Exception as e:
            print(f"  打不开 {z.name}: {e}"); continue
        hit = want.intersection(names)
        if hit:
            where[z.name] = hit
            want -= hit
        if not want:
            break
    for zname, members in where.items():
        print(f"  {zname}: {len(members)} 个")
    if EXTRACT:
        for zname, members in where.items():
            print(f"  解压 {zname} 中的 {len(members)} 个文件 ...", flush=True)
            with zipfile.ZipFile(ORIG / zname) as zf:
                for i, m in enumerate(sorted(members), 1):
                    if not is_file(DST / m):
                        zf.extract(m, DST)
                    if i % 1000 == 0:
                        print(f"    {i}/{len(members)}", flush=True)
        print("  解压完成，请再运行一次本脚本（不带 --extract）确认全部齐全")
    else:
        print("  加 --extract 重新运行即可只解压这些文件")
    if want:
        print(f"  还有 {len(want)} 个在所有 zip 里都没找到，例如: {sorted(want)[:3]}")

print("\n" + ("全部齐全 ✓" if ok_all else "有缺失 ✗，按上面提示补齐"))