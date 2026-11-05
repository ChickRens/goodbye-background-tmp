"""SAM2.1-tiny 動画トラッキングのサーバー側(CPU)ベンチマーク。使い捨てのスパイク。

ブラウザ版 index.html で正しさを確認したループを、そのまま numpy + onnxruntime に移したもの。
本番の Rust (ort クレート) も同じ ONNX Runtime の C++ 実装を使うので、推論時間はほぼ同じになる。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from convert import prepare

REPO = "https://huggingface.co/diffusionstudio/sam2.1-tiny-video-onnx-fp16/resolve/main/"
GRAPHS = ["vision_encoder", "mask_decoder", "memory_encoder", "memory_attention", "pointer_tpos"]
S, F = 512, 32
F2 = F * F
MEM_FRAMES, MAX_PTRS = 7, 16
GREEN_BGR = (64, 177, 0)  # クロマグリーン (0,177,64)

NP_TYPES = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int32)": np.int32,
    "tensor(int64)": np.int64, "tensor(bool)": np.bool_,
}


# ---------------- モデル ----------------
def fetch_models(model_dir: Path) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ["constants.json"] + [f"onnx/{g}.onnx" for g in GRAPHS]:
        dst = model_dir / Path(name).name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        print(f"ダウンロード: {name}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(REPO + name, tmp)
        tmp.rename(dst)
    return json.loads((model_dir / "constants.json").read_text())


DEVICE = "cpu"


class Graph:
    def __init__(self, path: Path, threads: int):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.name = path.stem
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if DEVICE == "cuda"
                     else ["CPUExecutionProvider"])
        self.s = ort.InferenceSession(str(path), so, providers=providers)
        if DEVICE == "cuda" and self.s.get_providers()[0] != "CUDAExecutionProvider":
            raise SystemExit(f"{self.name}: GPU(CUDA) を使えませんでした。使えたのは {self.s.get_providers()}")
        self.inputs = {i.name: i for i in self.s.get_inputs()}

    def run(self, **feeds) -> dict:
        fixed = {}
        for k, v in feeds.items():
            meta = self.inputs[k]
            arr = np.asarray(v, dtype=NP_TYPES[meta.type])
            if all(isinstance(d, int) for d in meta.shape) and int(np.prod(meta.shape)) == arr.size:
                arr = arr.reshape(meta.shape)  # 例: binarize のスカラー形状
            fixed[k] = arr
        names = [o.name for o in self.s.get_outputs()]
        out = self.s.run(names, fixed)
        return {n: (o.astype(np.float32) if o.dtype == np.float16 else o) for n, o in zip(names, out)}


# ---------------- 純粋関数 (index.html の CORE と同じ) ----------------
def to_pixel_values(frame_bgr: np.ndarray, mean, std) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (S, S), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - np.array(mean, np.float32)) / np.array(std, np.float32)
    return x.transpose(2, 0, 1)[None]


def chw_to_tokens(feats: np.ndarray) -> np.ndarray:  # [1,C,F,F] → [F²,1,C]
    c = feats.shape[1]
    return feats.reshape(c, F2).T.reshape(F2, 1, c)


def assemble_memory(t, cond, recent, tpos):
    blocks = [(cond, 6)]
    for k in range(1, 7):
        f = t - k
        if f < 0:
            break
        if f == cond["frame"]:
            continue
        if f in recent:
            blocks.append((recent[f], k - 1))
    most_recent = blocks[1] if len(blocks) > 1 else blocks[0]
    while len(blocks) < MEM_FRAMES:
        blocks.append(most_recent)
    mem = np.concatenate([b["tokens"].reshape(F2, 64) for b, _ in blocks])
    pos = np.concatenate([b["pos"].reshape(F2, 64) + tpos[row][None] for b, row in blocks])
    return mem, pos


def pointer_list(t, total, cond, recent_ptrs):
    max_ptrs = min(total, MAX_PTRS)
    denom = max(max_ptrs - 1, 1)
    lst = [(cond["ptr"], abs(t - cond["frame"]))]
    for d in range(1, max_ptrs):
        f = t - d
        if f < 0:
            break
        if f == cond["frame"]:
            continue
        if f in recent_ptrs:
            lst.append((recent_ptrs[f], d))
    most_recent = lst[1] if len(lst) > 1 else lst[0]
    while len(lst) < MAX_PTRS:
        lst.append(most_recent)
    ptrs = np.stack([p for p, _ in lst])                       # [16,256]
    diffs = np.array([d / denom for _, d in lst], np.float32)  # [16]
    return ptrs, diffs


# ---------------- 動画 ----------------
def sample_frames(path: str, fps: float, secs: float):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"動画を開けません: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(min(secs, cap.get(cv2.CAP_PROP_FRAME_COUNT) / src_fps) * fps)
    idx, want = 0, 0
    while want < total:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= round(want * src_fps / fps):
            yield want, frame
            want += 1
        idx += 1
    cap.release()


def composite_green(frame_bgr, low_logits):
    h, w = frame_bgr.shape[:2]
    m = cv2.resize(low_logits, (w, h), interpolation=cv2.INTER_LINEAR) > 0
    out = np.empty_like(frame_bgr)
    out[:] = GREEN_BGR
    out[m] = frame_bgr[m]
    return out


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "不明"


def gpu_name() -> str:
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "不明"
    except Exception:
        return "不明"


def stats(xs):
    if not xs:
        return "—"
    a = np.array(xs)
    return f"平均 {a.mean():.0f} ms / 中央値 {np.median(a):.0f} ms / 最大 {a.max():.0f} ms"


# ---------------- トラッキング本体 ----------------
def track(g, frames, pts, tpos, mean, std, stride=1, on_frame=None):
    """frames[::stride] で推論し、間のフレームはマスクのロジットを線形補間する。
    戻り値: 全出力フレーム分の low-res ロジット [256,256] のリスト、計測値"""
    infer = frames[::stride]
    total = len(infer)
    T = {k: [] for k in ["enc", "attn", "dec", "menc", "frame"]}
    lost, logits = [], []
    cond, recent, recent_ptrs = None, {}, {}
    for i, frame in enumerate(infer):
        a = time.perf_counter()
        enc = g["vision_encoder"].run(pixel_values=to_pixel_values(frame, mean, std))
        b = time.perf_counter()
        if i == 0:
            feats_cond = enc["feats2_no_mem"]
            coords = np.array([[[[x, y] for x, y, _ in pts]]], np.float32)
            labels = np.array([[[l for *_, l in pts]]], np.int32)
        else:
            mem, pos = assemble_memory(i, cond, recent, tpos)
            ptrs, diffs = pointer_list(i, total, cond, recent_ptrs)
            ppos = g["pointer_tpos"].run(normalized_diffs=diffs)["pointer_pos"]
            mem = np.concatenate([mem, ptrs.reshape(MAX_PTRS * 4, 64)])
            pos = np.concatenate([pos, np.repeat(ppos, 4, axis=0)])
            feats_cond = g["memory_attention"].run(
                current_vision_features=chw_to_tokens(enc["feats2"]),
                current_vision_position_embeddings=chw_to_tokens(enc["vision_pos_embed"]),
                memory=mem[:, None, :], memory_pos=pos[:, None, :],
            )["conditioned_feats"]
            coords = np.zeros((1, 1, 1, 2), np.float32)
            labels = np.full((1, 1, 1), -1, np.int32)
        c = time.perf_counter()
        dec = g["mask_decoder"].run(feats0=enc["feats0"], feats1=enc["feats1"], feats2_cond=feats_cond,
                                    input_points=coords, input_labels=labels)
        d = time.perf_counter()
        score = float(dec["object_score_logits"].reshape(-1)[0])
        me = g["memory_encoder"].run(feats2=enc["feats2"], high_res_mask=dec["high_res_mask"],
                                     object_score_logits=[[score]], binarize=1 if i == 0 else 0)
        e = time.perf_counter()
        ptr = dec["object_pointer"].reshape(-1)[:256]
        if i == 0:
            cond = {"frame": 0, "tokens": me["memory_tokens"], "pos": me["memory_pos"], "ptr": ptr}
        else:
            recent[i] = {"tokens": me["memory_tokens"], "pos": me["memory_pos"]}
            recent_ptrs[i] = ptr
            recent.pop(i - 7, None)
            recent_ptrs.pop(i - MAX_PTRS, None)
            T["attn"].append((c - b) * 1000)
        if score <= 0:
            lost.append(i * stride)
        logits.append(dec["low_res_mask"].reshape(256, 256).astype(np.float32))
        T["enc"].append((b - a) * 1000)
        T["dec"].append((d - c) * 1000)
        T["menc"].append((e - d) * 1000)
        T["frame"].append((e - a) * 1000)
        if on_frame:
            on_frame(i + 1, total, T["frame"][-1])
    out = []
    for j in range(len(frames)):
        k, r = divmod(j, stride)
        if r == 0 or k + 1 >= len(logits):
            out.append(logits[min(k, len(logits) - 1)])
        else:
            w = r / stride
            out.append((1 - w) * logits[k] + w * logits[k + 1])
    return out, T, lost


def iou(a, b):
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def write_video(path: Path, frames, logits, fps):
    writer = None
    for frame, low in zip(frames, logits):
        green = composite_green(frame, low)
        if writer is None:
            h, w = green.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(green)
    writer.release()


def load_graphs(model_dir: Path, precision: str, threads: int):
    d = prepare(model_dir, GRAPHS, precision)
    return {n: Graph(d / f"{n}.onnx", threads) for n in GRAPHS}


def progress(label):
    return lambda i, n, ms: print(f"\r  {label}: {i}/{n}  {ms:.0f} ms   ", end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--point", action="append", required=True,
                    help="x,y[,label] を 0〜1 の比率で。label は 1=含める(既定) 0=除外。複数指定可")
    ap.add_argument("--fps", type=float, default=15, help="出力の fps")
    ap.add_argument("--secs", type=float, default=None, help="処理する秒数（既定: 通常 10、--compare 時 4）")
    ap.add_argument("--precision", choices=["fp16", "fp32", "int8"], default="fp32")
    ap.add_argument("--stride", type=int, default=1, help="何フレームごとに推論するか（2 なら半分だけ推論して補間）")
    ap.add_argument("--compare", action="store_true", help="複数の設定をまとめて比較する")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--models", default="/models")
    ap.add_argument("--out", default="/data/out")
    args = ap.parse_args()
    secs = args.secs or (4 if args.compare else 10)
    global DEVICE
    DEVICE = args.device
    if DEVICE == "cuda" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()   # pip で入れた CUDA / cuDNN を読み込む

    model_dir = Path(args.models)
    consts = fetch_models(model_dir)
    tpos = np.array(consts["memory_temporal_positional_encoding"], np.float32)
    mean, std = consts["image_mean"], consts["image_std"]
    pts = []
    for p in args.point:
        v = [float(x) for x in p.split(",")]
        pts.append((v[0] * S, v[1] * S, int(v[2]) if len(v) > 2 else 1))
    frames = [f for _, f in sample_frames(args.video, args.fps, secs)]
    if len(frames) < 4:
        raise SystemExit("処理できるフレームが少なすぎます")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.compare:
        variants = [(args.precision, args.stride)]
    elif DEVICE == "cuda":
        variants = [("fp32", 1), ("fp16", 1)]          # GPU では fp16 が本命。int8・間引きは除外
    else:
        variants = [("fp32", 1), ("fp16", 1), ("int8", 1), ("fp32", 2), ("int8", 2), ("int8", 3)]
    head = [
        "# SAM2 サーバー(CPU)ベンチ結果" + (" — 比較" if args.compare else ""),
        f"CPU: {cpu_name()} / 論理コア {os.cpu_count()} / 使用スレッド {args.threads} / onnxruntime {ort.__version__}",
        f"実行: {DEVICE}" + (f" / GPU: {gpu_name()}" if DEVICE == "cuda" else ""),
        f"動画: {len(frames)} フレーム @{args.fps:g}fps ({secs:g}秒分) / クリック {len(pts)} 点",
        "",
        "設定 | 1出力フレームあたり | 10秒動画1本(15fps) | 基準との一致(IoU 平均/最小) | 見失い",
        "---|---|---|---|---",
    ]
    print("\n".join(head))
    rows, base, details, cache = [], None, [], {}
    for prec, stride in variants:
        label = f"{prec} / {stride}枚ごと推論" if stride > 1 else f"{prec} / 全フレーム推論"
        if prec not in cache:
            t0 = time.perf_counter()
            cache = {prec: load_graphs(model_dir, prec, args.threads)}   # 1種類ずつ持ってメモリを節約
            details.append(f"{prec} 読み込み: {(time.perf_counter() - t0) * 1000:.0f} ms")
        w0 = time.perf_counter()
        logits, T, lost = track(cache[prec], frames, pts, tpos, mean, std, stride, progress(label))
        wall = time.perf_counter() - w0
        print()
        # 初回（GPU の初期化・最適化を含む）を除いた、出力1フレームあたりの時間
        per_out = (wall * 1000 - T["frame"][0]) / max(1, len(frames) - stride)
        if base is None:
            base = logits
        ious = [iou(a, b) for a, b in zip(logits, base)]
        row = (f"{label} | {per_out:.0f} ms | 約 {per_out * 150 / 1000:.0f} 秒 | "
               f"{np.mean(ious):.3f} / {np.min(ious):.3f} | {len(lost)}")
        rows.append(row)
        print(row)
        details.append(f"{label}: 初回 {T['frame'][0]:.0f} ms / エンコーダ {stats(T['enc'][1:])} / メモリ注意 {stats(T['attn'])} / "
                       f"デコーダ {stats(T['dec'])} / メモリエンコーダ {stats(T['menc'])}")
        tag = f"{prec}_s{stride}"
        write_video(out_dir / f"greenback_{tag}.mp4", frames, logits, args.fps)
        cv2.imwrite(str(out_dir / f"frame0_{tag}.png"), composite_green(frames[0], logits[0]))

    report = "\n".join(head + rows + ["", "## 内訳"] + details + [
        "",
        "IoU は最初の設定（" + f"{variants[0][0]} / {variants[0][1]}枚ごと" + "）のマスクとの一致度。1.000 が完全一致、0.9 以上なら見た目はほぼ同じ",
        f"出力: {out_dir}",
    ])
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
