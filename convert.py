"""モデル変換: fp16 → fp32、Gemm → MatMul+Add、INT8 動的量子化。

- fp32: CPU は fp16 の計算が苦手なので、重みと内部計算をすべて fp32 に戻す（計算結果はほぼ同じ）
- int8: fp32 化したうえで Gemm を MatMul+Add に書き換え（動的量子化は Gemm を対象にしないため）、
        MatMul の重みを 8bit 整数にする
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, helper

F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT


def _tensor_to_f32(t: onnx.TensorProto) -> onnx.TensorProto:
    if t.data_type != F16:
        return t
    arr = numpy_helper.to_array(t).astype(np.float32)
    return numpy_helper.from_array(arr, t.name)


def _fix_type(tp: onnx.TypeProto):
    if tp.HasField("tensor_type") and tp.tensor_type.elem_type == F16:
        tp.tensor_type.elem_type = F32


def _graph_to_f32(g: onnx.GraphProto):
    for i, t in enumerate(g.initializer):
        if t.data_type == F16:
            g.initializer[i].CopyFrom(_tensor_to_f32(t))
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        _fix_type(vi.type)
    for n in g.node:
        for a in n.attribute:
            if n.op_type == "Cast" and a.name == "to" and a.i == F16:
                a.i = F32
            elif a.type == onnx.AttributeProto.TENSOR and a.t.data_type == F16:
                a.t.CopyFrom(_tensor_to_f32(a.t))
            elif a.type == onnx.AttributeProto.GRAPH:
                _graph_to_f32(a.g)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for sg in a.graphs:
                    _graph_to_f32(sg)
            elif a.name in ("dtype", "to") and a.type == onnx.AttributeProto.INT and a.i == F16:
                a.i = F32


def to_fp32(src: Path, dst: Path):
    m = onnx.load(str(src))
    _graph_to_f32(m.graph)
    onnx.save(m, str(dst))


def gemm_to_matmul(m: onnx.ModelProto) -> int:
    """重みが定数の Gemm を MatMul(+Add) に置き換える。書き換えた数を返す。"""
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    new_nodes, count = [], 0
    for n in g.node:
        at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        ok = (n.op_type == "Gemm" and at.get("transA", 0) == 0
              and at.get("alpha", 1.0) == 1.0 and at.get("beta", 1.0) == 1.0 and n.input[1] in inits)
        if not ok:
            new_nodes.append(n)
            continue
        w = numpy_helper.to_array(inits[n.input[1]])
        if at.get("transB", 0):
            w = w.T
        wname = n.input[1] + "_mm"
        g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(w), wname))
        has_bias = len(n.input) > 2 and n.input[2] != ""
        mm_out = n.output[0] + "_mm" if has_bias else n.output[0]
        new_nodes.append(helper.make_node("MatMul", [n.input[0], wname], [mm_out], name=f"{n.name or 'gemm'}_{count}_mm"))
        if has_bias:
            new_nodes.append(helper.make_node("Add", [mm_out, n.input[2]], [n.output[0]], name=f"{n.name or 'gemm'}_{count}_add"))
        count += 1
    del g.node[:]
    g.node.extend(new_nodes)
    return count


def to_int8(fp32_src: Path, dst: Path) -> int:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    m = onnx.load(str(fp32_src))
    n = gemm_to_matmul(m)
    tmp = dst.with_suffix(".mm.onnx")
    onnx.save(m, str(tmp))
    quantize_dynamic(str(tmp), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    tmp.unlink()
    return n


def prepare(model_dir: Path, graphs: list[str], precision: str) -> Path:
    """precision 用のモデルフォルダを返す（なければ作る）。"""
    if precision == "fp16":
        return model_dir
    fp32_dir = model_dir / "fp32"
    fp32_dir.mkdir(exist_ok=True)
    for g in graphs:
        if not (fp32_dir / f"{g}.onnx").exists():
            print(f"fp32 に変換: {g}")
            to_fp32(model_dir / f"{g}.onnx", fp32_dir / f"{g}.onnx")
    if precision == "fp32":
        return fp32_dir
    int8_dir = model_dir / "int8"
    int8_dir.mkdir(exist_ok=True)
    for g in graphs:
        if (int8_dir / f"{g}.onnx").exists():
            continue
        if g == "pointer_tpos":  # 小さすぎて量子化の意味がない
            (int8_dir / f"{g}.onnx").write_bytes((fp32_dir / f"{g}.onnx").read_bytes())
            continue
        print(f"INT8 に量子化: {g}")
        n = to_int8(fp32_dir / f"{g}.onnx", int8_dir / f"{g}.onnx")
        print(f"  Gemm → MatMul 書き換え: {n} 個")
    return int8_dir
