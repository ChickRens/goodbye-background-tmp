# sam2-bench（使い捨てのスパイク）

SAM2.1-tiny の動画トラッキングを CPU で計測し、高速化の手段を比べます。

## ファイル

```
sam2-bench/
  Dockerfile
  bench.py      計測本体
  convert.py    fp16→fp32 変換 / INT8 量子化
  data/input.mp4
```

## 比較モード（おすすめ）

1回の実行で 6 通りの設定を、最初の 4 秒分で比べます（15〜20 分ほどかかります）。

| 設定 | 内容 |
|---|---|
| fp32 / 全フレーム | 基準。fp16 の重みを fp32 に戻したモデル |
| fp16 / 全フレーム | 前回と同じモデル |
| int8 / 全フレーム | 行列積の重みを 8bit 整数にしたモデル |
| fp32 / 2枚ごと | 半分のフレームだけ推論し、間を補間 |
| int8 / 2枚ごと・3枚ごと | INT8 と間引きの組み合わせ |

### Windows で直接実行する場合

```powershell
pip install onnxruntime==1.30.0 opencv-python-headless==4.14.0.94 numpy==2.4.6 onnx==1.19.1
python bench.py --video data\input.mp4 --point 0.51,0.44 --compare --models models --out data\out
```

### Docker で実行する場合

```powershell
docker build -t sam2-bench .
docker run --rm -v ${PWD}\data:/data -v ${PWD}\models:/models `
  sam2-bench --video /data/input.mp4 --point 0.51,0.44 --compare
```

- 初回は fp32 変換と INT8 量子化に数分かかります。変換済みのモデルは `models/fp32`、`models/int8` に残り、2回目以降は使い回します。
- 1つの設定だけ測るときは `--compare` の代わりに `--precision int8 --stride 2 --secs 10` のように指定します。
- 4コア8スレッドの CPU では、`--threads 4` の方が速いことがあります。

## 出力（`data/out/`）

- `report.txt` … 比較表。これをチャットに貼ってください
- `greenback_<設定>.mp4` … 設定ごとのグリーンバック動画。見た目の違いを確認します
- IoU は基準（fp32 / 全フレーム）のマスクとどれだけ一致するか。0.9 以上なら見た目はほぼ同じです
