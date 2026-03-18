# Rapsodo → Nathan Excel 変換（最小依存）

`MyBallTrajectorySim.py` 全体に依存せず、**時計→角度**と **角速度→B,S,G** と **rapsodo_to_nathan** のみを同梱します。

## ファイル

| ファイル | 内容 |
|----------|------|
| `clock_time_to_angle_deg.py` | Rapsodo 時計表記 → 角度 [deg] |
| `pitch_parameters_bsg.py` | `PitchParameters` + `angular_velocity_xyz_to_backspin_sidespin_wg` |
| `rapsodo_to_nathan.py` | Rapsodo 入力 → Nathan Excel 用 `PitchParameters` / 1行出力 |

## セットアップ

```bash
cd code
pip install -r requirements.txt
python rapsodo_to_nathan.py
```

## Git に載せる場合

リポジトリルートで `code/` を追加し、上記3ファイル + `requirements.txt` + `README.md` をコミットしてください。

親ディレクトリの `../rapsodo_to_nathan.py` は、この `code/rapsodo_to_nathan.py` を読み込む薄いラッパーです（後方互換）。
