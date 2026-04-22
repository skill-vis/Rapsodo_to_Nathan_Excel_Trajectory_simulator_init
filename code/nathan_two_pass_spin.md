# 2 パス角速度推定 — 計算方法レポート

## 目的

MLB Statcast（あるいは HawkEye）の投球データから、ボールの 3D 角速度ベクトル $\vec{\omega} = (\omega_x, \omega_y, \omega_z)$ を **既存 Nathan 1 パス法よりも高精度に推定**する。
手法は Nathan (2020) 論文 [`spinaxis-2.pdf`](../tmp/spinaxis-2.pdf) の式を基礎に、**Pass 1 で得た粗推定 $\vec\omega$ を用いて軌道を数値積分し、そこから $\langle\vec a_M\rangle, \langle\vec v\rangle$ を再評価して Pass 2 で $\omega_T$ を refine** する。

---

## 記号・物理定数

| 記号 | 意味 | 値 |
|---|---|---|
| $m$ | ボール質量 | 0.145 kg |
| $r$ | ボール半径 | 0.037 m |
| $A$ | ボール断面積 $\pi r^2$ | — |
| $\rho$ | 空気密度 | 1.2 kg/m³ |
| $K$ | $=\frac{1}{2}\rho A/m$ | — (eq 2) |
| $\vec g$ | 重力加速度 | $(0, 0, -9.80)$ m/s² |
| $C_{L,A}, C_{L,B}$ | Nathan fit 定数 | 0.336, 6.041 |
| $\omega$ | 総スピン (Trackman 測定値) | [rpm] |
| $\omega_T$ | transverse spin | [rad/s] |
| $\omega_G$ | gyrospin | [rad/s] |

座標: PITCHf/x（$+x$ = 捕手の右、$+y$ = 投手方向、$+z$ = 鉛直上向き）

---

## Pass 1 (既存 Nathan 1 パス法)

### 入力
9 パラメータ定加速度フィット $(x_0, y_0, z_0, v_{x0}, v_{y0}, v_{z0}, a_x, a_y, a_z)$ と総スピン $\omega$ [rpm]、投手利き $\in\{R, L\}$。

### 手順

**S1. 飛行時間と平均速度**（定加速度モデル）

$$
y(t) = y_0 + v_{y0}t + \tfrac{1}{2}a_y t^2, \qquad
t_{\text{flight}} = \text{smallest positive root of } y(t) = y_{\text{plate}}
$$

$$
\langle \vec v\rangle_1 = \vec v_0 + \vec a \cdot t_{\text{flight}}/2,\qquad
\langle v\rangle_1 = |\langle\vec v\rangle_1|, \qquad
\hat{v}_1 = \langle\vec v\rangle_1 / \langle v\rangle_1
$$

**S2. Magnus 加速度** — Nathan 論文 (6)

$$
\boxed{\vec a_M = \vec a - \vec g + [(\vec a - \vec g)\cdot\hat v]\,\hat v} \quad (\text{定加速度}\ \vec a\ \text{を使用}) \qquad\cdots(6)
$$

大きさ $a_M = |\vec a_M|$、方向単位ベクトル $\hat a_M = \vec a_M / a_M$。

**S3. 揚力係数** — Nathan (7)

$$
C_L = \frac{a_M}{K\langle v\rangle^2} \qquad\cdots(7)
$$

**S4. $\omega_T$ 算出** — Nathan (8)(9)

Nathan fit: $C_L = C_{L,A}\bigl[1 - \exp(-C_{L,B}\,S)\bigr]$, ここで $S = r\,\omega_T / \langle v\rangle$。

$$
\boxed{\omega_T = \frac{\langle v\rangle}{r\,C_{L,B}}\ln\!\left(\frac{C_{L,A}}{C_{L,A} - C_L}\right)} \qquad\cdots(9)
$$

**S5. $\omega_T$ 方向** — Nathan (10)

$$
\boxed{\hat\omega_T = \hat v \times \hat a_M} \qquad\cdots(10)
$$

**S6. $\omega_G$ の大きさ** — Nathan (4)

$$
\omega_G = \sqrt{\omega^2 - \omega_T^2} \qquad\cdots(4)
$$

**S7. $\omega_G$ 方向** — Nathan 論文の heuristic: 投手利き $R \Rightarrow \hat\omega_G = +\hat v$, $L \Rightarrow \hat\omega_G = -\hat v$

**S8. 合成** — Nathan (11)

$$
\boxed{\vec\omega = \omega_T\hat\omega_T + \omega_G\hat\omega_G} \qquad\cdots(11)
$$

成分 $(\omega_x, \omega_y, \omega_z) = \vec\omega$。

---

## Pass 2 (軌道シミュレーション経由の refine)

### 発想
式 (6) を Pass 1 で使う際、$\vec a$ を「9 パラメータ **定加速度** フィットの単一値」としている。
実際の $\vec a$ は速度減衰に伴い時々刻々変化するため、**$\langle\vec a_M\rangle$ の正しい平均値**を得るには、物理シミュレータで積分した軌道から時間平均する方が精確。

### 手順

**S9. Pass 1 の $\vec\omega$ を初期値として RK4 積分**

`MyBallTrajectorySim.BallTrajectorySimulator2` を RK4 モードで回し、各時刻 $t_i$ での $(\vec r_i, \vec v_i, \vec a_i)$ を取得。使用する spin モデル: $C_L = C_{L,A}[1-\exp(-C_{L,B}\,S(t))]$（時刻ごとに $S(t) = r\omega_T/v(t)$ で再評価）。

**S10. 時刻ごとの Magnus 加速度**

$$
\vec a_M(t_i) = \vec a(t_i) - \vec g - \bigl[(\vec a(t_i) - \vec g)\cdot\hat v(t_i)\bigr]\hat v(t_i)
\qquad\cdots(6')
$$

**S11. 時間平均**

$$
\langle\vec a_M\rangle = \frac{1}{N}\sum_{i=1}^{N} \vec a_M(t_i), \quad
\langle\vec v\rangle = \frac{1}{N}\sum_{i=1}^{N} \vec v(t_i)
$$

$\langle v\rangle = |\langle\vec v\rangle|$, $\hat v = \langle\vec v\rangle / \langle v\rangle$, $a_M = |\langle\vec a_M\rangle|$, $\hat a_M = \langle\vec a_M\rangle / a_M$.

**S12. $C_L$ と $\omega_T$ 再計算** — (7)(9) を同形で適用

$$
C_L^{(2)} = \frac{a_M}{K\langle v\rangle^2}, \qquad
\omega_T^{(2)} = \frac{\langle v\rangle}{r\,C_{L,B}}\ln\!\left(\frac{C_{L,A}}{C_{L,A} - C_L^{(2)}}\right)
$$

**S13. $\hat\omega_T^{(2)}, \omega_G^{(2)}, \vec\omega^{(2)}$ 合成**

S5〜S8 と同じ式（(10)(4)(11)）を Pass 2 の $\hat v, \hat a_M, \omega_T$ で再計算。

---

## 違い（Pass 1 と Pass 2）

| 量 | Pass 1 | Pass 2 |
|---|---|---|
| $\vec v(t), \vec a(t)$ | 定加速度モデル（9-param fit 由来、単一値）| RK4 軌道の時系列 |
| $\vec a_M$ | 9-param $\vec a$ から 1 点で (6) 計算 | (6') で各 $t_i$ に計算後、時間平均 |
| $C_L$ | (7) に $a_M$ と $\langle v\rangle_1$（1 点評価）| (7) に時間平均値 |
| $\omega_T$ | (9) から直接 | (7)(8)(9) を Pass2 の平均値で再計算 |
| $\omega_G, \vec\omega$ | (4)(10)(11) | 同一式、Pass2 の $\omega_T, \hat v, \hat a_M$ で |

---

## 期待される改良

9-param は「定 $\vec a$」近似なので、速度減衰の効果が $\vec a_M$ 推定に入らない。Pass 2 の RK4 では速度が減衰するにつれ $C_L, C_D$ が現実的に変化する。このため $\langle\vec a_M\rangle$ の**時間平均**は Pass 1 の単一値より物理的真値に近づく可能性がある。
ただし反復 1 回（2 パス）での改良効果は入力データ品質や球種に依存する。

## 制限・注意

- **反復回数**: 現在 2 パス固定。Pass 2 の $\vec\omega$ で再度シミュレート → Pass 3... と収束させる実装も可能だが未導入。
- **$\omega_G$ 方向**: Nathan 論文の heuristic（RHP → $+\hat v$）に従う。左投手や特殊 delivery で誤る可能性あり。
- **シミュレータの数値整合性**: Pass 1 の $\vec\omega$ を $(B, S, G)_{\text{rpm}}$ に変換し simulator に渡す際、`MyBallTrajectorySim` 内の `sqrt(spin_{\text{total}}^2 - \omega_G^2)` が浮動小数誤差で負化し `ValueError` が出るケースあり（16 箇所 clipping なし、要対処）。

## 実装ファイル

- [`nathan_two_pass_spin.py`](nathan_two_pass_spin.py) — Pass 1/Pass 2 推定関数
- [`nathan_vs_hawkeye.py`](nathan_vs_hawkeye.py) — HawkEye .action の実測 $\vec\omega$ との比較
