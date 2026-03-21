#!/usr/bin/env python3
"""
BallTrajectorySim_MKS.py を使ったシミュレーションと可視化の例

使い方:
  cd MyTrajectorySimulator
  python run_sim_and_visualize.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MyBallTrajectorySim import (
    BallTrajectorySimulator2,
    IntegrationMethod,
    PitchParameters,
    EnvironmentParameters,
)
import matplotlib
# GUI環境がない場合でも落ちないように Agg を使用
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TrajectoryCalculator-new-3D-May2021.xlsx 用の単位換算
FT_PER_M = 3.28084
MPH_PER_MPS = 2.23694  # 1 m/s = 2.23694 mph


def _traj_to_arrays(traj):
    xs = [p["x"] for p in traj]
    ys = [p["y"] for p in traj]
    zs = [p["z"] for p in traj]
    return xs, ys, zs


def plot_compare_projections(sim_rk4_spin, sim_nathan_spin, sim_rk4_nospin, sim_nathan_nospin, title="Trajectory projections comparison"):
    """
    RK4 と Nathan（Excel系差分）の軌道を、スピン有/無で 3 平面（XY/XZ/YZ）に比較表示する。
    """
    xs1, ys1, zs1 = _traj_to_arrays(sim_rk4_spin.trajectory)
    xs2, ys2, zs2 = _traj_to_arrays(sim_nathan_spin.trajectory)
    xs3, ys3, zs3 = _traj_to_arrays(sim_rk4_nospin.trajectory)
    xs4, ys4, zs4 = _traj_to_arrays(sim_nathan_nospin.trajectory)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle(title)

    # XY
    ax = axes[0]
    ax.plot(xs1, ys1, label="RK4 (spin)", lw=2)
    ax.plot(xs2, ys2, label="Nathan (spin)", lw=2, ls="--")
    ax.plot(xs3, ys3, label="RK4 (no spin)", lw=1.8, alpha=0.85)
    ax.plot(xs4, ys4, label="Nathan (no spin)", lw=1.8, ls="--", alpha=0.85)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("XY")
    ax.grid(True, alpha=0.3)

    # XZ
    ax = axes[1]
    ax.plot(xs1, zs1, label="RK4 (spin)", lw=2)
    ax.plot(xs2, zs2, label="Nathan (spin)", lw=2, ls="--")
    ax.plot(xs3, zs3, label="RK4 (no spin)", lw=1.8, alpha=0.85)
    ax.plot(xs4, zs4, label="Nathan (no spin)", lw=1.8, ls="--", alpha=0.85)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("XZ")
    ax.grid(True, alpha=0.3)

    # YZ
    ax = axes[2]
    ax.plot(ys1, zs1, label="RK4 (spin)", lw=2)
    ax.plot(ys2, zs2, label="Nathan (spin)", lw=2, ls="--")
    ax.plot(ys3, zs3, label="RK4 (no spin)", lw=1.8, alpha=0.85)
    ax.plot(ys4, zs4, label="Nathan (no spin)", lw=1.8, ls="--", alpha=0.85)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    # 右端サブ図のタイトルが線と重なりやすいので、少し下げる
    ax.set_title("YZ", y=0.96)
    ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    # 凡例（説明ボックス）を少し下げる：ボックス下端が y≈0.05 になるように配置
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.02, 0.05))
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    return fig


def print_excel_params(pitch, env):
    """
    現在の pitch / env を Excel「TrajectoryCalculator-new-3D-May2021.xlsx」用の
    ヤード・ポンド単位で表示する。Excel に同じ条件で入力して比較可能。
    """
    print("\n=== Excel (TrajectoryCalculator-new-3D-May2021.xlsx) 用パラメータ ===")
    print("  【投球】")
    print(f"    x0 (ft)        = {pitch.x0 * FT_PER_M:.4f}")
    print(f"    y0 (ft)        = {pitch.y0 * FT_PER_M:.4f}   # 投手からの距離")
    print(f"    z0 (ft)        = {pitch.z0 * FT_PER_M:.4f}   # リリース高さ")
    print(f"    v0 (mph)       = {pitch.v0_mps * MPH_PER_MPS:.2f}")
    print(f"    theta_deg      = {pitch.theta_deg:.2f}   # 負=下向き(Excelも同慣習の想定)")
    print(f"    phi_deg        = {pitch.phi_deg:.2f}")
    print(f"    backspin_rpm   = {pitch.backspin_rpm:.1f}")
    print(f"    sidespin_rpm   = {pitch.sidespin_rpm:.1f}")
    print(f"    wg_rpm         = {pitch.wg_rpm:.1f}")
    print("  【環境】")
    temp_C_from_F = (env.temp_F - 32.0) * (5.0 / 9.0)
    print(f"    temp (℃)      = {temp_C_from_F:.1f}  (Excel用 temp_F = {env.temp_F:.1f})")
    print(f"    elev_ft        = {env.elev_m * FT_PER_M:.2f}   # 標高")
    print(f"    relative_humidity = {env.relative_humidity:.1f}")
    # EnvironmentParameters は pressure_mmHg を使用
    pressure_inHg = env.pressure_mmHg / 25.4
    print(f"    pressure_inHg = {pressure_inHg:.2f}")
    print(f"    vwind_mph      = {env.vwind_mph:.1f}")
    print(f"    phiwind_deg    = {env.phiwind_deg:.1f}")
    print(f"    hwind_ft       = {env.hwind_m * FT_PER_M:.2f}")
    print("  ※ Excel のセル名はシートにより異なります。上記を対応する入力欄に手で入力してください。")


def run_example(show_plots=True):
    """
    1本の投球をシミュレートし、各種可視化を行う。
    """
    # 1. シミュレータの作成（RK4推奨）
    sim = BallTrajectorySimulator2(integration_method=IntegrationMethod.RK4)

    # 2. 投球パラメータ
    # theta_deg: リリース角度(deg)。野球慣習で 正=水平より下向き、負=上向き。
    pitch = PitchParameters(
        x0=-0.790991, # -0.309180,
        y0=18.44 - 2.111809, #1.958155,
        z0=1.401341 + 0.254,#1.659248 + 0.254, #1.829,
        v0_mps=36.8611, #436.8611,   km/h -> m/s
        theta_deg=1.06, #-1.47,
        phi_deg=3.53, # 0.17,
        backspin_rpm=2099.6, # 2152.3,
        sidespin_rpm=0.0,
        wg_rpm=0.0,
        batter_hand='R',
    )

    # 3. 環境パラメータ（気温は摂氏 °C で指定）
    temp_C = 21.1  # 気温 (℃)。例: 21.1 °C ≒ 70 °F
    env = EnvironmentParameters(
        temp_F=temp_C * (9.0 / 5.0) + 32.0,  # 摂氏→華氏でシミュレータに渡す
        elev_m=87.3, #4.572, 87.3は厚木
        relative_humidity=40.0,
        pressure_mmHg=760.0, # 1013.25 hPa 相当（標準）
        vwind_mph=0.0,
        phiwind_deg=0.0,
        hwind_m=0.0,
    )

    # Excel 用パラメータを表示（同じ条件を TrajectoryCalculator-new-3D-May2021.xlsx に入力可能）
    print_excel_params(pitch, env)

    # 4. シミュレーション実行（RK4 / Nathan、スピン有/無で4通り）
    sim_rk4_spin = BallTrajectorySimulator2(integration_method=IntegrationMethod.RK4)
    sim_nathan_spin = BallTrajectorySimulator2(integration_method=IntegrationMethod.NATHAN)

    pitch_nospin = PitchParameters(**{**pitch.__dict__, "backspin_rpm": 0.0, "sidespin_rpm": 0.0, "wg_rpm": 0.0})
    sim_rk4_nospin = BallTrajectorySimulator2(integration_method=IntegrationMethod.RK4)
    sim_nathan_nospin = BallTrajectorySimulator2(integration_method=IntegrationMethod.NATHAN)

    sim_rk4_spin.simulate(pitch=pitch, env=env, max_time=1.0, save_interval=1)
    sim_nathan_spin.simulate(pitch=pitch, env=env, max_time=1.0, save_interval=1)
    sim_rk4_nospin.simulate(pitch=pitch_nospin, env=env, max_time=1.0, save_interval=1)
    sim_nathan_nospin.simulate(pitch=pitch_nospin, env=env, max_time=1.0, save_interval=1)

    # 互換: 以降の既存表示は RK4(スピン有) を代表として使う
    sim = sim_rk4_spin

    # 4.5 初期速度ベクトルと仰角の確認（theta_deg が上下の初速の角度であることの検証用）
    if sim.trajectory:
        p0 = sim.trajectory[0]
        vx, vy, vz = p0['vx'], p0['vy'], p0['vz']
        v_horiz = (vx**2 + vy**2) ** 0.5
        elev_deg = math.degrees(math.atan2(vz, v_horiz)) if v_horiz > 0 else 0.0
        print("\n=== 初期速度ベクトル（theta_deg の反映確認） ===")
        print(f"  (vx, vy, vz) = ({vx:.3f}, {vy:.3f}, {vz:.3f}) m/s")
        print(f"  仰角（水平から、正=上/負=下）: {elev_deg:.2f} deg  （pitch.theta_deg={pitch.theta_deg} は正=下・負=上）")

    # 5. サマリー表示（m/s と km/h を併記）
    summary = sim.get_summary()
    if summary:
        v0_ms = summary['initial_velocity_mps']
        vend_ms = summary['final_velocity_mps']
        print("\n=== シミュレーション結果 ===")
        print(f"初速度: {v0_ms:.2f} m/s ({v0_ms * 3.6:.1f} km/h)")
        print(f"最終速度: {vend_ms:.2f} m/s ({vend_ms * 3.6:.1f} km/h)")
        print(f"最大高度: {summary['max_height']:.2f} m")
        print(f"終端高さ（Z）: {summary['final_position'][2]:.3f} m")
        print(f"総時間: {summary['total_time']:.3f} sec")
        if summary.get('home_plate_crossing'):
            h = summary['home_plate_crossing']
            v_home = h['v']
            print(f"ホームプレート通過: 速度 {v_home:.2f} m/s ({v_home * 3.6:.1f} km/h), 高さ z={h['z']:.3f} m")

    if not show_plots:
        return sim, summary

    # 6. 可視化
    # (0) 比較図（RK4 vs Nathan、スピン有/無、3平面）
    fig0 = plot_compare_projections(
        sim_rk4_spin, sim_nathan_spin, sim_rk4_nospin, sim_nathan_nospin,
        title="RK4 vs Nathan (with/without spin) — projections"
    )
    out_png = os.path.join(os.path.dirname(__file__), "compare_rk4_nathan_spin_projections.png")
    fig0.savefig(out_png, dpi=200)
    plt.close('all')

    # (A) 側面図（Y-Z）
    sim.plot_trajectory_2d(plane='yz')
    plt.close('all')

    # (B) 時系列（Y, Z vs 時間）
    sim.plot_time_series()
    plt.close('all')

    # (C) 3面図（YZ, XY, XZ）
    sim.plot_all_projections()
    plt.close('all')

    # (D) 3D軌道
    sim.plot_trajectory_3d()

    # (E) CSV出力
    out_csv = os.path.join(os.path.dirname(__file__), 'trajectory_output.csv')
    sim.export_to_csv(out_csv)

    return sim, summary


def run_minimal():
    """最小限: デフォルトでシミュレート → 側面図と3Dのみ"""
    sim = BallTrajectorySimulator2(integration_method=IntegrationMethod.RK4)
    sim.simulate(max_time=1.0)
    print(sim.get_summary())
    sim.plot_trajectory_2d(plane='yz')
    sim.plot_trajectory_3d()
    return sim


if __name__ == "__main__":
    run_example(show_plots=True)
