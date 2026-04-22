"""Synthetic controlled test: known ω_true → RK4 軌道 → 9-param フィット → Pass1/Pass2 逆推定.

ground truth が既知のため、Pass 1 と Pass 2 の真の推定精度を比較できる。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)

from MyBallTrajectorySim import (
    BallTrajectorySimulator2, IntegrationMethod,
    PitchParameters, EnvironmentParameters,
    angular_velocity_xyz_to_backspin_sidespin_wg,
)
from nathan_two_pass_spin import (
    estimate_spin_two_pass, FPS_TO_MS, RPM_TO_RADS, RADS_TO_RPM,
)

MS_TO_FPS = 1.0 / FPS_TO_MS


def _bsg_from_xyz(wx, wy, wz, theta_deg, phi_deg):
    """angular_velocity_xyz_to_backspin_sidespin_wg の薄いラッパ (単位 rad/s → rpm)"""
    return angular_velocity_xyz_to_backspin_sidespin_wg(wx, wy, wz, theta_deg, phi_deg)


def _xyz_from_bsg(B_rpm, S_rpm, G_rpm, theta_deg, phi_deg):
    """backspin/sidespin/wg から world-frame ω (rad/s) を計算 (逆変換)"""
    th = math.radians(theta_deg); ph = math.radians(phi_deg)
    cth, sth = math.cos(th), math.sin(th)
    cph, sph = math.cos(ph), math.sin(ph)
    ux = cth * sph
    uy = -cth * cph
    uz = sth
    M = np.array([
        [-cph, -sth * sph, ux],
        [sph, -sth * cph, uy],
        [0, cth, uz],
    ])
    rpm_per_rad_s = 30.0 / math.pi
    rad_per_rpm = 1.0 / rpm_per_rad_s
    w = M @ np.array([B_rpm, S_rpm, G_rpm]) * rad_per_rpm
    return float(w[0]), float(w[1]), float(w[2])


def _fit_9param_from_trajectory(traj):
    """RK4 軌道から定加速度 (最小二乗) フィット。出力 ft 単位 (Statcast 互換)."""
    pts = [p for p in traj if p.get('y', 0) > 0.5]  # 1.4 ft 以上 (home plate 前まで)
    if len(pts) < 10:
        return None
    t = np.array([p['t'] for p in pts])
    # simulator 座標系: x = catcher's right? 要確認 → PitchParameters と同じなので statcast 互換
    x = np.array([p['x'] for p in pts])
    y = np.array([p['y'] for p in pts])
    z = np.array([p['z'] for p in pts])
    # t=0 から 0 秒後のパラメータとする
    def fit(pts_arr):
        A = np.stack([np.ones_like(t), t, 0.5 * t**2], axis=1)
        coef, *_ = np.linalg.lstsq(A, pts_arr, rcond=None)
        return tuple(float(c) for c in coef)
    x0, vx0, ax = fit(x)
    y0, vy0, ay = fit(y)
    z0, vz0, az = fit(z)
    return {
        'vx0': vx0 * MS_TO_FPS, 'vy0': vy0 * MS_TO_FPS, 'vz0': vz0 * MS_TO_FPS,
        'ax':  ax  * MS_TO_FPS, 'ay':  ay  * MS_TO_FPS, 'az':  az  * MS_TO_FPS,
        'release_pos_x': x0 * MS_TO_FPS,
        'release_pos_y': y0 * MS_TO_FPS,
        'release_pos_z': z0 * MS_TO_FPS,
    }


def _angle_deg(v1, v2):
    a, b = np.array(v1), np.array(v2)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return None
    c = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def synthesize_and_test(label, v0_mps, theta_deg, phi_deg,
                        B_rpm, S_rpm, G_rpm, batter_hand='R',
                        pitcher_hand='R',
                        x0_m=-0.47, y0_m=16.48, z0_m=1.8):
    """1 投球合成 → フィット → Pass1/Pass2 → 真値と比較."""
    # 1. True ω (world frame)
    wx, wy, wz = _xyz_from_bsg(B_rpm, S_rpm, G_rpm, theta_deg, phi_deg)
    omega_true = (wx, wy, wz)
    omega_total_rads = math.sqrt(wx**2 + wy**2 + wz**2)
    omega_total_rpm = omega_total_rads * RADS_TO_RPM

    # 2. RK4 で軌道生成
    pp = PitchParameters(
        x0=x0_m, y0=y0_m, z0=z0_m, v0_mps=v0_mps,
        theta_deg=theta_deg, phi_deg=phi_deg,
        backspin_rpm=B_rpm, sidespin_rpm=S_rpm, wg_rpm=G_rpm,
        batter_hand=batter_hand,
    )
    sim = BallTrajectorySimulator2(integration_method=IntegrationMethod.RK4)
    traj = sim.simulate(pitch=pp, env=EnvironmentParameters(),
                        max_time=1.0, save_interval=1)

    # 3. 9-param フィット
    fit = _fit_9param_from_trajectory(traj)
    if fit is None:
        return {'label': label, 'error': 'fit failed'}

    # 4. Pass 1/Pass 2
    try:
        result = estimate_spin_two_pass(
            fit, spin_rate_rpm=omega_total_rpm,
            pitcher_hand=pitcher_hand)
    except Exception as e:
        result = {'pass1': None, 'pass2': None, 'err': str(e)}

    return {
        'label': label,
        'omega_true': omega_true,
        'omega_total_rpm': omega_total_rpm,
        'pass1': result.get('pass1'),
        'pass2': result.get('pass2'),
        'err': result.get('err'),
    }


def _fmt(o, lab):
    wx, wy, wz = o
    rpm = math.sqrt(wx**2 + wy**2 + wz**2) * RADS_TO_RPM
    return f'{lab}: ({wx:+7.2f}, {wy:+7.2f}, {wz:+7.2f}) rad/s  |ω|={rpm:.0f} rpm'


def main():
    # 多様な球種の合成ケース
    cases = [
        # RHP の典型 4-seam (強 backspin, 小 gyro)
        ('RHP 4-seam (2400rpm, efficient 92%)',
         {'v0_mps': 40.5, 'theta_deg': -1.0, 'phi_deg': 2.0,
          'B_rpm': 2200, 'S_rpm': -500, 'G_rpm': 700,
          'pitcher_hand': 'R'}),
        # RHP pure backspin (ideal)
        ('RHP pure backspin (2400rpm, 100%)',
         {'v0_mps': 40.5, 'theta_deg': -1.0, 'phi_deg': 0.0,
          'B_rpm': 2400, 'S_rpm': 0, 'G_rpm': 0,
          'pitcher_hand': 'R'}),
        # RHP cutter/slider (gyro 多め)
        ('RHP slider (2500rpm, gyro 70%)',
         {'v0_mps': 36.0, 'theta_deg': -2.0, 'phi_deg': 0.5,
          'B_rpm': 800, 'S_rpm': -1000, 'G_rpm': 2000,
          'pitcher_hand': 'R'}),
        # RHP curve (topspin)
        ('RHP curveball (2600rpm)',
         {'v0_mps': 33.5, 'theta_deg': 1.5, 'phi_deg': -2.0,
          'B_rpm': -1800, 'S_rpm': -900, 'G_rpm': 1600,
          'pitcher_hand': 'R'}),
        # LHP 4-seam
        ('LHP 4-seam (2300rpm)',
         {'v0_mps': 39.0, 'theta_deg': -1.0, 'phi_deg': -2.0,
          'B_rpm': 2100, 'S_rpm': 600, 'G_rpm': 700,
          'pitcher_hand': 'L'}),
        # Low spin changeup
        ('RHP changeup (1600rpm)',
         {'v0_mps': 34.0, 'theta_deg': -0.5, 'phi_deg': 3.0,
          'B_rpm': 1200, 'S_rpm': -700, 'G_rpm': 800,
          'pitcher_hand': 'R'}),
    ]

    for label, params in cases:
        res = synthesize_and_test(label, **params)
        print(f'=== {label}  |ω_true|={res.get("omega_total_rpm", 0):.0f} rpm ===')
        if 'error' in res:
            print(f'  SKIP: {res["error"]}'); print(); continue
        true = res['omega_true']
        print('  ' + _fmt(true, 'TRUE   '))
        if res['pass1']:
            p1 = res['pass1']
            o1 = (p1['omega_x_rads'], p1['omega_y_rads'], p1['omega_z_rads'])
            print('  ' + _fmt(o1, 'Pass 1 '))
            ang = _angle_deg(true, o1)
            diff = np.array(o1) - np.array(true)
            d_rpm = np.linalg.norm(diff) * RADS_TO_RPM
            print(f'    Δangle={ang:.2f}°  |Δω|={d_rpm:.1f} rpm')
        if res['pass2']:
            p2 = res['pass2']
            o2 = (p2['omega_x_rads'], p2['omega_y_rads'], p2['omega_z_rads'])
            print('  ' + _fmt(o2, 'Pass 2 '))
            ang = _angle_deg(true, o2)
            diff = np.array(o2) - np.array(true)
            d_rpm = np.linalg.norm(diff) * RADS_TO_RPM
            print(f'    Δangle={ang:.2f}°  |Δω|={d_rpm:.1f} rpm')
        elif res.get('err'):
            print(f'  Pass 2: FAILED ({res["err"]})')
        print()


if __name__ == '__main__':
    main()
