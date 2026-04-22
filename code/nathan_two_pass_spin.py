"""Nathan 2-pass spin estimation.

Pass 1 (conventional): 9-parameter constant-acceleration fit → ω_T via
(Nathan 2020) eqs (6)(7)(9)(10), ω_G via (4), compose (11).

Pass 2 (refined): Use the Pass 1 ω as input to a full RK4 trajectory
simulation. From the simulated trajectory, extract a(t), v(t), compute
time-averaged ⟨a_M⟩ and ⟨v⟩, and re-invert (7)(8) for ω_T.

Only computes angular velocity (no downstream use). Returns both passes
for side-by-side comparison.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from MyBallTrajectorySim import (  # noqa: E402
    BallTrajectorySimulator2,
    IntegrationMethod,
    PitchParameters,
    EnvironmentParameters,
    angular_velocity_xyz_to_backspin_sidespin_wg,
)


# ── 物理定数 (statcast_to_sim.py と同値) ──
BALL_MASS_KG = 0.145
BALL_RADIUS_M = 0.037
BALL_CROSS_SECTION_M2 = math.pi * BALL_RADIUS_M ** 2
AIR_DENSITY_KG_M3 = 1.2
FPS_TO_MS = 0.3048
RPM_TO_RADS = 2.0 * math.pi / 60.0
RADS_TO_RPM = 60.0 / (2.0 * math.pi)
G_MPS2 = 9.80

# Nathan (2020) C_L fit: C_L = A [1 - exp(-B S)]
CL_A = 0.336
CL_B = 6.041


@dataclass
class SpinEstimate:
    omega_x_rads: float
    omega_y_rads: float
    omega_z_rads: float
    omega_T_mag_rads: float
    omega_G_mag_rads: float
    efficiency: float      # ω_T / ω_total (≤ 1)
    a_M_mag_mps2: float
    v_mean_mps: float
    C_L: float

    def to_dict(self) -> Dict:
        return {
            'omega_x_rads': self.omega_x_rads,
            'omega_y_rads': self.omega_y_rads,
            'omega_z_rads': self.omega_z_rads,
            'omega_total_rpm': math.sqrt(
                self.omega_x_rads**2 + self.omega_y_rads**2 +
                self.omega_z_rads**2) * RADS_TO_RPM,
            'omega_T_rpm': self.omega_T_mag_rads * RADS_TO_RPM,
            'omega_G_rpm': self.omega_G_mag_rads * RADS_TO_RPM,
            'efficiency': self.efficiency,
            'a_M_mag_mps2': self.a_M_mag_mps2,
            'v_mean_mps': self.v_mean_mps,
            'C_L': self.C_L,
        }


def _solve_omega_T_from_CL(C_L: float, v_mean: float) -> float:
    """Eq (8) 反転: ω_T = (⟨v⟩/rB) · ln(A/(A - C_L))"""
    if C_L <= 0 or C_L >= CL_A or v_mean <= 0:
        return 0.0
    return (v_mean / (BALL_RADIUS_M * CL_B)) * math.log(CL_A / (CL_A - C_L))


def _magnus_from_9p(vx0, vy0, vz0, ax, ay, az):
    """9-param constant-a fit から Pass 1 の (⟨v⟩, a_M) を算出 (全て m 単位)."""
    # Flight time from y=50ft to y=17/12 ft (home plate front)
    y_plate_ft = 17.0 / 12.0
    A_ = 0.5 * ay
    B_ = vy0
    C_ = 50.0 - y_plate_ft
    disc = B_**2 - 4 * A_ * C_
    if disc < 0:
        return None
    roots = sorted(r for r in (
        (-B_ - math.sqrt(disc)) / (2 * A_),
        (-B_ + math.sqrt(disc)) / (2 * A_),
    ) if r > 0)
    if not roots:
        return None
    t_flight = roots[0]
    # Midpoint velocity (constant a)
    vx_mean = (vx0 + ax * t_flight / 2) * FPS_TO_MS
    vy_mean = (vy0 + ay * t_flight / 2) * FPS_TO_MS
    vz_mean = (vz0 + az * t_flight / 2) * FPS_TO_MS
    v_mean = math.sqrt(vx_mean**2 + vy_mean**2 + vz_mean**2)
    if v_mean <= 0:
        return None
    v_hat = np.array([vx_mean, vy_mean, vz_mean]) / v_mean
    # Convert acceleration ft/s^2 → m/s^2 and a* = a - g (PITCHf/x: g_z = -g so az + g)
    astar = np.array([ax * FPS_TO_MS, ay * FPS_TO_MS, az * FPS_TO_MS + G_MPS2])
    a_M = astar - (astar @ v_hat) * v_hat
    return v_mean, v_hat, a_M, t_flight


def _compose_omega(omega_T_mag, omega_T_hat, omega_total, v_hat, pitcher_hand):
    """Pass 1/2 共通: ω_T ベクトルと ω_G から ω_total ベクトルを合成。"""
    # ω_T
    omega_T_vec = omega_T_mag * omega_T_hat
    # ω_G magnitude from (4)
    omega_G_mag2 = omega_total**2 - omega_T_mag**2
    omega_G_mag = math.sqrt(omega_G_mag2) if omega_G_mag2 > 0 else 0.0
    # ω_G direction: parallel to <v_hat> for RHP, antiparallel for LHP (heuristic per paper)
    sign = 1.0 if pitcher_hand.upper().startswith('R') else -1.0
    omega_G_vec = sign * omega_G_mag * v_hat
    omega_vec = omega_T_vec + omega_G_vec
    return omega_vec, omega_G_mag


def _pass1(pitch_9p: Dict, spin_rate_rpm: float,
           pitcher_hand: str) -> Optional[SpinEstimate]:
    got = _magnus_from_9p(
        pitch_9p['vx0'], pitch_9p['vy0'], pitch_9p['vz0'],
        pitch_9p['ax'], pitch_9p['ay'], pitch_9p['az'])
    if got is None:
        return None
    v_mean, v_hat, a_M, _ = got
    a_M_mag = float(np.linalg.norm(a_M))
    if a_M_mag < 1e-6:
        return None
    K = 0.5 * AIR_DENSITY_KG_M3 * BALL_CROSS_SECTION_M2 / BALL_MASS_KG
    C_L = a_M_mag / (K * v_mean**2)
    omega_T_mag = _solve_omega_T_from_CL(C_L, v_mean)
    a_M_hat = a_M / a_M_mag
    omega_T_hat = np.cross(v_hat, a_M_hat)
    omega_total = spin_rate_rpm * RPM_TO_RADS
    omega_vec, omega_G_mag = _compose_omega(
        omega_T_mag, omega_T_hat, omega_total, v_hat, pitcher_hand)
    return SpinEstimate(
        omega_x_rads=float(omega_vec[0]),
        omega_y_rads=float(omega_vec[1]),
        omega_z_rads=float(omega_vec[2]),
        omega_T_mag_rads=omega_T_mag,
        omega_G_mag_rads=omega_G_mag,
        efficiency=omega_T_mag / omega_total if omega_total > 0 else 0.0,
        a_M_mag_mps2=a_M_mag,
        v_mean_mps=v_mean,
        C_L=C_L,
    )


def _simulate_trajectory(pitch_9p: Dict, spin_est: SpinEstimate,
                         pitcher_hand: str,
                         env: Optional[EnvironmentParameters] = None):
    """Pass 1 の ω を入力として RK4 軌道を生成し (v(t), a(t)) を返す."""
    # Convert Statcast 9-param release conditions to simulator PitchParameters
    vx0 = pitch_9p['vx0'] * FPS_TO_MS
    vy0 = pitch_9p['vy0'] * FPS_TO_MS
    vz0 = pitch_9p['vz0'] * FPS_TO_MS
    v0 = math.sqrt(vx0**2 + vy0**2 + vz0**2)
    # theta (elevation), phi (azimuth)  — 同 convention as simulator
    theta_deg = math.degrees(math.asin(vz0 / v0))
    phi_deg = math.degrees(math.atan2(vx0, -vy0))
    # x0, y0, z0 at 50 ft. Statcast release extrapolated to y0=50ft by convention.
    x0 = pitch_9p.get('release_pos_x', 0.0) * FPS_TO_MS
    y0 = pitch_9p.get('release_pos_y', 50.0) * FPS_TO_MS
    z0 = pitch_9p.get('release_pos_z', 6.0) * FPS_TO_MS

    # ω_x, ω_y, ω_z [rad/s] → backspin / sidespin / wg [rpm]
    B_rpm, S_rpm, G_rpm = angular_velocity_xyz_to_backspin_sidespin_wg(
        spin_est.omega_x_rads, spin_est.omega_y_rads, spin_est.omega_z_rads,
        theta_deg, phi_deg)

    pp = PitchParameters(
        x0=x0, y0=y0, z0=z0,
        v0_mps=v0, theta_deg=theta_deg, phi_deg=phi_deg,
        backspin_rpm=B_rpm, sidespin_rpm=S_rpm, wg_rpm=G_rpm,
        batter_hand='R',
    )
    if env is None:
        env = EnvironmentParameters()
    sim = BallTrajectorySimulator2(
        integration_method=IntegrationMethod.RK4, use_spin_decay=True)
    traj = sim.simulate(pitch=pp, env=env, max_time=1.5, save_interval=1)
    return traj, sim, pp


def _pass2_from_trajectory(traj, spin_rate_rpm: float,
                           pitcher_hand: str) -> Optional[SpinEstimate]:
    """RK4 軌道から ⟨a_M⟩, ⟨v⟩ を計算し ω_T を再推定."""
    if not traj or len(traj) < 5:
        return None
    # Array of position and time; compute v(t) and a(t) via central differences
    # Simulator already provides vx, vy, vz, ax, ay, az per step.
    # Restrict to pre-home-plate region (y > 17/12 ft in ft? Simulator uses m)
    # Home plate front: y ≈ 0 (simulator uses y=0 at home plate point)
    # Actually simulator's y is distance from home plate FRONT (positive = toward pitcher)
    # Restrict to y >= 0 (ball in flight)
    pts = [p for p in traj if p.get('y', 0) > 0]
    if len(pts) < 5:
        return None
    # Average velocity vector (m/s)
    vx = np.array([p['vx'] for p in pts])
    vy = np.array([p['vy'] for p in pts])
    vz = np.array([p['vz'] for p in pts])
    ax = np.array([p.get('ax', 0) for p in pts])
    ay = np.array([p.get('ay', 0) for p in pts])
    az = np.array([p.get('az', 0) for p in pts])

    # Simulator acceleration already includes gravity; remove it: a* = a - g
    g_vec = np.array([0.0, 0.0, -G_MPS2])
    # Per-step Magnus: a_M(t) = a(t) - g - [ (a(t) - g) · v̂(t) ] v̂(t)
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    mask = v_mag > 1e-6
    if mask.sum() < 5:
        return None
    v_hat = np.stack([vx/v_mag, vy/v_mag, vz/v_mag], axis=1)
    astar = np.stack([ax, ay, az - g_vec[2]], axis=1)
    dot = np.einsum('ij,ij->i', astar, v_hat)
    a_M_t = astar - dot[:, None] * v_hat

    # Time-averaged ⟨a_M⟩, ⟨v⟩ (ベクトル平均)
    a_M_mean = a_M_t[mask].mean(axis=0)
    v_mean_vec = np.array([vx[mask].mean(), vy[mask].mean(), vz[mask].mean()])
    v_mean = float(np.linalg.norm(v_mean_vec))
    v_hat_mean = v_mean_vec / v_mean if v_mean > 0 else np.array([0, -1, 0])
    a_M_mag = float(np.linalg.norm(a_M_mean))
    if a_M_mag < 1e-6:
        return None
    a_M_hat = a_M_mean / a_M_mag

    # (7) → C_L
    K = 0.5 * AIR_DENSITY_KG_M3 * BALL_CROSS_SECTION_M2 / BALL_MASS_KG
    C_L = a_M_mag / (K * v_mean**2)
    # (9) inversion: ω_T
    omega_T_mag = _solve_omega_T_from_CL(C_L, v_mean)
    # (10) direction
    omega_T_hat = np.cross(v_hat_mean, a_M_hat)

    omega_total = spin_rate_rpm * RPM_TO_RADS
    omega_vec, omega_G_mag = _compose_omega(
        omega_T_mag, omega_T_hat, omega_total, v_hat_mean, pitcher_hand)
    return SpinEstimate(
        omega_x_rads=float(omega_vec[0]),
        omega_y_rads=float(omega_vec[1]),
        omega_z_rads=float(omega_vec[2]),
        omega_T_mag_rads=omega_T_mag,
        omega_G_mag_rads=omega_G_mag,
        efficiency=omega_T_mag / omega_total if omega_total > 0 else 0.0,
        a_M_mag_mps2=a_M_mag,
        v_mean_mps=v_mean,
        C_L=C_L,
    )


def estimate_spin_two_pass(pitch_9p: Dict, spin_rate_rpm: float,
                           pitcher_hand: str = 'R',
                           env: Optional[EnvironmentParameters] = None) -> Dict:
    """
    2 パス角速度推定。

    Parameters
    ----------
    pitch_9p : dict
        Statcast の 9 パラメータフィット値 (ft, ft/s, ft/s^2):
        {vx0, vy0, vz0, ax, ay, az, release_pos_x?, release_pos_y?, release_pos_z?}
    spin_rate_rpm : float
        Trackman 総スピン量 (rpm)
    pitcher_hand : 'R' or 'L'
        ω_G 方向選択用
    env : EnvironmentParameters
        環境条件 (省略時デフォルト)

    Returns
    -------
    dict
        {'pass1': SpinEstimate.to_dict(), 'pass2': SpinEstimate.to_dict()}
        失敗時は該当 key が None
    """
    p1 = _pass1(pitch_9p, spin_rate_rpm, pitcher_hand)
    if p1 is None:
        return {'pass1': None, 'pass2': None}
    traj, _, _ = _simulate_trajectory(pitch_9p, p1, pitcher_hand, env)
    p2 = _pass2_from_trajectory(traj, spin_rate_rpm, pitcher_hand)
    return {
        'pass1': p1.to_dict(),
        'pass2': p2.to_dict() if p2 else None,
    }


# ── CLI デモ ──
if __name__ == '__main__':
    # Ohtani 2023 season の平均的な 4-seamer (Statcast 例)
    # 実データを差し替え可
    demo = {
        'vx0': 5.0,    # ft/s (x = catcher's right)
        'vy0': -137.0, # ft/s (y toward pitcher; ball moves in -y)
        'vz0': -4.0,   # ft/s (z up)
        'ax': -6.5,    # ft/s^2
        'ay': 25.0,    # ft/s^2
        'az': -16.5,   # ft/s^2 (includes gravity)
        'release_pos_x': -1.5,  # ft
        'release_pos_y': 54.0,  # ft (≈ extension from 55ft)
        'release_pos_z': 6.0,   # ft
    }
    result = estimate_spin_two_pass(demo, spin_rate_rpm=2400.0, pitcher_hand='R')
    print('=== Pass 1 (conventional Nathan) ===')
    if result['pass1']:
        for k, v in result['pass1'].items():
            print(f'  {k:22s} = {v:+.3f}' if isinstance(v, float)
                  else f'  {k}: {v}')
    print()
    print('=== Pass 2 (RK4 trajectory refined) ===')
    if result['pass2']:
        for k, v in result['pass2'].items():
            print(f'  {k:22s} = {v:+.3f}' if isinstance(v, float)
                  else f'  {k}: {v}')
    print()
    print('=== Δ (Pass 2 - Pass 1) ===')
    if result['pass1'] and result['pass2']:
        p1, p2 = result['pass1'], result['pass2']
        for k in ['omega_x_rads', 'omega_y_rads', 'omega_z_rads',
                 'omega_T_rpm', 'omega_G_rpm', 'C_L', 'a_M_mag_mps2']:
            d = p2[k] - p1[k]
            print(f'  Δ {k:22s} = {d:+.4f}')
