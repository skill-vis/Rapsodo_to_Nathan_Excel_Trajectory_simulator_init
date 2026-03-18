"""
Rapsodo 計測値 → Nathan Excel 用入力への変換（独立パッケージ版）

同ディレクトリの clock_time_to_angle_deg / pitch_parameters_bsg のみに依存。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple, Union

# スクリプト直接実行・親から importlib で読み込む場合に同階層を path に追加
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from clock_time_to_angle_deg import clock_time_to_angle_deg
from pitch_parameters_bsg import PitchParameters, angular_velocity_xyz_to_backspin_sidespin_wg

RPM_TO_RAD_S = math.pi / 30.0
KMH_TO_MPS = 1.0 / 3.6
MPS_TO_MPH = 2.2369362920544
M_TO_FT = 3.28084


def rapsodo_velocity_to_theta_phi(
    v0_mps: float,
    vel_angle_vertical_deg: float,
    vel_azimuth_deg: float,
) -> Tuple[float, float, float]:
    theta_deg = vel_angle_vertical_deg
    phi_deg = vel_azimuth_deg
    return v0_mps, theta_deg, phi_deg


def rapsodo_spin_to_omega_xyz(
    spin_rate_rpm: float,
    spin_tilt_deg: float,
    spin_azimuth_deg: float,
    *,
    tilt_0_is_horizontal: bool = True,
    azimuth_0_toward_catcher: bool = False,
) -> Tuple[float, float, float]:
    tilt_rad = math.radians(spin_tilt_deg)
    azim_rad = math.radians(spin_azimuth_deg)
    if not tilt_0_is_horizontal:
        tilt_rad = math.radians(90 - spin_tilt_deg)
    ct, st = math.cos(tilt_rad), math.sin(tilt_rad)
    ca, sa = math.cos(azim_rad), math.sin(azim_rad)
    if azimuth_0_toward_catcher:
        hx, hy = sa, -ca
    else:
        hx, hy = -ca, -sa
    wx = ct * hx
    wy = ct * hy
    wz = st
    omega_mag_rad_s = spin_rate_rpm * RPM_TO_RAD_S
    n = math.sqrt(wx * wx + wy * wy + wz * wz)
    if n > 1e-12:
        wx *= omega_mag_rad_s / n
        wy *= omega_mag_rad_s / n
        wz *= omega_mag_rad_s / n
    else:
        wx = wy = wz = 0.0
    return wx, wy, wz


def rapsodo_to_nathan(
    v0_kmh: float,
    vel_angle_vertical_deg: float,
    vel_azimuth_deg: float,
    spin_rate_rpm: float,
    spin_tilt_deg: Union[float, str],
    spin_azimuth_deg: float,
    pitcher_hand: str = "R",
    x0: float = 0.0,
    y0: float = 16.764,
    z0: float = 1.829,
    batter_hand: str = "R",
    spin_tilt_0_is_horizontal: bool = True,
    spin_azimuth_0_toward_catcher: bool = False,
) -> PitchParameters:
    if isinstance(spin_tilt_deg, str):
        spin_tilt_deg = clock_time_to_angle_deg(spin_tilt_deg, pitcher_hand=pitcher_hand)

    v0_mps = float(v0_kmh) * KMH_TO_MPS
    vel_angle_vertical_deg = float(vel_angle_vertical_deg)
    vel_azimuth_deg = float(vel_azimuth_deg)
    spin_tilt_deg = float(spin_tilt_deg)
    spin_azimuth_deg = float(spin_azimuth_deg)
    spin_rate_rpm = float(spin_rate_rpm)

    _, theta_deg, phi_deg = rapsodo_velocity_to_theta_phi(
        v0_mps, vel_angle_vertical_deg, vel_azimuth_deg
    )
    phi_deg_nathan = -phi_deg
    wx, wy, wz = rapsodo_spin_to_omega_xyz(
        spin_rate_rpm,
        spin_tilt_deg,
        spin_azimuth_deg,
        tilt_0_is_horizontal=spin_tilt_0_is_horizontal,
        azimuth_0_toward_catcher=spin_azimuth_0_toward_catcher,
    )
    backspin_rpm, sidespin_rpm, wg_rpm = angular_velocity_xyz_to_backspin_sidespin_wg(
        wx, wy, wz, theta_deg, phi_deg_nathan
    )
    return PitchParameters(
        x0=x0,
        y0=y0,
        z0=z0,
        v0_mps=v0_mps,
        theta_deg=theta_deg,
        phi_deg=phi_deg_nathan,
        backspin_rpm=backspin_rpm,
        sidespin_rpm=sidespin_rpm,
        wg_rpm=wg_rpm,
        batter_hand=batter_hand,
    )


def pitch_parameters_to_nathan_excel_units(pitch: PitchParameters) -> dict:
    mph = pitch.v0_mps * MPS_TO_MPH
    return {
        "release_speed_mph": mph,
        "release_angle_deg": pitch.theta_deg,
        "release_direction_deg": pitch.phi_deg,
        "backspin_rpm": pitch.backspin_rpm,
        "sidespin_rpm": pitch.sidespin_rpm,
        "wg_rpm": pitch.wg_rpm,
        "x0_m": pitch.x0,
        "y0_m": pitch.y0,
        "z0_m": pitch.z0,
        "x0_ft": pitch.x0 * M_TO_FT,
        "y0_ft": pitch.y0 * M_TO_FT,
        "z0_ft": pitch.z0 * M_TO_FT,
    }


def format_nathan_excel_line(pitch: PitchParameters, sep: str = "\t") -> str:
    d = pitch_parameters_to_nathan_excel_units(pitch)
    return sep.join(
        [
            f"{d['v0_mph']:.4f}",
            f"{d['theta_deg']:.4f}",
            f"{d['phi_deg']:.4f}",
            f"{d['backspin_rpm']:.2f}",
            f"{d['sidespin_rpm']:.2f}",
            f"{d['wg_rpm']:.2f}",
        ]
    )


def main():
    pitcher_hand = "R"
    release_side_m = 0.47
    x0_m = -abs(release_side_m) if str(pitcher_hand).upper().startswith("L") else abs(release_side_m)
    z0_m = 1.5
    pitch = rapsodo_to_nathan(
        v0_kmh=135.4,
        vel_angle_vertical_deg=0.1,
        vel_azimuth_deg=-2.6,
        spin_rate_rpm=1772,
        spin_tilt_deg="01:18",
        spin_azimuth_deg=21.0,
        pitcher_hand=pitcher_hand,
        x0=x0_m,
        z0=z0_m,
    )
    print("PitchParameters:", pitch)
    print("backspin_rpm=%.1f, sidespin_rpm=%.1f, wg_rpm=%.1f" % (pitch.backspin_rpm, pitch.sidespin_rpm, pitch.wg_rpm))
    print("\n--- Nathan Excel 用 ---")
    print("辞書:", pitch_parameters_to_nathan_excel_units(pitch))
    print("貼り付け用1行:", format_nathan_excel_line(pitch))


if __name__ == "__main__":
    main()
