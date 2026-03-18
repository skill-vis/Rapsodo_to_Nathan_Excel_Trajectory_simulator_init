"""
MyBallTrajectorySim から rapsodo_to_nathan 用に必要な部分のみ抽出:
PitchParameters と angular_velocity_xyz_to_backspin_sidespin_wg
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PitchParameters:
    """投球パラメータ（Nathan / Excel 入力用）"""
    x0: float = 0.0
    y0: float = 16.764
    z0: float = 1.829
    v0_mps: float = 37.611111
    theta_deg: float = 0.1
    phi_deg: float = 2.6
    backspin_rpm: float = 0.0
    sidespin_rpm: float = 0.0
    wg_rpm: float = 0.0
    batter_hand: str = "R"


def angular_velocity_xyz_to_backspin_sidespin_wg(
    wx_rad_s: float,
    wy_rad_s: float,
    wz_rad_s: float,
    theta_deg: float,
    phi_deg: float,
) -> Tuple[float, float, float]:
    """
    角速度 (wx, wy, wz) [rad/s] → backspin_rpm, sidespin_rpm, wg_rpm [rpm]
    """
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    cth, sth = math.cos(th), math.sin(th)
    cph, sph = math.cos(ph), math.sin(ph)
    ux = cth * sph
    uy = -cth * cph
    uz = sth
    M = np.array(
        [
            [-cph, -sth * sph, ux],
            [sph, -sth * cph, uy],
            [0, cth, uz],
        ]
    )
    rpm_per_rad_s = 30.0 / math.pi
    B, S, G = rpm_per_rad_s * (np.linalg.inv(M) @ np.array([wx_rad_s, wy_rad_s, wz_rad_s]))
    return float(B), float(S), float(G)
