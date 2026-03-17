"""
Rapsodo 計測値 → Nathan Excel 用入力への変換

Rapsodo の計測データ（単位: 速度 km/h, 角度 deg, 回転 rpm, 回転方向は "HH:MM" 可）
を入力とし、Nathan の Excel（TrajectoryCalculator 等）に貼り付け可能な形式
（mph, deg, rpm）に変換する。

参考: https://rapsodo.com/blogs/baseball/understanding-rapsodo-pitching-data-break-profile-introduction
"""

import math
from typing import Tuple, Union

from MyBallTrajectorySim import (
    PitchParameters,
    angular_velocity_xyz_to_backspin_sidespin_wg,
)
from rapsodo_to_mysimulator import clock_time_to_angle_deg

# 内部計算用
RPM_TO_RAD_S = math.pi / 30.0
KMH_TO_MPS = 1.0 / 3.6

# Nathan Excel 出力用（速度 mph）
MPS_TO_MPH = 2.2369362920544
M_TO_FT = 3.28084


def rapsodo_velocity_to_theta_phi(
    v0_mps: float,
    vel_angle_vertical_deg: float,
    vel_azimuth_deg: float,
) -> Tuple[float, float, float]:
    """
    Rapsodo 風の初速度（大きさ＋2角度）→ v0, theta_deg, phi_deg

    Convention:
      vel_angle_vertical_deg: 水平面からの角度 [deg]。正＝下向き（Nathan の theta と同様）
      vel_azimuth_deg: 水平面内の方位 [deg]。0＝捕手方向（Nathan の -Y）、
                       正＝右（+X）方向に測る、など（要 Rapsodo 仕様確認）
                       ※Rapsodo 図の定義では φ の向きが Nathan と反対のため、
                         rapsodo_to_nathan() で Nathan 定義に合わせて符号反転して使用する。

    Returns
    -------
    v0_mps, theta_deg, phi_deg
    """
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
    """
    Rapsodo 風の角速度（大きさ＋後方から見た傾き＋真上から見た方位）→ wx, wy, wz [rad/s]

    Convention（デフォルト）:
      spin_tilt_deg: 回転軸が水平面となす角 [deg]。0＝水平、90＝鉛直（後方から見た傾き）
      spin_azimuth_deg: 真上から見た回転軸の方位 [deg]
        - デフォルト（azimuth_0_toward_catcher=False）:
            spin_azimuth_deg=0 を「バックスピンの回転軸（-X）方向」とみなす
            （theta,phi が小さい球で tilt=0, azimuth=0 のとき backspin_rpm ≒ spin_rate_rpm、
             sidespin_rpm,wg_rpm が小さくなるように合わせる）
        - 互換用（azimuth_0_toward_catcher=True）:
            spin_azimuth_deg=0 を捕手方向（-Y）とみなす（旧挙動）
      回転軸単位ベクトル: 水平成分の長さ cos(tilt), 鉛直成分 sin(tilt)

    Returns
    -------
    wx, wy, wz [rad/s]
    """
    tilt_rad = math.radians(spin_tilt_deg)
    azim_rad = math.radians(spin_azimuth_deg)
    if not tilt_0_is_horizontal:
        tilt_rad = math.radians(90 - spin_tilt_deg)  # 鉛直からの角なら要変換
    ct, st = math.cos(tilt_rad), math.sin(tilt_rad)
    ca, sa = math.cos(azim_rad), math.sin(azim_rad)
    # 水平成分: 方位で方向付け
    if azimuth_0_toward_catcher:
        # 旧挙動: 0＝捕手(-Y) なら (sin(azim), -cos(azim), 0)
        hx, hy = sa, -ca
    else:
        # デフォルト: 0＝バックスピン軸(-X) なら (-cos(azim), -sin(azim), 0)
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
    # Rapsodo 単位: 速度 km/h, 角度 deg, 回転 rpm
    v0_kmh: float,
    vel_angle_vertical_deg: float,
    vel_azimuth_deg: float,
    spin_rate_rpm: float,
    spin_tilt_deg: Union[float, str],
    spin_azimuth_deg: float,
    pitcher_hand: str = "R",
    # オプション（リリース位置・打者利き手・convention）
    x0: float = 0.0,
    y0: float = 16.764,
    z0: float = 1.829,
    batter_hand: str = "R",
    spin_tilt_0_is_horizontal: bool = True,
    spin_azimuth_0_toward_catcher: bool = False,
) -> PitchParameters:
    """
    Rapsodo の 1 球分データを Nathan 用 PitchParameters に変換する。

    入力はすべて Rapsodo の単位で与える: 速度 km/h, 角度 deg, 回転 rpm。
    回転軸の傾き（spin_tilt_deg）のみ、Rapsodo の時刻表記 "HH:MM" でも指定可能。

    Parameters
    ----------
    v0_kmh : float
        初速度 [km/h]（Rapsodo の「最高速度」）
    vel_angle_vertical_deg : float
        縦のリリース角度 [deg]。下向きが正
    vel_azimuth_deg : float
        横のリリース角度 [deg]
    spin_rate_rpm : float
        回転数 [rpm]
    spin_tilt_deg : float or str
        回転軸の傾き。float の場合は [deg]（0＝水平、90＝鉛直）。
        str の場合は Rapsodo の時刻表記 "HH:MM"（例: "01:18"）。
        00:00＝-90 deg, 03:00＝0 deg で変換される。
    spin_azimuth_deg : float
        ジャイロ角度 [deg]（回転軸の方位、真上から見た）
    pitcher_hand : str
        投手の利き腕（'R' / 'L'）。spin_tilt_deg を時計表記で与えた場合に、
        利き腕による左右反転（3:00↔9:00）を吸収して角度へ変換する。
    x0, y0, z0 : float
        リリース位置 [m]
    batter_hand : str
        打者利き手
    spin_tilt_0_is_horizontal : bool
        回転軸傾きの convention（デフォルト True）
    spin_azimuth_0_toward_catcher : bool
        回転軸方位の convention（デフォルト False）

    Returns
    -------
    PitchParameters
        Nathan Excel 用に pitch_parameters_to_nathan_excel_units / format_nathan_excel_line へ渡すか、
        MyBallTrajectorySim の simulate(pitch=...) に渡す。
    """
    # spin_tilt_deg が時刻文字列の場合は角度 [deg] に変換（利き腕による鏡映を考慮）
    if isinstance(spin_tilt_deg, str):
        spin_tilt_deg = clock_time_to_angle_deg(spin_tilt_deg, pitcher_hand=pitcher_hand)

    # Rapsodo 単位で固定: km/h → m/s, 角度・回転はそのまま deg, rpm
    v0_mps = float(v0_kmh) * KMH_TO_MPS
    vel_angle_vertical_deg = float(vel_angle_vertical_deg)
    vel_azimuth_deg = float(vel_azimuth_deg)
    spin_tilt_deg = float(spin_tilt_deg)
    spin_azimuth_deg = float(spin_azimuth_deg)
    spin_rate_rpm = float(spin_rate_rpm)

    _, theta_deg, phi_deg = rapsodo_velocity_to_theta_phi(
        v0_mps, vel_angle_vertical_deg, vel_azimuth_deg
    )
    # Rapsodo 図: φ は Nathan と向きが反対 → Nathan 定義の φ に変換して以降は統一して使用
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
    """
    PitchParameters を Nathan Excel 入力向けの単位（mph, deg, rpm）に変換した辞書を返す。
    Excel 貼り付けやログ用。
    """
    mph = pitch.v0_mps * MPS_TO_MPH
    return {
        "v0_mph": mph,
        "theta_deg": pitch.theta_deg,
        "phi_deg": pitch.phi_deg,
        "backspin_rpm": pitch.backspin_rpm,
        "sidespin_rpm": pitch.sidespin_rpm,
        "wg_rpm": pitch.wg_rpm,
        # 位置: Excel は ft が多いので ft も併記
        "x0_m": pitch.x0,
        "y0_m": pitch.y0,
        "z0_m": pitch.z0,
        "x0_ft": pitch.x0 * M_TO_FT,
        "y0_ft": pitch.y0 * M_TO_FT,
        "z0_ft": pitch.z0 * M_TO_FT,
    }


def format_nathan_excel_line(pitch: PitchParameters, sep: str = "\t") -> str:
    """
    Nathan Excel 用の主要パラメータを1行にした文字列（タブ区切りデフォルト）。
    列順: v0_mph, theta_deg, phi_deg, backspin_rpm, sidespin_rpm, wg_rpm
    """
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


    # spin_tilt_deg が時刻文字列の場合は角度 [deg] に変換
def main():
    """使用例: Rapsodo データを入力し、Nathan Excel 用に変換して表示"""
    # Rapsodo の 1 球分
    # - 単位: 速度 km/h, 角度 deg, 回転 rpm, 回転方向 "HH:MM"
    # - 位置: release side / release height は Rapsodo の m 入力をそのまま使い、Excel 用に ft 換算も出力する
    pitcher_hand = "R"  # "R" or "L"

    # Rapsodo: リリースサイド(m) → x0 に相当
    # 左投手の場合は符号を負にする（ご要望の convention）
    release_side_m = 0.47
    x0_m = -abs(release_side_m) if str(pitcher_hand).upper().startswith("L") else abs(release_side_m)

    # Rapsodo: リリースの高さ(m) → z0 に相当
    release_height_m = 1.5
    z0_m = float(release_height_m)

    # y0 は Excel/Nathan 既定に合わせたまま（必要なら上書き）
    pitch = rapsodo_to_nathan(
        v0_kmh=135.4, # 「１．最高速度」
        vel_angle_vertical_deg=0.1, # 「２．縦のリリース角度（deg）」
        vel_azimuth_deg=-2.6, # 「３．横のリリース角度（deg）」
        spin_rate_rpm=1772, # 「４．回転数」
        spin_tilt_deg="01:18", # 「５．回転方向（時刻形式: "HH:MM"）」
        spin_azimuth_deg=21.0, # 「６．ジャイロ角度（deg）」
        pitcher_hand=pitcher_hand,
        x0=x0_m,
        z0=z0_m,
    )
    print("PitchParameters:", pitch)
    print("backspin_rpm=%.1f, sidespin_rpm=%.1f, wg_rpm=%.1f" % (pitch.backspin_rpm, pitch.sidespin_rpm, pitch.wg_rpm))
    print("\n--- Nathan Excel 用 ---")
    print("辞書（mph, deg, rpm, ft 併記）:", pitch_parameters_to_nathan_excel_units(pitch))
    print("貼り付け用1行（タブ区切り）:", format_nathan_excel_line(pitch))


if __name__ == "__main__":
    main()
