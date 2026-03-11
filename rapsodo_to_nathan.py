"""
Rapsodo 計測値 → Nathan 軌道シミュレータ入力への変換

Rapsodo の出力（初速度の大きさ＋2角度、角速度の大きさ＋2角度）を
MyBallTrajectorySim の PitchParameters に変換する。

座標・角度の convention は docstring とオプションで調整可能。
参考: https://rapsodo.com/blogs/baseball/understanding-rapsodo-pitching-data-break-profile-introduction
"""

import math
from typing import Tuple, Optional

from MyBallTrajectorySim import (
    PitchParameters,
    angular_velocity_xyz_to_backspin_sidespin_wg,
)


# --- Convention（要確認時は Rapsodo 仕様に合わせて変更）---
# 速度: vel_angle_vertical_deg 正＝水平より下, vel_azimuth_deg 0＝捕手方向（-Y）
# スピン: spin_tilt_deg 0＝水平, 90＝鉛直. spin_azimuth_deg 0＝捕手方向（-Y）と一致
RPM_TO_RAD_S = math.pi / 30.0

# --- Nathan Excel（TrajectoryCalculator 等）の入力単位 ---
# Excel_parameter_conversion.md: 初速度は mph、角度は deg、回転は rpm
MPS_TO_MPH = 2.2369362920544
MPH_TO_MPS = 1.0 / MPS_TO_MPH
KMH_TO_MPS = 1.0 / 3.6
RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)  # rad/s → rpm


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


def _normalize_velocity_to_mps(v0: float, velocity_input_unit: str) -> float:
    """velocity_input_unit に応じて m/s に正規化"""
    u = (velocity_input_unit or "kmh").lower()
    if u in ("mps", "m/s", "ms"):
        return float(v0)
    if u in ("kmh", "km/h", "kph"):
        return float(v0) * KMH_TO_MPS
    if u in ("mph", ):
        return float(v0) * MPH_TO_MPS
    raise ValueError(
        "velocity_input_unit は 'kmh'（デフォルト）/'mps'/'mph' のいずれかである必要があります: "
        f"{velocity_input_unit!r}"
    )


def _normalize_angles_to_deg(
    vel_angle_vertical: float,
    vel_azimuth: float,
    spin_tilt: float,
    spin_azimuth: float,
    angle_input_unit: str,
) -> Tuple[float, float, float, float]:
    """angle_input_unit に応じて全角度を deg に正規化"""
    u = (angle_input_unit or "deg").lower()
    if u in ("deg", "degree", "degrees"):
        return (
            float(vel_angle_vertical),
            float(vel_azimuth),
            float(spin_tilt),
            float(spin_azimuth),
        )
    if u in ("rad", "radian", "radians"):
        r2d = 180.0 / math.pi
        return (
            float(vel_angle_vertical) * r2d,
            float(vel_azimuth) * r2d,
            float(spin_tilt) * r2d,
            float(spin_azimuth) * r2d,
        )
    raise ValueError(
        f"angle_input_unit は 'deg' または 'rad' である必要があります: {angle_input_unit!r}"
    )


def _normalize_spin_rate_to_rpm(spin_rate: float, spin_rate_input_unit: str) -> float:
    """spin_rate_input_unit に応じて rpm に正規化"""
    u = (spin_rate_input_unit or "rpm").lower()
    if u in ("rpm", ):
        return float(spin_rate)
    if u in ("rad_s", "rad/s", "rads", "radian_per_s"):
        return float(spin_rate) * RAD_S_TO_RPM
    raise ValueError(
        f"spin_rate_input_unit は 'rpm' または 'rad_s' である必要があります: {spin_rate_input_unit!r}"
    )


def rapsodo_to_nathan(
    # 初速度（単位は velocity_input_unit で指定、デフォルト km/h = Rapsodo）
    v0_mps: float,
    vel_angle_vertical_deg: float,
    vel_azimuth_deg: float,
    # 角速度（単位は spin_rate_input_unit で指定、デフォルト rpm）
    spin_rate_rpm: float,
    spin_tilt_deg: float,
    spin_azimuth_deg: float,
    # オプション
    x0: float = 0.0,
    y0: float = 16.764,
    z0: float = 1.829,
    batter_hand: str = "R",
    spin_tilt_0_is_horizontal: bool = True,
    spin_azimuth_0_toward_catcher: bool = False,
    # --- Nathan Excel 入力形式に合わせた単位オプション ---
    # Rapsodo: 速度=km/h, 角度=deg, 回転=rpm
    # Nathan Excel: 速度=mph, 角度=deg, 回転=rpm（出力側で format_nathan_excel_line を使用）
    velocity_input_unit: str = "kmh",
    angle_input_unit: str = "deg",
    spin_rate_input_unit: str = "rpm",
) -> PitchParameters:
    """
    Rapsodo 風の 1 球分データを Nathan の PitchParameters に変換する。

    Parameters
    ----------
    v0_mps : float
        初速度（デフォルトは Rapsodo 出力の km/h。velocity_input_unit で指定）
    vel_angle_vertical_deg : float
        リリースの鉛直角 [deg]。正＝下向き
    vel_azimuth_deg : float
        リリースの水平方位 [deg]。0＝捕手方向など（convention 要確認）
    spin_rate_rpm : float
        回転数 [rpm]
    spin_tilt_deg : float
        回転軸の傾き（後方から見た水平軸との角）[deg]。0＝水平、90＝鉛直
    spin_azimuth_deg : float
        回転軸の方位（真上から見た）[deg]
    x0, y0, z0 : float
        リリース位置 [m]（必要なら上書き）
    batter_hand : str
        打者利き手
    spin_tilt_0_is_horizontal : bool
        True のとき spin_tilt_deg=0 を水平とする
    spin_azimuth_0_toward_catcher : bool
        True のとき spin_azimuth_deg=0 を捕手方向（-Y）とする（旧挙動）
        False（デフォルト）のとき spin_azimuth_deg=0 をバックスピン軸（-X）とする
    velocity_input_unit : str
        'kmh'（デフォルト）… v0 は km/h（Rapsodo）。'mps'… m/s。'mph'… mph（いずれも内部で m/s に換算）
    angle_input_unit : str
        'deg'（デフォルト）… 各角度は度。'rad' … ラジアン（内部で度に換算）
    spin_rate_input_unit : str
        'rpm'（デフォルト）… spin_rate は rpm。'rad_s' … rad/s（内部で rpm に換算）

    Returns
    -------
    PitchParameters
        MyBallTrajectorySim の simulate(pitch=...) に渡す引数
    """
    # 入力単位 → 内部計算用（m/s, deg, rpm）に正規化
    v0_mps = _normalize_velocity_to_mps(v0_mps, velocity_input_unit)
    vel_angle_vertical_deg, vel_azimuth_deg, spin_tilt_deg, spin_azimuth_deg = (
        _normalize_angles_to_deg(
            vel_angle_vertical_deg,
            vel_azimuth_deg,
            spin_tilt_deg,
            spin_azimuth_deg,
            angle_input_unit,
        )
    )
    spin_rate_rpm = _normalize_spin_rate_to_rpm(spin_rate_rpm, spin_rate_input_unit)

    _, theta_deg, phi_deg = rapsodo_velocity_to_theta_phi(
        v0_mps, vel_angle_vertical_deg, vel_azimuth_deg
    )
    wx, wy, wz = rapsodo_spin_to_omega_xyz(
        spin_rate_rpm,
        spin_tilt_deg,
        spin_azimuth_deg,
        tilt_0_is_horizontal=spin_tilt_0_is_horizontal,
        azimuth_0_toward_catcher=spin_azimuth_0_toward_catcher,
    )
    backspin_rpm, sidespin_rpm, wg_rpm = angular_velocity_xyz_to_backspin_sidespin_wg(
        wx, wy, wz, theta_deg, phi_deg
    )
    return PitchParameters(
        x0=x0,
        y0=y0,
        z0=z0,
        v0_mps=v0_mps,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
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
        # 位置は Excel では ft が多い（必要なら別換算）
        "x0_m": pitch.x0,
        "y0_m": pitch.y0,
        "z0_m": pitch.z0,
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


def main():
    """使用例（デフォルト: Rapsodo 速度=km/h 入力）"""
    pitch = rapsodo_to_nathan(
        ##### 速度関係
        v0_mps=132.7,  # 最高速度＝初期速度と同一と解釈（km/h）
        vel_angle_vertical_deg=-1.47, # 縦のリリース角度（deg）　ボールの上下速度方向角度（下向きが正なので注意）
        vel_azimuth_deg=2.0, # 横のリリース角度（deg）　と急の方位角　（例：右ピッチャーがマウンドの右側から，ホームベースの内側の左方向に投げるときは正）

        ##### 回転数関係
        spin_rate_rpm=2152.3, #　回転数
        spin_tilt_deg=-25.0, # 回転方向（deg）　投手から見た回転軸の水平軸に対する角度（ユーザが角度に変換してほしい。ここでは0としています）
        # Rapsodoの回転方向の角度は時間表記なので注意．
        # 変換コードを作っていません．Rapsodoの時間表記がナンセンスで嫌いなだけです．気が向いたら変換できるようにします．
        # 回転軸が水平面となす角 [deg]。0＝水平、90＝鉛直（後方から見た傾き）
        # 通常，右投手のストレートの場合：水平面から下向きになるので，数値は通常は負の数値となる
        spin_azimuth_deg=0.0, # ジャイロ角度（deg）
        # 回転軸の方位（真上から見た）
        # spin_azimuth_deg 0＝捕手方向（-Y）と一致
    )
    print("PitchParameters:", pitch)
    print("backspin_rpm=%.1f, sidespin_rpm=%.1f, wg_rpm=%.1f" % (pitch.backspin_rpm, pitch.sidespin_rpm, pitch.wg_rpm))

    # Nathan Excel 入力単位（mph, deg, rpm）で直接渡す例（速度だけ mph で与える）
    # 89.5 mph ≈ 144 km/h ≈ 40 m/s
    pitch2 = rapsodo_to_nathan(
        v0_mps=82.455957, #89.5,
        vel_angle_vertical_deg=-1.47,
        vel_azimuth_deg=2.0,
        spin_rate_rpm=2152.3,
        spin_tilt_deg=-25.0,
        spin_azimuth_deg=10.0,
        velocity_input_unit="mph",
    )
    print("\n--- mph 入力（v0 を mph で指定）---")
    print("PitchParameters (from mph):", pitch2)
    print("Nathan Excel 用（mph, deg, rpm）:", pitch_parameters_to_nathan_excel_units(pitch2))
    print("貼り付け用1行:", format_nathan_excel_line(pitch2))


if __name__ == "__main__":
    main()
