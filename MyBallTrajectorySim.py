"""
野球のボールの投球軌道シミュレータ
シミュレータは以下の物理効果を考慮します：
- 重力
- 空気抵抗（速度依存の抗力係数）
- マグヌス効果（スピンによる揚力）
- 風の影響
- 標高と気温による空気密度の変化
- 投球軌道の計算
- Cdは速度に依存するモデル
"""

"""
野球のボールの投球飛翔シミュレータ（改良版）
BallTrajectorySim.pyの計算原理を元に、より高度な機能を追加

主な改良点：
- ルンゲ・クッタ法による高精度な数値積分
- バッチ処理機能（複数の投球条件を一度にシミュレート）
- パラメータスタディ機能
- より詳細な分析機能
- リアルタイム可視化オプション
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
# 日本語表示用（CJK フォント未検出時の警告を防ぐ）
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Yu Gothic', 'Meiryo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import math
import csv
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class IntegrationMethod(Enum):
    """数値積分手法"""
    EULER = "euler"
    NATHAN = "nathan"  # Excel（TrajectoryCalculator）系の差分更新（速度→位置の順で更新）
    RK4 = "rk4"  # ルンゲ・クッタ4次


@dataclass
class PitchParameters:
    """投球パラメータを格納するデータクラス"""
    x0: float = -0.47 # -0.31  # 初期X位置 (m)
    y0: float = 16.48  # 初期Y位置 (m) = 55 ft
    z0: float = 1.5 # 1.91  # 初期Z位置 (m) = 6 ft
    """この値は、132.7 km/hをmphに変換したもの"""
    v0_mps: float = 37.611111 # 36.8611  Hottaのmocap data # 初速度 (m/s) = 132.7 km/h
    theta_deg: float = 0.1 # 1.47  Hottaのmocap data # リリース角度 (deg)。正=水平より上向き、負=下向き（Excel/Nathanシートと一致）
    phi_deg: float = 2.6 # Hottaのmocap data 2.0  # リリース方向 (deg)
    """この値は、2152.3 rpmに変換したもの"""
    backspin_rpm: float = 1062.74 # 1824 Hottaのmocap data  # バックスピン (rpm)
    sidespin_rpm: float = -1377.89 # -900 # 717.4  Hottaのmocap data # サイドスピン (rpm)
    wg_rpm: float = 451.0  # 回転軸方向のスピン (rpm)
    batter_hand: str = 'R'  # バッターの利き手


def angular_velocity_xyz_to_backspin_sidespin_wg(
    wx_rad_s: float, wy_rad_s: float, wz_rad_s: float,
    theta_deg: float, phi_deg: float,
) -> Tuple[float, float, float]:
    """
    角速度ベクトル (wx, wy, wz) [rad/s] を、シミュレータ用の
    backspin_rpm, sidespin_rpm, wg_rpm に変換する。

    リリース方向・角度 (theta_deg, phi_deg) が必要（3軸の定義がそれに依存するため）。

    Returns
    -------
    backspin_rpm, sidespin_rpm, wg_rpm : float
    """
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    cth, sth = math.cos(th), math.sin(th)
    cph, sph = math.cos(ph), math.sin(ph)
    # 初速度単位ベクトル (v0x/v0, v0y/v0, v0z/v0)。正のθ=上向きなので uz = +sin(θ)
    ux = cth * sph
    uy = -cth * cph
    uz = sth
    # [wx,wy,wz]^T = (pi/30) * M * [B,S,G]^T  →  [B,S,G]^T = (30/pi) * M^{-1} * [wx,wy,wz]^T
    M = np.array([
        [-cph, -sth * sph, ux],
        [sph, -sth * cph, uy],
        [0, cth, uz],
    ])
    rpm_per_rad_s = 30.0 / math.pi
    B, S, G = rpm_per_rad_s * (np.linalg.inv(M) @ np.array([wx_rad_s, wy_rad_s, wz_rad_s]))
    return float(B), float(S), float(G)


@dataclass
class EnvironmentParameters:
    """環境パラメータを格納するデータクラス"""
    temp_F: float = 70.0  # 気温 (deg F)
    elev_m: float = 4.572  # 標高 (m) = 15 ft
    relative_humidity: float = 50.0  # 相対湿度 (%)
    # pressure_inHg: float = 29.92  # 大気圧 (in Hg) Barometric Pressure in inches of  Hg.  Note:  this is the "corrected" value (i.e., referred to sea level)
    pressure_mmHg: float = 760.0  # 大気圧 (mm Hg)
    vwind_mph: float = 0.0  # 風速 (mph)
    phiwind_deg: float = 0.0  # 風向 (deg)
    hwind_m: float = 0.0  # 風の高さ (m)


class BallTrajectorySimulator2:
    """野球のボールの軌道シミュレータクラス（改良版）"""
    
    def __init__(
        self,
        integration_method: IntegrationMethod = IntegrationMethod.RK4,
        use_spin_decay: bool = True,
        excel_compat: bool = False,
    ):
        """
        初期化
        
        Parameters:
        -----------
        integration_method : IntegrationMethod
            数値積分手法（EULER または RK4）
        use_spin_decay : bool
            抗力・揚力のスピン由来項の時間減衰を有効化する（Excel版に近い）。デフォルト True。
        excel_compat : bool
            Nathan Excel（TrajectoryCalculator-new-3D-May2021.xlsx）に合わせた定数・球半径を使用する。
            具体的には cd0 を Excel の既定値に揃える。
        """
        # ボールの物理特性
        self.mass_kg = 0.145  # 質量 (kg) = 5.125 oz
        """original valueは，0.229 mという円周で計算されたもの"""
        self.radius_m = 0.037 #self.circumference_m / (2 * math.pi)  # 半径 (m) original value = 0.03644648196804404
        self.circumference_m = self.radius_m * 2 * math.pi #0.229  # 円周 (m) = 9.125 inches = 0.2324778563656447 m
        
        # 空気抵抗・揚力パラメータ
        """cd0を変更（0.3008から0.297に変更）"""
        self.cd0 = 0.297  # 基本抗力係数

        self.cdspin = 0.0292  # スピンによる抗力係数の増加
        self.cl0 = 0.583  # 基本揚力係数
        self.cl1 = 2.333  # 揚力係数パラメータ1
        self.cl2 = 1.12  # 揚力係数パラメータ2
        
        # 計算パラメータ
        self.dt = 0.001  # 時間刻み (sec)
        self.tau = 10000  # 時間定数 (sec)
        self.beta = 0.0001217  # 標高係数 (1/m)
        self.use_spin_decay = use_spin_decay
        self.excel_compat = excel_compat
        # 参照速度（Excel: 146.7 ft/s = 100 mph）
        self.v_ref_ms = 44.704  # m/s
        
        # 物理定数
        self.g = 9.79  # 重力加速度 (m/s^2)
        self.rho_kg_m3 = 1.197  # 基準空気密度 (kg/m^3) = 0.074742 lb/ft^3
        # 単位変換定数
        self.rpm_to_rad_per_sec = math.pi / 30.0
        self.rad_per_sec_to_rpm = 30.0 / math.pi
        
        # 数値積分手法
        self.integration_method = integration_method

        # Excel互換モード: 定数・半径を Excel 既定に揃える
        if self.excel_compat:
            # Excelは cd0=0.3008 を使用
            self.cd0 = 0.3008
        
        # 計算結果を保存するリスト
        self.trajectory = []
        self.home_plate_crossing = None  # ホームプレート通過時のデータ
        
    def calculate_air_density(self, temp_C: float, elev_m: float, 
                              relative_humidity: float, pressure_mmHg: float) -> float:
        """
        air density in kg/m^3, taking into account temperature, elevation, pressure, and relative humidity.  
        Note that the factor 0.3783 was inserted on Jully 5, 2012 to correctly take into account the mass of
        the water molecule.  See CRC, 54th Ed, p. F-9.
        空気密度を計算（標高、気温、気圧、湿度を考慮）
        
        Parameters:
        -----------
        temp_C : float
            気温 (deg C)
        elev_m : float
            標高 (m)
        relative_humidity : float
            相対湿度 (%)
        pressure_mmHg : float
            気圧 (mm Hg)
        
        Returns:
        --------
        float
            空気密度 (kg/m^3)
        """
        # 飽和水蒸気圧を計算：Buck equation
        svp_mmHg = 4.5841 * math.exp((18.687 - temp_C/234.5) * temp_C / (257.14 + temp_C))
        
        # 空気密度を計算 (kg/m^3)
        rho_kg_m3 = 1.2929 * (273 / (temp_C + 273)) * \
                    (pressure_mmHg * math.exp(-self.beta * elev_m) - 
                     0.3783 * relative_humidity * svp_mmHg / 100) / 760
        
        return rho_kg_m3
    
      
    def _spin_decay_factor(self, v_rel: float, t: float) -> float:
        """
        スピン由来項の時間減衰係数（Excelに準拠）:
          exp( -t / (tau * v_ref / v_rel) )
        """
        if not self.use_spin_decay:
            return 1.0
        if v_rel <= 0:
            return 0.0
        return math.exp(-t / (self.tau * self.v_ref_ms / v_rel))

    def calculate_drag_coefficient(self, v_rel: float, spin_eff: float, t: float) -> float:
        """
        抗力係数Cdを計算（速度とスピンに依存）
        
        Parameters:
        -----------
        v_rel : float
            相対速度 (m/s)
        spin_eff : float
            有効スピン (rpm)
        t : float
            経過時間 (sec)
        
        Returns:
        --------
        float
            抗力係数
        """
        # スピンによる補正項
        spin_term = self.cdspin * spin_eff / 1000
        
        # 時間依存の減衰項
        decay_term = self._spin_decay_factor(v_rel, t)
        
        cd = self.cd0 + spin_term * decay_term
        return cd
    
    def calculate_lift_coefficient(self, romega: float, v_rel: float, t: float) -> float:
        """
        揚力係数を計算
        
        Parameters:
        -----------
        romega : float
            回転速度 (m/s)
        v_rel : float
            相対速度 (m/s)
        t : float
            経過時間 (sec)
        
        Returns:
        --------
        float
            揚力係数
        """
        # S（spin parameter）値を計算
        # 146.7 ft/s = 44.704 m/s (100 mph)
        # v_ref_ms = 44.704  # 参照速度 (m/s)
        S = (romega / v_rel) * self._spin_decay_factor(v_rel, t) if v_rel > 0 else 0
        
        # 揚力係数を計算
        cl = self.cl2 * S / (self.cl0 + self.cl1 * S) if (self.cl0 + self.cl1 * S) > 0 else 0
        return cl
    
    def calculate_const(self, rho: float) -> float:
        """
        定数c0を計算（空気抵抗とマグヌス効果の係数）
        抗力の大きさが F = (1/2)*rho*A*Cd*v^2 になるように、
        ｜加速度｜= const*Cd*v^2 となる const = (1/2)*rho*A/m を使用する。
        従来のExcel由来係数(0.02618...)は約1.8倍大きく終端速度が過小評価されがちなため、
        標準的な抗力式に合わせた。
        
        Parameters:
        -----------
        rho : float
            空気密度 (kg/m^3)
        
        Returns:
        --------
        float
            定数c0 (1/m)。抗力加速度の係数 |a_drag| = const * Cd * v^2
        """
        # 断面積 A = pi * r^2 = circumference^2 / (4*pi)
        A = math.pi * self.radius_m ** 2 #self.circumference_m ** 2 / (4.0 * math.pi)
        # 標準抗力: F = (1/2)*rho*A*Cd*v^2 → a = F/m → const = 0.5*rho*A/m
        const = 0.5 * rho * A / self.mass_kg
        return const
        
    def calculate_wind_velocity(self, z: float, env: EnvironmentParameters) -> Tuple[float, float]:
        """
        風速ベクトルを計算（高さに応じて）
        
        Parameters:
        -----------
        z : float
            高さ (m)
        env : EnvironmentParameters
            環境パラメータ
        
        Returns:
        --------
        Tuple[float, float]
            (vxw, vyw) 風速ベクトル (m/s)
        """
        if z >= env.hwind_m:
            # 1 mph = 0.44704 m/s
            vxw = env.vwind_mph * 0.44704 * math.sin(math.radians(env.phiwind_deg))
            vyw = env.vwind_mph * 0.44704 * math.cos(math.radians(env.phiwind_deg))
        else:
            vxw = 0
            vyw = 0
        return vxw, vyw
    
    def calculate_acceleration(self, state: np.ndarray, t: float, 
                              pitch: PitchParameters, env: EnvironmentParameters,
                              const: float, rho: float, romega_initial: float) -> np.ndarray:
        """
        加速度を計算
        
        Parameters:
        -----------
        state : np.ndarray
            状態ベクトル [x, y, z, vx, vy, vz, wx, wy, wz, spin_total, omega_total]
        t : float
            経過時間 (sec)
        pitch : PitchParameters
            投球パラメータ
        env : EnvironmentParameters
            環境パラメータ
        const : float
            定数c0
        rho : float
            空気密度 (kg/m^3)
        
        Returns:
        --------
        np.ndarray
            加速度ベクトル [ax, ay, az]
        """
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        wx, wy, wz = state[6], state[7], state[8]
        spin_total = state[9] # norm of spin vector(rpm)
        omega_total = state[10] # norm of angular velocity vector (rad/s)
        
        # 現在の速度
        v = math.sqrt(vx**2 + vy**2 + vz**2)
        
        # 風速ベクトル
        vxw, vyw = self.calculate_wind_velocity(z, env)
        
        # 相対速度（風を考慮）
        if z >= env.hwind_m:
            v_rel = math.sqrt((vx - vxw)**2 + (vy - vyw)**2 + vz**2)
        else:
            v_rel = v
        
        # 有効スピン（マグヌス効果に寄与する成分）
        flag = 1  # マグヌス効果を有効にする
        spin_eff = math.sqrt(max(0.0, spin_total**2 - 
                            flag * (self.rad_per_sec_to_rpm * (wx*vx + wy*vy + wz*vz) / v)**2)) if v > 0 else spin_total
        
        # 回転速度（romega）
        romega = (spin_eff * self.rpm_to_rad_per_sec) * self.radius_m
        
        # 抗力係数と揚力係数を計算
        cd = self.calculate_drag_coefficient(v_rel, spin_eff, t)
        cl = self.calculate_lift_coefficient(romega, v_rel, t)
        
        # 空気抵抗（抗力）
        if v_rel > 0:
            drag_x = -const * cd * v_rel * (vx - vxw)
            drag_y = -const * cd * v_rel * (vy - vyw)
            drag_z = -const * cd * v_rel * vz
        else:
            drag_x = drag_y = drag_z = 0
        
        # マグヌス効果（揚力）
        # Excelの式: const*(cl/omega)*v_rel*(cross_product)/X
        # X = M33/romega, ここでM33は現在のromega（有効スピンから計算）、romegaは初期romega
        if v_rel > 0 and omega_total > 0 and romega > 0:
            # ExcelではromegaはRow 14で定義され、時間に依存しない定数
            # X = M33/romega_initial, ここでM33は現在のromega、romega_initialは初期romega
            X = romega / romega_initial if romega_initial > 0 else 1.0
            
            # 風を考慮した相対速度成分
            vx_rel = vx - vxw if z >= env.hwind_m else vx
            vy_rel = vy - vyw if z >= env.hwind_m else vy
            
            # マグヌス効果（クロス積による）
            magnus_x = const * (cl / omega_total) * v_rel * (wy * vz - wz * vy_rel) / X
            magnus_y = const * (cl / omega_total) * v_rel * (wz * vx_rel - wx * vz) / X
            magnus_z = const * (cl / omega_total) * v_rel * (wx * vy_rel - wy * vx_rel) / X
        else:
            magnus_x = magnus_y = magnus_z = 0
        
        # 加速度（重力 + 空気抵抗 + マグヌス効果）
        ax = drag_x + magnus_x
        ay = drag_y + magnus_y
        az = drag_z + magnus_z - self.g
        
        return np.array([ax, ay, az])
    
    def rk4_step(self, state: np.ndarray, t: float, dt: float,
                 pitch: PitchParameters, env: EnvironmentParameters,
                 const: float, rho: float, romega_initial: float) -> np.ndarray:
        """
        ルンゲ・クッタ4次法による1ステップの計算
        
        Parameters:
        -----------
        state : np.ndarray
            現在の状態ベクトル
        t : float
            現在の時間
        dt : float
            時間刻み
        pitch : PitchParameters
            投球パラメータ
        env : EnvironmentParameters
            環境パラメータ
        const : float
            定数c0
        rho : float
            空気密度
        
        Returns:
        --------
        np.ndarray
            次の状態ベクトル
        """
        # k1
        acc1 = self.calculate_acceleration(state, t, pitch, env, const, rho, romega_initial)
        k1 = np.concatenate([state[3:6], acc1, np.zeros(5)])  # [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0]
        
        # k2
        state2 = state + 0.5 * dt * k1
        acc2 = self.calculate_acceleration(state2, t + 0.5*dt, pitch, env, const, rho, romega_initial)
        k2 = np.concatenate([state2[3:6], acc2, np.zeros(5)])
        
        # k3
        state3 = state + 0.5 * dt * k2
        acc3 = self.calculate_acceleration(state3, t + 0.5*dt, pitch, env, const, rho, romega_initial)
        k3 = np.concatenate([state3[3:6], acc3, np.zeros(5)])
        
        # k4
        state4 = state + dt * k3
        acc4 = self.calculate_acceleration(state4, t + dt, pitch, env, const, rho, romega_initial)
        k4 = np.concatenate([state4[3:6], acc4, np.zeros(5)])
        
        # 次の状態
        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return next_state
    
    def euler_step(self, state: np.ndarray, t: float, dt: float,
                   pitch: PitchParameters, env: EnvironmentParameters,
                   const: float, rho: float, romega_initial: float) -> np.ndarray:
        """
        オイラー法による1ステップの計算
        
        Parameters:
        -----------
        state : np.ndarray
            現在の状態ベクトル
        t : float
            現在の時間
        dt : float
            時間刻み
        pitch : PitchParameters
            投球パラメータ
        env : EnvironmentParameters
            環境パラメータ
        const : float
            定数c0
        rho : float
            空気密度
        
        Returns:
        --------
        np.ndarray
            次の状態ベクトル
        """
        acc = self.calculate_acceleration(state, t, pitch, env, const, rho, romega_initial)
        
        # 位置と速度を更新
        next_state = state.copy()
        next_state[0:3] = state[0:3] + state[3:6] * dt + 0.5 * acc * dt**2
        next_state[3:6] = state[3:6] + acc * dt
        
        return next_state

    def nathan_step(self, state: np.ndarray, t: float, dt: float,
                    pitch: PitchParameters, env: EnvironmentParameters,
                    const: float, rho: float, romega_initial: float) -> np.ndarray:
        """
        Nathan Excel（TrajectoryCalculator）系の差分更新に近い 1 ステップ。

        典型的な更新順（シート実装でよくある形）:
          1) 加速度 a_n を計算
          2) 速度 v_{n+1} = v_n + a_n * dt
          3) 位置 x_{n+1} = x_n + v_{n+1} * dt + 0.5 * a_n * dt^2

        ※加速度は当該ステップ開始時の状態（n）で評価する。
        """
        acc = self.calculate_acceleration(state, t, pitch, env, const, rho, romega_initial)

        next_state = state.copy()
        # 先に速度を更新
        next_state[3:6] = state[3:6] + acc * dt
        # 更新後速度を使って位置を更新（Excelの「vx行→x行」更新の並びに合わせる）
        next_state[0:3] = state[0:3] + next_state[3:6] * dt + 0.5 * acc * dt**2
        return next_state
    
    def simulate(self, pitch: Optional[PitchParameters] = None,
                env: Optional[EnvironmentParameters] = None,
                max_time: float = 1.0, save_interval: int = 1) -> List[Dict]:
        """
        軌道をシミュレート
        
        Parameters:
        -----------
        pitch : PitchParameters, optional
            投球パラメータ（Noneの場合はデフォルト値を使用）
        env : EnvironmentParameters, optional
            環境パラメータ（Noneの場合はデフォルト値を使用）
        max_time : float
            最大シミュレーション時間 (sec)
        save_interval : int
            データ保存間隔（1なら全データ、10なら10ステップごと）
        
        Returns:
        --------
        List[Dict]
            軌道データのリスト
        """
        # デフォルトパラメータを使用
        if pitch is None:
            pitch = PitchParameters()
        if env is None:
            env = EnvironmentParameters()
        
        # 初期条件を設定
        v0 = pitch.v0_mps  # 初速度 (m/s)
        
        # 初期速度ベクトル（Excel/Nathan: 正のθ=上向き → v0z = +v0*sin(θ)）
        v0x = pitch.v0_mps * math.cos(math.radians(pitch.theta_deg)) * math.sin(math.radians(pitch.phi_deg))
        v0y = -pitch.v0_mps * math.cos(math.radians(pitch.theta_deg)) * math.cos(math.radians(pitch.phi_deg))
        v0z = pitch.v0_mps * math.sin(math.radians(pitch.theta_deg))
        
        # スピンベクトル（rad/s）
        spin_total = math.sqrt(pitch.backspin_rpm**2 + pitch.sidespin_rpm**2 + pitch.wg_rpm**2) + 0.001
        omega_total = math.sqrt(pitch.backspin_rpm**2 + pitch.sidespin_rpm**2) * self.rpm_to_rad_per_sec + 0.001
        
        # スピン角速度ベクトル
        wx = (-pitch.backspin_rpm * math.cos(math.radians(pitch.phi_deg)) - 
              pitch.sidespin_rpm * math.sin(math.radians(pitch.theta_deg)) * math.sin(math.radians(pitch.phi_deg)) + 
              pitch.wg_rpm * v0x / v0) * self.rpm_to_rad_per_sec
        wy = (pitch.backspin_rpm * math.sin(math.radians(pitch.phi_deg)) - 
              pitch.sidespin_rpm * math.sin(math.radians(pitch.theta_deg)) * math.cos(math.radians(pitch.phi_deg)) + 
              pitch.wg_rpm * v0y / v0) * self.rpm_to_rad_per_sec
        wz = (pitch.sidespin_rpm * math.cos(math.radians(pitch.theta_deg)) + 
              pitch.wg_rpm * v0z / v0) * self.rpm_to_rad_per_sec
        
        # 空気密度と定数を計算
        temp_C = (5/9) * (env.temp_F - 32)
        # pressure_mmHg = env.pressure_inHg * 1000 / 39.37
        rho = self.calculate_air_density(temp_C, env.elev_m, env.relative_humidity, env.pressure_mmHg)
        const = self.calculate_const(rho)
        
        # 初期romegaを計算
        romega_initial = omega_total * self.radius_m
        
        # 初期状態ベクトル [x, y, z, vx, vy, vz, wx, wy, wz, spin_total, omega_total]
        state = np.array([
            pitch.x0, pitch.y0, pitch.z0,  # 位置
            v0x, v0y, v0z,  # 速度
            wx, wy, wz,  # 角速度
            spin_total, omega_total  # スピン関連
        ])
        
        # 軌道データを保存
        self.trajectory = []
        self.home_plate_crossing = None
        
        # ホームプレートまでの距離
        home_plate_y = 0.432  # 17 inches = 0.432 m
        
        # 初期状態を保存（t=0）
        t = 0.0
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        v = math.sqrt(vx**2 + vy**2 + vz**2)
        
        # 初期状態の抗力係数と揚力係数を計算（保存用）
        vxw, vyw = self.calculate_wind_velocity(z, env)
        if z >= env.hwind_m:
            v_rel = math.sqrt((vx - vxw)**2 + (vy - vyw)**2 + vz**2)
        else:
            v_rel = v
        
        flag = 1
        spin_eff = math.sqrt(max(0.0, spin_total**2 - 
                            flag * (self.rad_per_sec_to_rpm * (wx*vx + wy*vy + wz*vz) / v)**2)) if v > 0 else spin_total
        romega = (spin_eff * self.rpm_to_rad_per_sec) * self.radius_m
        cd = self.calculate_drag_coefficient(v_rel, spin_eff, t)
        cl = self.calculate_lift_coefficient(romega, v_rel, t)

        # 初期加速度を計算
        acc0 = self.calculate_acceleration(state, t, pitch, env, const, rho, romega_initial)

        self.trajectory.append({
            't': t,
            'x': x,
            'y': y,
            'z': z,
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'ax': acc0[0],
            'ay': acc0[1],
            'az': acc0[2],
            'v': v,
            'v_mph': v / 0.44704,
            'distance': math.sqrt(x**2 + y**2),
            'height': z,
            'cd': cd,
            'cl': cl
        })

        # シミュレーションループ
        step = 0
        
        while t < max_time:
            # 数値積分
            if self.integration_method == IntegrationMethod.RK4:
                state = self.rk4_step(state, t, self.dt, pitch, env, const, rho, romega_initial)
            elif self.integration_method == IntegrationMethod.NATHAN:
                state = self.nathan_step(state, t, self.dt, pitch, env, const, rho, romega_initial)
            else:
                state = self.euler_step(state, t, self.dt, pitch, env, const, rho, romega_initial)
            
            x, y, z = state[0], state[1], state[2]
            vx, vy, vz = state[3], state[4], state[5]
            wx, wy, wz = state[6], state[7], state[8]
            spin_total, omega_total = state[9], state[10]
            v = math.sqrt(vx**2 + vy**2 + vz**2)
            acc = self.calculate_acceleration(state, t + self.dt, pitch, env, const, rho, romega_initial)

            # ホームプレートを通過したかチェック（データ保存前に）
            if len(self.trajectory) > 0:
                prev_y = self.trajectory[-1]['y']
                if prev_y > home_plate_y and y <= home_plate_y and self.home_plate_crossing is None:
                    # 補間してホームプレート通過時の値を計算
                    prev_point = self.trajectory[-1]
                    fraction = (home_plate_y - prev_y) / (y - prev_y) if (y - prev_y) != 0 else 0
                    t_home = prev_point['t'] + fraction * self.dt * save_interval
                    x_home = prev_point['x'] + fraction * (x - prev_point['x'])
                    z_home = prev_point['z'] + fraction * (z - prev_point['z'])
                    vx_home = prev_point['vx'] + fraction * (vx - prev_point['vx'])
                    vy_home = prev_point['vy'] + fraction * (vy - prev_point['vy'])
                    vz_home = prev_point['vz'] + fraction * (vz - prev_point['vz'])
                    v_home = math.sqrt(vx_home**2 + vy_home**2 + vz_home**2)
                    
                    # ホームプレート通過時の抗力係数と揚力係数を計算
                    vxw, vyw = self.calculate_wind_velocity(z_home, env)
                    if z_home >= env.hwind_m:
                        v_rel_home = math.sqrt((vx_home - vxw)**2 + (vy_home - vyw)**2 + vz_home**2)
                    else:
                        v_rel_home = v_home
                    
                    flag = 1
                    spin_eff_home = math.sqrt(max(0.0, spin_total**2 - 
                                            flag * (self.rad_per_sec_to_rpm * (wx*vx_home + wy*vy_home + wz*vz_home) / v_home)**2)) if v_home > 0 else spin_total
                    romega_home = (spin_eff_home * self.rpm_to_rad_per_sec) * self.radius_m
                    cd_home = self.calculate_drag_coefficient(v_rel_home, spin_eff_home, t_home)
                    cl_home = self.calculate_lift_coefficient(romega_home, v_rel_home, t_home)
                    
                    # ホームプレート通過時の加速度を補間
                    prev_acc = self.trajectory[-1]
                    ax_home = prev_acc.get('ax', 0) + fraction * (acc[0] - prev_acc.get('ax', 0)) if 'ax' in prev_acc else acc[0]
                    ay_home = prev_acc.get('ay', 0) + fraction * (acc[1] - prev_acc.get('ay', 0)) if 'ay' in prev_acc else acc[1]
                    az_home = prev_acc.get('az', 0) + fraction * (acc[2] - prev_acc.get('az', 0)) if 'az' in prev_acc else acc[2]

                    self.home_plate_crossing = {
                        't': t_home,
                        'x': x_home,
                        'y': home_plate_y,
                        'z': z_home,
                        'vx': vx_home,
                        'vy': vy_home,
                        'vz': vz_home,
                        'v': v_home,
                        'v_mph': v_home / 0.44704
                    }

                    # ホームプレート通過時点のデータを追加
                    self.trajectory.append({
                        't': t_home,
                        'x': x_home,
                        'y': home_plate_y,
                        'z': z_home,
                        'vx': vx_home,
                        'vy': vy_home,
                        'vz': vz_home,
                        'ax': ax_home,
                        'ay': ay_home,
                        'az': az_home,
                        'v': v_home,
                        'v_mph': v_home / 0.44704,
                        'distance': math.sqrt(x_home**2 + home_plate_y**2),
                        'height': z_home,
                        'cd': cd_home,
                        'cl': cl_home
                    })
                    
                    # ホームプレート通過を検出したら計算を終了
                    break
            
            # データを保存（指定間隔ごと）
            if step % save_interval == 0:
                # 抗力係数と揚力係数を計算（保存用）
                vxw, vyw = self.calculate_wind_velocity(z, env)
                if z >= env.hwind_m:
                    v_rel = math.sqrt((vx - vxw)**2 + (vy - vyw)**2 + vz**2)
                else:
                    v_rel = v
                
                flag = 1
                spin_eff = math.sqrt(max(0.0, spin_total**2 -
                                    flag * (self.rad_per_sec_to_rpm * (wx*vx + wy*vy + wz*vz) / v)**2)) if v > 0 else spin_total
                romega = (spin_eff * self.rpm_to_rad_per_sec) * self.radius_m
                cd = self.calculate_drag_coefficient(v_rel, spin_eff, t)
                cl = self.calculate_lift_coefficient(romega, v_rel, t)

                acc = self.calculate_acceleration(state, t, pitch, env, const, rho, romega_initial)

                self.trajectory.append({
                    't': t,
                    'x': x,
                    'y': y,
                    'z': z,
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'ax': acc[0],
                    'ay': acc[1],
                    'az': acc[2],
                    'v': v,
                    'v_mph': v / 1.467,
                    'distance': math.sqrt(x**2 + y**2),
                    'height': z,
                    'cd': cd,
                    'cl': cl
                })
            
            # 地面に着地したら終了
            if z <= 0:
                break
            
            t += self.dt
            step += 1
        
        return self.trajectory
    
    def batch_simulate(self, pitch_list: List[PitchParameters],
                      env: Optional[EnvironmentParameters] = None,
                      max_time: float = 1.0) -> List[List[Dict]]:
        """
        複数の投球条件を一度にシミュレート（バッチ処理）
        
        Parameters:
        -----------
        pitch_list : List[PitchParameters]
            投球パラメータのリスト
        env : EnvironmentParameters, optional
            環境パラメータ（全投球で共通）
        max_time : float
            最大シミュレーション時間 (sec)
        
        Returns:
        --------
        List[List[Dict]]
            各投球の軌道データのリスト
        """
        results = []
        for i, pitch in enumerate(pitch_list):
            print(f"シミュレーション {i+1}/{len(pitch_list)}: v0={pitch.v0_mps:.1f} m/s, "
                  f"theta={pitch.theta_deg:.1f} deg, spin={pitch.backspin_rpm:.0f} rpm")
            trajectory = self.simulate(pitch=pitch, env=env, max_time=max_time)
            results.append(trajectory)
        return results
    
    def parameter_study(self, param_name: str, param_values: List[float],
                       base_pitch: Optional[PitchParameters] = None,
                       base_env: Optional[EnvironmentParameters] = None,
                       max_time: float = 1.0) -> Dict:
        """
        パラメータスタディ（1つのパラメータを変化させてシミュレート）
        
        Parameters:
        -----------
        param_name : str
            変化させるパラメータ名（'v0_mps', 'theta_deg', 'backspin_rpm'など）
        param_values : List[float]
            パラメータの値のリスト
        base_pitch : PitchParameters, optional
            ベースとなる投球パラメータ
        base_env : EnvironmentParameters, optional
            ベースとなる環境パラメータ
        max_time : float
            最大シミュレーション時間 (sec)
        
        Returns:
        --------
        Dict
            パラメータ値と結果の辞書
        """
        if base_pitch is None:
            base_pitch = PitchParameters()
        if base_env is None:
            base_env = EnvironmentParameters()
        
        results = {}
        
        for value in param_values:
            # パラメータを設定
            pitch = PitchParameters(
                x0=base_pitch.x0,
                y0=base_pitch.y0,
                z0=base_pitch.z0,
                v0_mps=value if param_name == 'v0_mps' else base_pitch.v0_mps,
                theta_deg=value if param_name == 'theta_deg' else base_pitch.theta_deg,
                phi_deg=value if param_name == 'phi_deg' else base_pitch.phi_deg,
                backspin_rpm=value if param_name == 'backspin_rpm' else base_pitch.backspin_rpm,
                sidespin_rpm=value if param_name == 'sidespin_rpm' else base_pitch.sidespin_rpm,
                wg_rpm=value if param_name == 'wg_rpm' else base_pitch.wg_rpm,
                batter_hand=base_pitch.batter_hand
            )
            
            # シミュレーション実行
            trajectory = self.simulate(pitch=pitch, env=base_env, max_time=max_time)
            
            # 結果を保存
            summary = self.get_summary()
            results[value] = {
                'trajectory': trajectory,
                'summary': summary,
                'home_plate_crossing': self.home_plate_crossing
            }
        
        return results
    
    def plot_trajectory_2d(self, ax=None, show=True, label=None, plane='yz'):
        """
        2D軌道をプロット
        
        Parameters:
        -----------
        ax : matplotlib.axes, optional
            既存のaxesオブジェクト（Noneの場合は新規作成）
        show : bool
            表示するかどうか
        label : str, optional
            凡例ラベル
        plane : str
            表示する平面 ('yz', 'xy', 'xz', 'time_series')
            - 'yz': Y-Z平面（側面図、デフォルト）
            - 'xy': X-Y平面（上面図）
            - 'xz': X-Z平面（正面図）
            - 'time_series': YとZの時系列グラフ
        """
        if not self.trajectory:
            print("軌道データがありません。先にsimulate()を実行してください。")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        x_data = [p['x'] for p in self.trajectory]
        y_data = [p['y'] for p in self.trajectory]
        z_data = [p['z'] for p in self.trajectory]
        t_data = [p['t'] for p in self.trajectory]
        
        if plane.lower() == 'time_series':
            # Y and Z time series graph
            ax.plot(t_data, y_data, linewidth=2, label='Y: Distance (m)', color='b')
            ax.plot(t_data, z_data, linewidth=2, label='Z: Height (m)', color='r')
            ax.set_xlabel('Time (sec)')
            ax.set_ylabel('Distance / Height (m)')
            ax.set_title('Y and Z Time Series')
            ax.grid(True)
            ax.legend()
            # アスペクト比は時系列グラフでは等しくしない
        elif plane.lower() == 'yz':
            # Y-Z plane (side view)
            ax.plot(y_data, z_data, linewidth=2, label=label if label else 'Trajectory')
            ax.axhline(y=0, color='g', linestyle='--', linewidth=1, label='Ground')
            ax.axvline(x=0.432, color='r', linestyle='--', linewidth=1, label='Home Plate')
            ax.set_xlabel('Y: Distance (m)')
            ax.set_ylabel('Z: Height (m)')
            ax.set_title('Ball Trajectory (Y-Z Plane: Side View)')
            ax.grid(True)

            # 初速と y=0 到達時の速度をグラフ中に表示（m/s と km/h を併記）
            first = self.trajectory[0]
            v0_ms = first['v']
            # y=0（ホームプレート）通過時点があればそこを、なければ終点を使用
            if self.home_plate_crossing is not None:
                last = self.home_plate_crossing
            else:
                last = self.trajectory[-1]
            vend_ms = last['v']
            v0_kmh = v0_ms * 3.6
            vend_kmh = vend_ms * 3.6

            text = (
                f"v0 = {v0_ms:.2f} m/s ({v0_kmh:.1f} km/h)\n"
                f"v(y=0) = {vend_ms:.2f} m/s ({vend_kmh:.1f} km/h)"
            )
            # 左上隅付近にテキストボックスとして表示
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
            )

            if label or len(ax.get_legend_handles_labels()[0]) > 0:
                ax.legend()
            ax.set_aspect('equal', adjustable='box')
        elif plane.lower() == 'xy':
            # X-Y plane (top view)
            ax.plot(x_data, y_data, linewidth=2, label=label if label else 'Trajectory')
            ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Center Line')
            ax.axvline(x=0.432, color='r', linestyle='--', linewidth=1, label='Home Plate')
            ax.set_xlabel('X: Lateral (m)')
            ax.set_ylabel('Y: Distance (m)')
            ax.set_title('Ball Trajectory (X-Y Plane: Top View)')
            ax.grid(True)
            if label or len(ax.get_legend_handles_labels()[0]) > 0:
                ax.legend()
            ax.set_aspect('equal', adjustable='box')
        elif plane.lower() == 'xz':
            # X-Z plane (front view)
            ax.plot(x_data, z_data, linewidth=2, label=label if label else 'Trajectory')
            ax.axhline(y=0, color='g', linestyle='--', linewidth=1, label='Ground')
            ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Center Line')
            ax.set_xlabel('X: Lateral (m)')
            ax.set_ylabel('Z: Height (m)')
            ax.set_title('Ball Trajectory (X-Z Plane: Front View)')
            ax.grid(True)
            if label or len(ax.get_legend_handles_labels()[0]) > 0:
                ax.legend()
            ax.set_aspect('equal', adjustable='box')
        else:
            raise ValueError(f"plane must be 'yz', 'xy', 'xz', or 'time_series', got '{plane}'")
        
        if show:
            plt.show()
        
        return ax
    
    def plot_time_series(self, show=True):
        """
        YとZの時系列グラフを表示
        
        Parameters:
        -----------
        show : bool
            表示するかどうか
        """
        return self.plot_trajectory_2d(show=show, plane='time_series')
    
    def plot_all_projections(self, show=True):
        """
        すべての投影図（YZ、XY、XZ平面）を一度に表示
        
        Parameters:
        -----------
        show : bool
            表示するかどうか
        """
        if not self.trajectory:
            print("軌道データがありません。先にsimulate()を実行してください。")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # YZ平面（側面図）
        self.plot_trajectory_2d(ax=axes[0], show=False, plane='yz')
        
        # XY平面（上面図）
        self.plot_trajectory_2d(ax=axes[1], show=False, plane='xy')
        
        # XZ平面（正面図）
        self.plot_trajectory_2d(ax=axes[2], show=False, plane='xz')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig, axes
    
    def plot_trajectory_3d(self, ax=None, show=True):
        """3D軌道をプロット（メートル単位、ホームプレート通過時点まで）"""
        if not self.trajectory:
            print("軌道データがありません。先にsimulate()を実行してください。")
            return
        
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # ホームプレート通過時点までのデータのみを使用
        # home_plate_y = 0.432 m
        home_plate_y = 0.432
        trajectory_to_plot = [p for p in self.trajectory if p['y'] >= home_plate_y]
        
        if not trajectory_to_plot:
            trajectory_to_plot = self.trajectory
        
        # データは既にメートル単位
        x_data = [p['x'] for p in trajectory_to_plot]
        y_data = [p['y'] for p in trajectory_to_plot]
        z_data = [p['z'] for p in trajectory_to_plot]
        
        ax.plot(x_data, y_data, z_data, 'b-', linewidth=2, label='Trajectory')
        ax.scatter([x_data[0]], [y_data[0]], [z_data[0]], color='g', s=100, label='Start Point')
        
        # ホームプレート通過時点を表示
        if self.home_plate_crossing:
            ax.scatter([self.home_plate_crossing['x']], 
                      [self.home_plate_crossing['y']], 
                      [self.home_plate_crossing['z']], 
                      color='orange', s=100, label='Home Plate Crossing')
            # 終了点はホームプレート通過時点
            ax.scatter([self.home_plate_crossing['x']], 
                      [self.home_plate_crossing['y']], 
                      [self.home_plate_crossing['z']], 
                      color='r', s=100, label='End Point (Home Plate)')
        else:
            # ホームプレート通過していない場合は最後の点を表示
            ax.scatter([x_data[-1]], [y_data[-1]], [z_data[-1]], color='r', s=100, label='End Point')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Ball Trajectory (3D)')
        ax.legend()
        
        # XYZ軸のスケールを等しくする
        # 各軸の範囲を取得
        x_range = max(x_data) - min(x_data) if len(x_data) > 0 else 1
        y_range = max(y_data) - min(y_data) if len(y_data) > 0 else 1
        z_range = max(z_data) - min(z_data) if len(z_data) > 0 else 1
        
        # 最大範囲を取得
        max_range = max(x_range, y_range, z_range)
        
        # 各軸の中心を計算
        x_center = (max(x_data) + min(x_data)) / 2 if len(x_data) > 0 else 0
        y_center = (max(y_data) + min(y_data)) / 2 if len(y_data) > 0 else 0
        z_center = (max(z_data) + min(z_data)) / 2 if len(z_data) > 0 else 0
        
        # 各軸の範囲を等しく設定
        ax.set_xlim([x_center - max_range/2, x_center + max_range/2])
        ax.set_ylim([y_center - max_range/2, y_center + max_range/2])
        ax.set_zlim([z_center - max_range/2, z_center + max_range/2])
        
        # アスペクト比を等しく設定（matplotlib 3.3.0以降）
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            # 古いバージョンの場合、手動で設定
            ax.set_aspect('equal')
        
        if show:
            plt.show()
        
        return ax
    
    def get_summary(self):
        """シミュレーション結果のサマリーを返す"""
        if not self.trajectory:
            return None
        
        initial = self.trajectory[0]
        final = self.trajectory[-1]
        
        summary = {
            'initial_velocity_mps': initial['v'],
            'final_velocity_mps': final['v'],
            'initial_position': (initial['x'], initial['y'], initial['z']),
            'final_position': (final['x'], final['y'], final['z']),
            'total_time': final['t'],
            'total_distance': final['distance'],
            'max_height': max([p['z'] for p in self.trajectory]),
            'home_plate_crossing': self.home_plate_crossing
        }
        
        return summary
    
    def export_to_csv(self, filename='trajectory_output.csv'):
        """軌道データをCSVファイルに出力"""
        if not self.trajectory:
            print("軌道データがありません。先にsimulate()を実行してください。")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # ヘッダー（速度は m/s で統一）
            writer.writerow([
                'Time (sec)', 'X (m)', 'Y (m)', 'Z (m)',
                'Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)',
                'Velocity (m/s)',
                'Distance (m)', 'Height (m)',
                'Drag Coefficient', 'Lift Coefficient'
            ])
            
            # データ
            for point in self.trajectory:
                writer.writerow([
                    point['t'],
                    point['x'],
                    point['y'],
                    point['z'],
                    point['vx'],
                    point['vy'],
                    point['vz'],
                    point['v'],
                    point['distance'],
                    point['height'],
                    point.get('cd', 0),
                    point.get('cl', 0)
                ])
        
        print(f"軌道データを {filename} に出力しました。")


def _get_home_plate_xy(
    trajectory: List[Dict],
    home_plate_crossing: Optional[Dict],
    home_plate_y: float = 0.432,
) -> Tuple[float, float]:
    """ホームプレート (y=home_plate_y) 通過時の X(左右), Z(上下) [m] を返す。"""
    if home_plate_crossing is not None:
        return home_plate_crossing['x'], home_plate_crossing['z']
    if not trajectory:
        return float('nan'), float('nan')
    # y が home_plate_y を跨ぐ区間で線形補間
    for i in range(len(trajectory) - 1):
        p0, p1 = trajectory[i], trajectory[i + 1]
        y0, y1 = p0['y'], p1['y']
        if (y0 - home_plate_y) * (y1 - home_plate_y) <= 0 and y0 != y1:
            f = (home_plate_y - y0) / (y1 - y0)
            x = p0['x'] + f * (p1['x'] - p0['x'])
            z = p0['z'] + f * (p1['z'] - p0['z'])
            return x, z
    return trajectory[-1]['x'], trajectory[-1]['z']


def plot_spin_comparison(
    traj_with_spin: List[Dict],
    traj_no_spin: List[Dict],
    home_with: Optional[Dict],
    home_no: Optional[Dict],
    home_plate_y: float = 0.432,
) -> None:
    """角速度あり・なしの2本の軌道を比較して図に表示する（Y-Z, X-Y, X-Z の3面）。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    y_w = [p['y'] for p in traj_with_spin]
    z_w = [p['z'] for p in traj_with_spin]
    x_w = [p['x'] for p in traj_with_spin]
    y_n = [p['y'] for p in traj_no_spin]
    z_n = [p['z'] for p in traj_no_spin]
    x_n = [p['x'] for p in traj_no_spin]

    # 左: Y-Z (側面)
    ax = axes[0]
    ax.plot(y_w, z_w, 'b-', linewidth=2, label='スピンあり')
    ax.plot(y_n, z_n, 'r--', linewidth=1.5, label='角速度=0')
    ax.axhline(y=0, color='g', linestyle=':', linewidth=1)
    ax.axvline(x=home_plate_y, color='k', linestyle='--', linewidth=1, label='ホームプレート')
    ax.set_xlabel('Y: 距離 (m)')
    ax.set_ylabel('Z: 高さ (m)')
    ax.set_title('軌道比較 (Y-Z 側面)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # 中央: X-Y (上から、左右・前後)
    ax = axes[1]
    ax.plot(x_w, y_w, 'b-', linewidth=2, label='スピンあり')
    ax.plot(x_n, y_n, 'r--', linewidth=1.5, label='角速度=0')
    ax.axvline(x=0, color='k', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(y=home_plate_y, color='k', linestyle='--', linewidth=1, label='ホームプレート')
    ax.set_xlabel('X: 左右 (m)')
    ax.set_ylabel('Y: 距離 (m)')
    ax.set_title('軌道比較 (X-Y 上から)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # 右: X-Z (バッター目線・正面)
    ax = axes[2]
    ax.plot(x_w, z_w, 'b-', linewidth=2, label='スピンあり')
    ax.plot(x_n, z_n, 'r--', linewidth=1.5, label='角速度=0')
    ax.axhline(y=0, color='g', linestyle=':', linewidth=1, label='Ground')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Center Line')
    # ホームプレートの位置をX軸上に太線で表示（Z=0、幅約0.43m=17インチ）
    home_plate_half_width = 0.43 / 2.0  # m
    ax.plot([-home_plate_half_width, home_plate_half_width], [0, 0], 'k-', linewidth=8,
            solid_capstyle='butt', label='ホームプレート', zorder=5)
    ax.set_xlabel('X: Lateral (m)')
    ax.set_ylabel('Z: Height (m)')
    ax.set_title('軌道比較 (X-Z バッター目線・正面)')
    # 凡例を少し下げて、軌道線と重ならない位置に配置（下げ過ぎない）
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.40))
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_spin_comparison_3d(
    traj_with_spin: List[Dict],
    traj_no_spin: List[Dict],
    home_with: Optional[Dict],
    home_no: Optional[Dict],
    home_plate_y: float = 0.432,
) -> None:
    """角速度あり・なしの2本の軌道を3Dで比較表示する。"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # ホームプレート通過時点までにトリム
    tw = [p for p in traj_with_spin if p['y'] >= home_plate_y]
    tn = [p for p in traj_no_spin if p['y'] >= home_plate_y]
    if not tw:
        tw = traj_with_spin
    if not tn:
        tn = traj_no_spin

    x_w = [p['x'] for p in tw]
    y_w = [p['y'] for p in tw]
    z_w = [p['z'] for p in tw]
    x_n = [p['x'] for p in tn]
    y_n = [p['y'] for p in tn]
    z_n = [p['z'] for p in tn]

    ax.plot(x_w, y_w, z_w, 'b-', linewidth=2, label='With spin')
    ax.plot(x_n, y_n, z_n, 'r--', linewidth=1.5, label='No spin (omega=0)')

    ax.scatter([x_w[0]], [y_w[0]], [z_w[0]], color='g', s=80, label='Start')
    if home_with:
        ax.scatter([home_with['x']], [home_with['y']], [home_with['z']],
                   color='orange', s=80, marker='o', label='Home (with spin)')
    if home_no:
        ax.scatter([home_no['x']], [home_no['y']], [home_no['z']],
                   color='cyan', s=80, marker='^', label='Home (no spin)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Trajectory comparison 3D (with spin vs no spin)')
    ax.legend()

    # 等スケール
    all_x = x_w + x_n
    all_y = y_w + y_n
    all_z = z_w + z_n
    mx, my, mz = max(all_x), max(all_y), max(all_z)
    nx, ny, nz = min(all_x), min(all_y), min(all_z)
    r = max(mx - nx, my - ny, mz - nz) or 1.0
    cx = (mx + nx) / 2
    cy = (my + ny) / 2
    cz = (mz + nz) / 2
    ax.set_xlim(cx - r / 2, cx + r / 2)
    ax.set_ylim(cy - r / 2, cy + r / 2)
    ax.set_zlim(cz - r / 2, cz + r / 2)
    plt.tight_layout()
    plt.show()


def animate_spin_comparison_3d(
    traj_with_spin: List[Dict],
    traj_no_spin: List[Dict],
    interval_ms: int = 100,
) -> None:
    """3D空間でボール位置をスローアニメーション表示（スピンあり／なしの比較）。"""
    if not traj_with_spin or not traj_no_spin:
        print("軌道データが空です。先に simulate() を実行してください。")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # データ
    xw = np.array([p['x'] for p in traj_with_spin])
    yw = np.array([p['y'] for p in traj_with_spin])
    zw = np.array([p['z'] for p in traj_with_spin])
    xn = np.array([p['x'] for p in traj_no_spin])
    yn = np.array([p['y'] for p in traj_no_spin])
    zn = np.array([p['z'] for p in traj_no_spin])

    # 全体軌道（薄い線）
    ax.plot(xw, yw, zw, 'b-', alpha=0.3)
    ax.plot(xn, yn, zn, 'r--', alpha=0.3)

    # 動くマーカー（3D Line オブジェクト。set_data にはシーケンスが必要）
    point_w, = ax.plot([xw[0]], [yw[0]], [zw[0]], 'bo', label='With spin')
    point_n, = ax.plot([xn[0]], [yn[0]], [zn[0]], 'ro', label='No spin (omega=0)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D animation: with spin vs no spin')
    ax.legend()

    # 等スケール
    all_x = np.concatenate([xw, xn])
    all_y = np.concatenate([yw, yn])
    all_z = np.concatenate([zw, zn])
    mx, my, mz = all_x.max(), all_y.max(), all_z.max()
    nx, ny, nz = all_x.min(), all_y.min(), all_z.min()
    r = max(mx - nx, my - ny, mz - nz) or 1.0
    cx = (mx + nx) / 2
    cy = (my + ny) / 2
    cz = (mz + nz) / 2
    ax.set_xlim(cx - r / 2, cx + r / 2)
    ax.set_ylim(cy - r / 2, cy + r / 2)
    ax.set_zlim(cz - r / 2, cz + r / 2)

    # フレーム数を揃える（短い方に合わせる）
    n_frames = min(len(xw), len(xn))

    def update(frame: int):
        i = frame
        # set_data には 1 要素でもシーケンスを渡す必要がある
        point_w.set_data([xw[i]], [yw[i]])
        point_w.set_3d_properties([zw[i]])
        point_n.set_data([xn[i]], [yn[i]])
        point_n.set_3d_properties([zn[i]])
        return point_w, point_n

    # 3D プロットでは blit=True は環境依存で問題を起こしやすいため False にする
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    # anim.save("spin_comparison_3d_2.mp4", fps=30, writer="ffmpeg")

    plt.show()
    return anim


def run_spin_comparison_2d_only() -> None:
    """
    スピンあり／角速度0 の2本の軌道について、
    2D比較図（Y-Z, X-Y, X-Z）のみを描画する簡易エントリポイント。
    """
    sim = BallTrajectorySimulator2(integration_method=IntegrationMethod.RK4)

    # PitchParameters のデフォルト値を使用
    pitch = PitchParameters()
    env = EnvironmentParameters(
        temp_F=70,
        elev_m=4.572,  # 15 ft
    )

    # スピンあり
    sim.simulate(pitch=pitch, env=env, max_time=1.0)
    traj_with_spin = list(sim.trajectory)
    home_with = sim.home_plate_crossing

    # 角速度=0
    pitch_nospin = PitchParameters(
        x0=pitch.x0, y0=pitch.y0, z0=pitch.z0,
        v0_mps=pitch.v0_mps, theta_deg=pitch.theta_deg, phi_deg=pitch.phi_deg,
        backspin_rpm=0.0, sidespin_rpm=0.0, wg_rpm=0.0,
        batter_hand=pitch.batter_hand,
    )
    sim.simulate(pitch=pitch_nospin, env=env, max_time=1.0)
    traj_no_spin = list(sim.trajectory)
    home_no = sim.home_plate_crossing

    home_plate_y = 0.432
    plot_spin_comparison(traj_with_spin, traj_no_spin, home_with, home_no, home_plate_y)


def main():
    """メイン関数：使用例"""
    # シミュレータを作成（ルンゲ・クッタ法を使用）
    sim = BallTrajectorySimulator2(integration_method=IntegrationMethod.RK4)
    
    # 投球パラメータは PitchParameters のデフォルト（43-60行目）を使用
    pitch = PitchParameters()
    
    # 環境パラメータを設定
    env = EnvironmentParameters(
        temp_F=70,
        elev_m=4.572  # 15 ft
    )
    
    # シミュレーション実行（スピンあり）
    sim.simulate(pitch=pitch, env=env, max_time=1.0)
    traj_with_spin = list(sim.trajectory)
    home_with = sim.home_plate_crossing
    summary = sim.get_summary()
    
    # 角速度=0 の投球で再計算（比較用）
    pitch_nospin = PitchParameters(
        x0=pitch.x0, y0=pitch.y0, z0=pitch.z0,
        v0_mps=pitch.v0_mps, theta_deg=pitch.theta_deg, phi_deg=pitch.phi_deg,
        backspin_rpm=0.0, sidespin_rpm=0.0, wg_rpm=0.0,
        batter_hand=pitch.batter_hand,
    )
    sim.simulate(pitch=pitch_nospin, env=env, max_time=1.0)
    traj_no_spin = list(sim.trajectory)
    home_no = sim.home_plate_crossing
    
    # ホームプレート上での位置（スピンあり・角速度0）と差分
    home_plate_y = 0.432
    x_with, z_with = _get_home_plate_xy(traj_with_spin, home_with, home_plate_y)
    x_no, z_no = _get_home_plate_xy(traj_no_spin, home_no, home_plate_y)
    delta_x = x_with - x_no  # 左右 (m)
    delta_z = z_with - z_no  # 上下 (m)
    
    # 結果を表示（スピンありの結果、m/s と km/h を併記）
    if summary:
        v0_ms = summary['initial_velocity_mps']
        vend_ms = summary['final_velocity_mps']
        v0_kmh = v0_ms * 3.6
        vend_kmh = vend_ms * 3.6
        print("\n=== シミュレーション結果 ===")
        print(f"初速度: {v0_ms:.2f} m/s ({v0_kmh:.1f} km/h)")
        print(f"最終速度: {vend_ms:.2f} m/s ({vend_kmh:.1f} km/h)")
        print(f"最大高度: {summary['max_height']:.2f} m")
        print(f"終端高さ（Z）: {summary['final_position'][2]:.3f} m")
        print(f"総時間: {summary['total_time']:.3f} sec")
        if summary['home_plate_crossing']:
            v_home = summary['home_plate_crossing']['v']
            print(f"ホームプレート通過時速度: {v_home:.2f} m/s ({v_home * 3.6:.1f} km/h)")
    
    # 角速度あり vs 角速度=0 のホームプレート上での差分（XY = 左右・上下 [m]）
    print("\n=== ホームプレート上でのスピンあり vs 角速度=0 の差分 ===")
    print(f"  スピンあり  位置: X(左右) = {x_with:.4f} m, Z(上下) = {z_with:.4f} m")
    print(f"  角速度=0    位置: X(左右) = {x_no:.4f} m, Z(上下) = {z_no:.4f} m")
    print(f"  差分 (スピンあり − 角速度=0): X(左右) = {delta_x:.4f} m, Z(上下) = {delta_z:.4f} m")
    
    # スピン比較図（角速度あり vs 角速度=0）2D
    plot_spin_comparison(traj_with_spin, traj_no_spin, home_with, home_no, home_plate_y)
    # スピン比較 3D
    plot_spin_comparison_3d(traj_with_spin, traj_no_spin, home_with, home_no, home_plate_y)
    # スローアニメーション（3D）— Animation オブジェクトを変数に保持して GC を防ぐ
    # 現在の約5倍速くするため、フレーム間隔を 120ms → 24ms に設定
    anim = animate_spin_comparison_3d(traj_with_spin, traj_no_spin, interval_ms=24)
    
    # 以降はスピンありの軌道で CSV・他プロット（sim.trajectory は直前に角速度=0で上書きされているので復元）
    sim.trajectory = traj_with_spin
    sim.home_plate_crossing = home_with
    
    # CSV出力
    sim.export_to_csv('trajectory_output2.csv')
    
    # プロット
    sim.plot_time_series()
    sim.plot_all_projections()
    sim.plot_trajectory_3d()
    
    return sim


if __name__ == "__main__":
    sim = main()

