"""
Baseball pitch trajectory simulator.

The underlying physics and formulation originate in A. Nathan's Excel Trajectory
Calculator; (https://baseball.physics.illinois.edu/trajectory-calculator-new3D.html)
this code is a Python port with the unit system converted to MKS
(meters, kilograms, seconds). Additional integrators (e.g. RK4) and extensions
beyond the spreadsheet are included.

Physics included:
- Gravity
- Air drag (velocity-dependent drag coefficient)
- Magnus force (spin-induced lift)
- Wind
- Air density vs elevation and temperature
- Trajectory integration
- Cd model depends on speed
"""

"""
Enhanced baseball pitch flight simulator.
Extends BallTrajectorySim-style physics with additional features:

- High-accuracy numerical integration (Runge–Kutta)
- Batch simulation (many pitch conditions at once)
- Parameter sweeps
- Richer analysis and plotting
- Optional real-time visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
plt.rcParams['axes.unicode_minus'] = False
import math
import csv
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class IntegrationMethod(Enum):
    """Numerical integration scheme."""
    EULER = "euler"
    NATHAN = "nathan"  # Excel TrajectoryCalculator-like update (velocity then position)
    RK4 = "rk4"  # 4th-order Runge–Kutta


@dataclass
class PitchParameters:
    """Pitch / release parameters."""
    x0: float = -0.47  # initial X (m)
    y0: float = 16.48  # initial Y (m) ≈ 55 ft
    z0: float = 1.5  # initial Z (m) ≈ 6 ft
    v0_mps: float = 37.611111  # release speed (m/s); e.g. 132.7 km/h; Hotta mocap refs in comments
    theta_deg: float = 0.1  # vertical release angle (deg); + = up from horizontal (Nathan/Excel)
    phi_deg: float = 2.6  # horizontal release direction (deg)
    backspin_rpm: float = 1062.74
    sidespin_rpm: float = -1377.89
    wg_rpm: float = 451.0  # spin about velocity axis (gyro), rpm
    batter_hand: str = 'R'


def angular_velocity_xyz_to_backspin_sidespin_wg(
    wx_rad_s: float, wy_rad_s: float, wz_rad_s: float,
    theta_deg: float, phi_deg: float,
) -> Tuple[float, float, float]:
    """
    Convert angular velocity (wx, wy, wz) [rad/s] to backspin_rpm, sidespin_rpm, wg_rpm.

    theta_deg and phi_deg define the release direction (axis convention).

    Returns
    -------
    backspin_rpm, sidespin_rpm, wg_rpm : float
    """
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    cth, sth = math.cos(th), math.sin(th)
    cph, sph = math.cos(ph), math.sin(ph)
    # Release-direction unit vector; +theta => uz = +sin(theta)
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
    """Environment (air, wind)."""
    temp_F: float = 70.0  # temperature (deg F)
    elev_m: float = 4.572  # elevation (m) ≈ 15 ft
    relative_humidity: float = 50.0  # relative humidity (%)
    pressure_mmHg: float = 760.0  # pressure (mm Hg)
    vwind_mph: float = 0.0  # wind speed (mph)
    phiwind_deg: float = 0.0  # wind direction (deg)
    hwind_m: float = 0.0  # height above which wind applies (m)


class BallTrajectorySimulator2:
    """Baseball trajectory simulator (enhanced)."""

    def __init__(
        self,
        integration_method: IntegrationMethod = IntegrationMethod.RK4,
        use_spin_decay: bool = True,
        excel_compat: bool = False,
    ):
        """
        Parameters
        ----------
        integration_method : IntegrationMethod
            EULER, RK4, or NATHAN (Excel-like).
        use_spin_decay : bool
            Time decay on spin-dependent drag/lift (Excel-like). Default True.
        excel_compat : bool
            Match Nathan Excel TrajectoryCalculator defaults (e.g. cd0 = 0.3008).
        """
        self.mass_kg = 0.145  # kg ≈ 5.125 oz
        self.radius_m = 0.037  # m (circumference-based value ~0.03645 m also used in literature)
        self.circumference_m = self.radius_m * 2 * math.pi  # m ≈ 9.125 in

        self.cd0 = 0.297  # base drag coefficient
        self.cdspin = 0.0292  # spin-dependent drag increment
        self.cl0 = 0.583
        self.cl1 = 2.333
        self.cl2 = 1.12

        self.dt = 0.001  # time step (s)
        self.tau = 10000  # time scale (s)
        self.beta = 0.0001217  # elevation factor (1/m)
        self.use_spin_decay = use_spin_decay
        self.excel_compat = excel_compat
        self.v_ref_ms = 44.704  # m/s (Excel: 146.7 ft/s ≈ 100 mph)

        self.g = 9.79  # gravity (m/s^2)
        self.rho_kg_m3 = 1.197  # reference air density (kg/m^3)
        self.rpm_to_rad_per_sec = math.pi / 30.0
        self.rad_per_sec_to_rpm = 30.0 / math.pi

        self.integration_method = integration_method

        if self.excel_compat:
            self.cd0 = 0.3008

        self.trajectory = []
        self.home_plate_crossing = None  # state at home-plate crossing
        
    def calculate_air_density(self, temp_C: float, elev_m: float, 
                              relative_humidity: float, pressure_mmHg: float) -> float:
        """
        Air density (kg/m^3) from temperature, elevation, pressure, and relative humidity.
        Factor 0.3783 accounts for water vapor mass (CRC, 54th Ed., p. F-9).

        Parameters
        ----------
        temp_C : float
            Temperature (deg C)
        elev_m : float
            Elevation (m)
        relative_humidity : float
            Relative humidity (%)
        pressure_mmHg : float
            Pressure (mm Hg)

        Returns
        -------
        float
            Air density (kg/m^3)
        """
        # Saturation vapor pressure (Buck equation)
        svp_mmHg = 4.5841 * math.exp((18.687 - temp_C/234.5) * temp_C / (257.14 + temp_C))
        rho_kg_m3 = 1.2929 * (273 / (temp_C + 273)) * \
                    (pressure_mmHg * math.exp(-self.beta * elev_m) - 
                     0.3783 * relative_humidity * svp_mmHg / 100) / 760
        
        return rho_kg_m3
    
      
    def _spin_decay_factor(self, v_rel: float, t: float) -> float:
        """
        Spin-term time decay factor (Excel-style):
          exp( -t / (tau * v_ref / v_rel) )
        """
        if not self.use_spin_decay:
            return 1.0
        if v_rel <= 0:
            return 0.0
        return math.exp(-t / (self.tau * self.v_ref_ms / v_rel))

    def calculate_drag_coefficient(self, v_rel: float, spin_eff: float, t: float) -> float:
        """
        Drag coefficient Cd vs speed and spin.

        Parameters
        ----------
        v_rel : float
            Relative speed (m/s)
        spin_eff : float
            Effective spin (rpm)
        t : float
            Time (s)

        Returns
        -------
        float
            Drag coefficient
        """
        spin_term = self.cdspin * spin_eff / 1000
        decay_term = self._spin_decay_factor(v_rel, t)
        
        cd = self.cd0 + spin_term * decay_term
        return cd
    
    def calculate_lift_coefficient(self, romega: float, v_rel: float, t: float) -> float:
        """
        Lift coefficient Cl.

        Parameters
        ----------
        romega : float
            r * omega tangential speed (m/s)
        v_rel : float
            Relative speed (m/s)
        t : float
            Time (s)

        Returns
        -------
        float
            Lift coefficient
        """
        S = (romega / v_rel) * self._spin_decay_factor(v_rel, t) if v_rel > 0 else 0
        cl = self.cl2 * S / (self.cl0 + self.cl1 * S) if (self.cl0 + self.cl1 * S) > 0 else 0
        return cl
    
    def calculate_const(self, rho: float) -> float:
        """
        Constant c0 for drag/Magnus scaling: |a_drag| = const * Cd * v^2 with
        F = (1/2)*rho*A*Cd*v^2 so const = (1/2)*rho*A/m.
        Uses standard drag form (legacy Excel-only factor ~0.02618 can underestimate terminal speed).

        Parameters
        ----------
        rho : float
            Air density (kg/m^3)

        Returns
        -------
        float
            c0 (1/m)
        """
        A = math.pi * self.radius_m ** 2
        const = 0.5 * rho * A / self.mass_kg
        return const
        
    def calculate_wind_velocity(self, z: float, env: EnvironmentParameters) -> Tuple[float, float]:
        """
        Wind velocity (m/s) vs height.

        Parameters
        ----------
        z : float
            Height (m)
        env : EnvironmentParameters
            Environment

        Returns
        -------
        Tuple[float, float]
            (vxw, vyw) wind components (m/s)
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
        Acceleration [ax, ay, az].

        Parameters
        ----------
        state : np.ndarray
            [x, y, z, vx, vy, vz, wx, wy, wz, spin_total, omega_total]
        t : float
            Time (s)
        pitch, env : parameters
        const : float
            Drag scale c0
        rho : float
            Air density (kg/m^3)

        Returns
        -------
        np.ndarray
            [ax, ay, az]
        """
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        wx, wy, wz = state[6], state[7], state[8]
        spin_total = state[9] # norm of spin vector(rpm)
        omega_total = state[10] # norm of angular velocity vector (rad/s)
        
        v = math.sqrt(vx**2 + vy**2 + vz**2)
        vxw, vyw = self.calculate_wind_velocity(z, env)
        if z >= env.hwind_m:
            v_rel = math.sqrt((vx - vxw)**2 + (vy - vyw)**2 + vz**2)
        else:
            v_rel = v
        
        flag = 1  # Magnus on
        spin_eff = math.sqrt(spin_total**2 - 
                            flag * (self.rad_per_sec_to_rpm * (wx*vx + wy*vy + wz*vz) / v)**2) if v > 0 else spin_total
        
        romega = (spin_eff * self.rpm_to_rad_per_sec) * self.radius_m
        cd = self.calculate_drag_coefficient(v_rel, spin_eff, t)
        cl = self.calculate_lift_coefficient(romega, v_rel, t)
        
        if v_rel > 0:
            drag_x = -const * cd * v_rel * (vx - vxw)
            drag_y = -const * cd * v_rel * (vy - vyw)
            drag_z = -const * cd * v_rel * vz
        else:
            drag_x = drag_y = drag_z = 0
        
        # Magnus: Excel-style const*(cl/omega)*v_rel*cross/X; X = romega/romega_initial
        if v_rel > 0 and omega_total > 0 and romega > 0:
            X = romega / romega_initial if romega_initial > 0 else 1.0
            vx_rel = vx - vxw if z >= env.hwind_m else vx
            vy_rel = vy - vyw if z >= env.hwind_m else vy
            magnus_x = const * (cl / omega_total) * v_rel * (wy * vz - wz * vy_rel) / X
            magnus_y = const * (cl / omega_total) * v_rel * (wz * vx_rel - wx * vz) / X
            magnus_z = const * (cl / omega_total) * v_rel * (wx * vy_rel - wy * vx_rel) / X
        else:
            magnus_x = magnus_y = magnus_z = 0
        
        ax = drag_x + magnus_x
        ay = drag_y + magnus_y
        az = drag_z + magnus_z - self.g
        
        return np.array([ax, ay, az])
    
    def rk4_step(self, state: np.ndarray, t: float, dt: float,
                 pitch: PitchParameters, env: EnvironmentParameters,
                 const: float, rho: float, romega_initial: float) -> np.ndarray:
        """
        One RK4 step.

        Parameters
        ----------
        state : np.ndarray
            Current state
        t : float
            Current time
        dt : float
            Time step
        pitch, env, const, rho, romega_initial : as in calculate_acceleration

        Returns
        -------
        np.ndarray
            Next state
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
        
        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return next_state
    
    def euler_step(self, state: np.ndarray, t: float, dt: float,
                   pitch: PitchParameters, env: EnvironmentParameters,
                   const: float, rho: float, romega_initial: float) -> np.ndarray:
        """One explicit Euler step."""
        acc = self.calculate_acceleration(state, t, pitch, env, const, rho, romega_initial)
        next_state = state.copy()
        next_state[0:3] = state[0:3] + state[3:6] * dt + 0.5 * acc * dt**2
        next_state[3:6] = state[3:6] + acc * dt
        
        return next_state

    def nathan_step(self, state: np.ndarray, t: float, dt: float,
                    pitch: PitchParameters, env: EnvironmentParameters,
                    const: float, rho: float, romega_initial: float) -> np.ndarray:
        """
        One step matching Nathan Excel TrajectoryCalculator-style ordering:
          1) a_n from state n
          2) v_{n+1} = v_n + a_n * dt
          3) x_{n+1} = x_n + v_{n+1}*dt + 0.5*a_n*dt^2
        """
        acc = self.calculate_acceleration(state, t, pitch, env, const, rho, romega_initial)

        next_state = state.copy()
        next_state[3:6] = state[3:6] + acc * dt
        next_state[0:3] = state[0:3] + next_state[3:6] * dt + 0.5 * acc * dt**2
        return next_state
    
    def simulate(self, pitch: Optional[PitchParameters] = None,
                env: Optional[EnvironmentParameters] = None,
                max_time: float = 1.0, save_interval: int = 1) -> List[Dict]:
        """
        Run trajectory integration.

        Parameters
        ----------
        pitch, env : optional; defaults if None
        max_time : float
            Max time (s)
        save_interval : int
            Store every N steps (1 = all)

        Returns
        -------
        List[Dict]
            Trajectory samples
        """
        if pitch is None:
            pitch = PitchParameters()
        if env is None:
            env = EnvironmentParameters()
        
        v0 = pitch.v0_mps
        v0x = pitch.v0_mps * math.cos(math.radians(pitch.theta_deg)) * math.sin(math.radians(pitch.phi_deg))
        v0y = -pitch.v0_mps * math.cos(math.radians(pitch.theta_deg)) * math.cos(math.radians(pitch.phi_deg))
        v0z = pitch.v0_mps * math.sin(math.radians(pitch.theta_deg))
        
        spin_total = math.sqrt(pitch.backspin_rpm**2 + pitch.sidespin_rpm**2 + pitch.wg_rpm**2) + 0.001
        omega_total = math.sqrt(pitch.backspin_rpm**2 + pitch.sidespin_rpm**2) * self.rpm_to_rad_per_sec + 0.001
        wx = (-pitch.backspin_rpm * math.cos(math.radians(pitch.phi_deg)) - 
              pitch.sidespin_rpm * math.sin(math.radians(pitch.theta_deg)) * math.sin(math.radians(pitch.phi_deg)) + 
              pitch.wg_rpm * v0x / v0) * self.rpm_to_rad_per_sec
        wy = (pitch.backspin_rpm * math.sin(math.radians(pitch.phi_deg)) - 
              pitch.sidespin_rpm * math.sin(math.radians(pitch.theta_deg)) * math.cos(math.radians(pitch.phi_deg)) + 
              pitch.wg_rpm * v0y / v0) * self.rpm_to_rad_per_sec
        wz = (pitch.sidespin_rpm * math.cos(math.radians(pitch.theta_deg)) + 
              pitch.wg_rpm * v0z / v0) * self.rpm_to_rad_per_sec
        
        temp_C = (5/9) * (env.temp_F - 32)
        # pressure_mmHg = env.pressure_inHg * 1000 / 39.37
        rho = self.calculate_air_density(temp_C, env.elev_m, env.relative_humidity, env.pressure_mmHg)
        const = self.calculate_const(rho)
        
        romega_initial = omega_total * self.radius_m
        state = np.array([
            pitch.x0, pitch.y0, pitch.z0,
            v0x, v0y, v0z,
            wx, wy, wz,
            spin_total, omega_total,
        ])
        self.trajectory = []
        self.home_plate_crossing = None
        home_plate_y = 0.432  # 17 in
        t = 0.0
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        v = math.sqrt(vx**2 + vy**2 + vz**2)
        
        # Initial Cd, Cl for first stored point
        vxw, vyw = self.calculate_wind_velocity(z, env)
        if z >= env.hwind_m:
            v_rel = math.sqrt((vx - vxw)**2 + (vy - vyw)**2 + vz**2)
        else:
            v_rel = v
        
        flag = 1
        spin_eff = math.sqrt(spin_total**2 - 
                            flag * (self.rad_per_sec_to_rpm * (wx*vx + wy*vy + wz*vz) / v)**2) if v > 0 else spin_total
        romega = (spin_eff * self.rpm_to_rad_per_sec) * self.radius_m
        cd = self.calculate_drag_coefficient(v_rel, spin_eff, t)
        cl = self.calculate_lift_coefficient(romega, v_rel, t)
        
        self.trajectory.append({
            't': t,
            'x': x,
            'y': y,
            'z': z,
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'v': v,
            'v_mph': v / 0.44704,
            'distance': math.sqrt(x**2 + y**2),
            'height': z,
            'cd': cd,
            'cl': cl
        })
        
        step = 0
        while t < max_time:
            if self.integration_method == IntegrationMethod.RK4:
                state = self.rk4_step(state, t, self.dt, pitch, env, const, rho, romega_initial)
            elif self.integration_method == IntegrationMethod.NATHAN:
                state = self.nathan_step(state, t, self.dt, pitch, env, const, rho, romega_initial)
            else:
                state = self.euler_step(state, t, self.dt, pitch, env, const, rho, romega_initial)
            
            x, y, z = state[0], state[1], state[2]
            vx, vy, vz = state[3], state[4], state[5]
            v = math.sqrt(vx**2 + vy**2 + vz**2)
            
            if len(self.trajectory) > 0:
                prev_y = self.trajectory[-1]['y']
                if prev_y > home_plate_y and y <= home_plate_y and self.home_plate_crossing is None:
                    prev_point = self.trajectory[-1]
                    fraction = (home_plate_y - prev_y) / (y - prev_y) if (y - prev_y) != 0 else 0
                    t_home = prev_point['t'] + fraction * self.dt * save_interval
                    x_home = prev_point['x'] + fraction * (x - prev_point['x'])
                    z_home = prev_point['z'] + fraction * (z - prev_point['z'])
                    vx_home = prev_point['vx'] + fraction * (vx - prev_point['vx'])
                    vy_home = prev_point['vy'] + fraction * (vy - prev_point['vy'])
                    vz_home = prev_point['vz'] + fraction * (vz - prev_point['vz'])
                    v_home = math.sqrt(vx_home**2 + vy_home**2 + vz_home**2)
                    
                    vxw, vyw = self.calculate_wind_velocity(z_home, env)
                    if z_home >= env.hwind_m:
                        v_rel_home = math.sqrt((vx_home - vxw)**2 + (vy_home - vyw)**2 + vz_home**2)
                    else:
                        v_rel_home = v_home
                    
                    flag = 1
                    spin_eff_home = math.sqrt(spin_total**2 - 
                                            flag * (self.rad_per_sec_to_rpm * (wx*vx_home + wy*vy_home + wz*vz_home) / v_home)**2) if v_home > 0 else spin_total
                    romega_home = (spin_eff_home * self.rpm_to_rad_per_sec) * self.radius_m
                    cd_home = self.calculate_drag_coefficient(v_rel_home, spin_eff_home, t_home)
                    cl_home = self.calculate_lift_coefficient(romega_home, v_rel_home, t_home)
                    
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
                    
                    self.trajectory.append({
                        't': t_home,
                        'x': x_home,
                        'y': home_plate_y,
                        'z': z_home,
                        'vx': vx_home,
                        'vy': vy_home,
                        'vz': vz_home,
                        'v': v_home,
                        'v_mph': v_home / 0.44704,
                        'distance': math.sqrt(x_home**2 + home_plate_y**2),
                        'height': z_home,
                        'cd': cd_home,
                        'cl': cl_home
                    })
                    
                    break

            if step % save_interval == 0:
                vxw, vyw = self.calculate_wind_velocity(z, env)
                if z >= env.hwind_m:
                    v_rel = math.sqrt((vx - vxw)**2 + (vy - vyw)**2 + vz**2)
                else:
                    v_rel = v
                
                flag = 1
                spin_eff = math.sqrt(spin_total**2 - 
                                    flag * (self.rad_per_sec_to_rpm * (wx*vx + wy*vy + wz*vz) / v)**2) if v > 0 else spin_total
                romega = (spin_eff * self.rpm_to_rad_per_sec) * self.radius_m
                cd = self.calculate_drag_coefficient(v_rel, spin_eff, t)
                cl = self.calculate_lift_coefficient(romega, v_rel, t)
                
                self.trajectory.append({
                    't': t,
                    'x': x,
                    'y': y,
                    'z': z,
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'v': v,
                    'v_mph': v / 1.467,
                    'distance': math.sqrt(x**2 + y**2),
                    'height': z,
                    'cd': cd,
                    'cl': cl
                })
            
            if z <= 0:
                break
            
            t += self.dt
            step += 1
        
        return self.trajectory
    
    def batch_simulate(self, pitch_list: List[PitchParameters],
                      env: Optional[EnvironmentParameters] = None,
                      max_time: float = 1.0) -> List[List[Dict]]:
        """
        Batch-run many pitches.

        Returns
        -------
        List[List[Dict]]
            One trajectory list per pitch
        """
        results = []
        for i, pitch in enumerate(pitch_list):
            print(f"Simulation {i+1}/{len(pitch_list)}: v0={pitch.v0_mps:.1f} m/s, "
                  f"theta={pitch.theta_deg:.1f} deg, spin={pitch.backspin_rpm:.0f} rpm")
            trajectory = self.simulate(pitch=pitch, env=env, max_time=max_time)
            results.append(trajectory)
        return results
    
    def parameter_study(self, param_name: str, param_values: List[float],
                       base_pitch: Optional[PitchParameters] = None,
                       base_env: Optional[EnvironmentParameters] = None,
                       max_time: float = 1.0) -> Dict:
        """
        Sweep one parameter over a list of values.

        Parameters
        ----------
        param_name : str
            e.g. 'v0_mps', 'theta_deg', 'backspin_rpm'
        param_values : List[float]
        base_pitch, base_env : optional baselines
        max_time : float

        Returns
        -------
        Dict
            value -> {trajectory, summary, home_plate_crossing}
        """
        if base_pitch is None:
            base_pitch = PitchParameters()
        if base_env is None:
            base_env = EnvironmentParameters()
        
        results = {}
        
        for value in param_values:
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
            
            trajectory = self.simulate(pitch=pitch, env=base_env, max_time=max_time)
            summary = self.get_summary()
            results[value] = {
                'trajectory': trajectory,
                'summary': summary,
                'home_plate_crossing': self.home_plate_crossing
            }
        
        return results
    
    def plot_trajectory_2d(self, ax=None, show=True, label=None, plane='yz'):
        """
        Plot 2D trajectory.

        plane : 'yz' | 'xy' | 'xz' | 'time_series'
        """
        if not self.trajectory:
            print("No trajectory data; run simulate() first.")
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
        elif plane.lower() == 'yz':
            # Y-Z plane (side view)
            ax.plot(y_data, z_data, linewidth=2, label=label if label else 'Trajectory')
            ax.axhline(y=0, color='g', linestyle='--', linewidth=1, label='Ground')
            ax.axvline(x=0.432, color='r', linestyle='--', linewidth=1, label='Home Plate')
            ax.set_xlabel('Y: Distance (m)')
            ax.set_ylabel('Z: Height (m)')
            ax.set_title('Ball Trajectory (Y-Z Plane: Side View)')
            ax.grid(True)

            first = self.trajectory[0]
            v0_ms = first['v']
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
        """Y and Z vs time."""
        return self.plot_trajectory_2d(show=show, plane='time_series')
    
    def plot_all_projections(self, show=True):
        """YZ, XY, XZ projections in one figure."""
        if not self.trajectory:
            print("No trajectory data; run simulate() first.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        self.plot_trajectory_2d(ax=axes[0], show=False, plane='yz')
        self.plot_trajectory_2d(ax=axes[1], show=False, plane='xy')
        self.plot_trajectory_2d(ax=axes[2], show=False, plane='xz')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig, axes
    
    def plot_trajectory_3d(self, ax=None, show=True):
        """3D trajectory in meters, trimmed to home-plate crossing if available."""
        if not self.trajectory:
            print("No trajectory data; run simulate() first.")
            return
        
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        home_plate_y = 0.432
        trajectory_to_plot = [p for p in self.trajectory if p['y'] >= home_plate_y]
        
        if not trajectory_to_plot:
            trajectory_to_plot = self.trajectory
        
        x_data = [p['x'] for p in trajectory_to_plot]
        y_data = [p['y'] for p in trajectory_to_plot]
        z_data = [p['z'] for p in trajectory_to_plot]
        
        ax.plot(x_data, y_data, z_data, 'b-', linewidth=2, label='Trajectory')
        ax.scatter([x_data[0]], [y_data[0]], [z_data[0]], color='g', s=100, label='Start Point')
        
        if self.home_plate_crossing:
            ax.scatter([self.home_plate_crossing['x']], 
                      [self.home_plate_crossing['y']], 
                      [self.home_plate_crossing['z']], 
                      color='orange', s=100, label='Home Plate Crossing')
            ax.scatter([self.home_plate_crossing['x']], 
                      [self.home_plate_crossing['y']], 
                      [self.home_plate_crossing['z']], 
                      color='r', s=100, label='End Point (Home Plate)')
        else:
            ax.scatter([x_data[-1]], [y_data[-1]], [z_data[-1]], color='r', s=100, label='End Point')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Ball Trajectory (3D)')
        ax.legend()
        
        x_range = max(x_data) - min(x_data) if len(x_data) > 0 else 1
        y_range = max(y_data) - min(y_data) if len(y_data) > 0 else 1
        z_range = max(z_data) - min(z_data) if len(z_data) > 0 else 1
        
        max_range = max(x_range, y_range, z_range)
        x_center = (max(x_data) + min(x_data)) / 2 if len(x_data) > 0 else 0
        y_center = (max(y_data) + min(y_data)) / 2 if len(y_data) > 0 else 0
        z_center = (max(z_data) + min(z_data)) / 2 if len(z_data) > 0 else 0
        
        ax.set_xlim([x_center - max_range/2, x_center + max_range/2])
        ax.set_ylim([y_center - max_range/2, y_center + max_range/2])
        ax.set_zlim([z_center - max_range/2, z_center + max_range/2])
        
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            ax.set_aspect('equal')
        
        if show:
            plt.show()
        
        return ax
    
    def get_summary(self):
        """Return summary dict for last simulate() run."""
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
        """Export trajectory to CSV."""
        if not self.trajectory:
            print("No trajectory data; run simulate() first.")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                'Time (sec)', 'X (m)', 'Y (m)', 'Z (m)',
                'Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)',
                'Velocity (m/s)',
                'Distance (m)', 'Height (m)',
                'Drag Coefficient', 'Lift Coefficient'
            ])
            
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
        
        print(f"Wrote trajectory to {filename}.")


def _get_home_plate_xy(
    trajectory: List[Dict],
    home_plate_crossing: Optional[Dict],
    home_plate_y: float = 0.432,
) -> Tuple[float, float]:
    """X and Z (m) at home plate (y = home_plate_y)."""
    if home_plate_crossing is not None:
        return home_plate_crossing['x'], home_plate_crossing['z']
    if not trajectory:
        return float('nan'), float('nan')
    # Linear interpolation where y crosses home_plate_y
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
    """Compare with-spin vs no-spin trajectories (YZ, XY, XZ)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    y_w = [p['y'] for p in traj_with_spin]
    z_w = [p['z'] for p in traj_with_spin]
    x_w = [p['x'] for p in traj_with_spin]
    y_n = [p['y'] for p in traj_no_spin]
    z_n = [p['z'] for p in traj_no_spin]
    x_n = [p['x'] for p in traj_no_spin]

    ax = axes[0]
    ax.plot(y_w, z_w, 'b-', linewidth=2, label='With spin')
    ax.plot(y_n, z_n, 'r--', linewidth=1.5, label='No spin (omega=0)')
    ax.axhline(y=0, color='g', linestyle=':', linewidth=1)
    ax.axvline(x=home_plate_y, color='k', linestyle='--', linewidth=1, label='Home plate')
    ax.set_xlabel('Y: distance (m)')
    ax.set_ylabel('Z: height (m)')
    ax.set_title('Trajectory (Y-Z side)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    ax = axes[1]
    ax.plot(x_w, y_w, 'b-', linewidth=2, label='With spin')
    ax.plot(x_n, y_n, 'r--', linewidth=1.5, label='No spin (omega=0)')
    ax.axvline(x=0, color='k', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(y=home_plate_y, color='k', linestyle='--', linewidth=1, label='Home plate')
    ax.set_xlabel('X: lateral (m)')
    ax.set_ylabel('Y: distance (m)')
    ax.set_title('Trajectory (X-Y top)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    ax = axes[2]
    ax.plot(x_w, z_w, 'b-', linewidth=2, label='With spin')
    ax.plot(x_n, z_n, 'r--', linewidth=1.5, label='No spin (omega=0)')
    ax.axhline(y=0, color='g', linestyle=':', linewidth=1, label='Ground')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Center line')
    home_plate_half_width = 0.43 / 2.0
    ax.plot([-home_plate_half_width, home_plate_half_width], [0, 0], 'k-', linewidth=8,
            solid_capstyle='butt', label='Home plate', zorder=5)
    ax.set_xlabel('X: lateral (m)')
    ax.set_ylabel('Z: height (m)')
    ax.set_title('Trajectory (X-Z batter view)')
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
    """3D comparison: with spin vs no spin."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

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
    """Slow 3D animation: with spin vs no spin."""
    if not traj_with_spin or not traj_no_spin:
        print("Trajectory empty; run simulate() first.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xw = np.array([p['x'] for p in traj_with_spin])
    yw = np.array([p['y'] for p in traj_with_spin])
    zw = np.array([p['z'] for p in traj_with_spin])
    xn = np.array([p['x'] for p in traj_no_spin])
    yn = np.array([p['y'] for p in traj_no_spin])
    zn = np.array([p['z'] for p in traj_no_spin])

    ax.plot(xw, yw, zw, 'b-', alpha=0.3)
    ax.plot(xn, yn, zn, 'r--', alpha=0.3)

    point_w, = ax.plot([xw[0]], [yw[0]], [zw[0]], 'bo', label='With spin')
    point_n, = ax.plot([xn[0]], [yn[0]], [zn[0]], 'ro', label='No spin (omega=0)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D animation: with spin vs no spin')
    ax.legend()

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

    n_frames = min(len(xw), len(xn))

    def update(frame: int):
        i = frame
        point_w.set_data([xw[i]], [yw[i]])
        point_w.set_3d_properties([zw[i]])
        point_n.set_data([xn[i]], [yn[i]])
        point_n.set_3d_properties([zn[i]])
        return point_w, point_n

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
    """Quick entry: 2D spin comparison only."""
    sim = BallTrajectorySimulator2(integration_method=IntegrationMethod.RK4)
    pitch = PitchParameters()
    env = EnvironmentParameters(
        temp_F=70,
        elev_m=4.572,  # 15 ft
    )

    sim.simulate(pitch=pitch, env=env, max_time=1.0)
    traj_with_spin = list(sim.trajectory)
    home_with = sim.home_plate_crossing

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
    """Example main."""
    sim = BallTrajectorySimulator2(integration_method=IntegrationMethod.RK4)
    pitch = PitchParameters()
    env = EnvironmentParameters(
        temp_F=70,
        elev_m=4.572  # 15 ft
    )
    
    sim.simulate(pitch=pitch, env=env, max_time=1.0)
    traj_with_spin = list(sim.trajectory)
    home_with = sim.home_plate_crossing
    summary = sim.get_summary()
    
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
    x_with, z_with = _get_home_plate_xy(traj_with_spin, home_with, home_plate_y)
    x_no, z_no = _get_home_plate_xy(traj_no_spin, home_no, home_plate_y)
    delta_x = x_with - x_no
    delta_z = z_with - z_no
    if summary:
        v0_ms = summary['initial_velocity_mps']
        vend_ms = summary['final_velocity_mps']
        v0_kmh = v0_ms * 3.6
        vend_kmh = vend_ms * 3.6
        print("\n=== Simulation summary ===")
        print(f"Release speed: {v0_ms:.2f} m/s ({v0_kmh:.1f} km/h)")
        print(f"Final speed: {vend_ms:.2f} m/s ({vend_kmh:.1f} km/h)")
        print(f"Max height: {summary['max_height']:.2f} m")
        print(f"Final Z: {summary['final_position'][2]:.3f} m")
        print(f"Total time: {summary['total_time']:.3f} s")
        if summary['home_plate_crossing']:
            v_home = summary['home_plate_crossing']['v']
            print(f"Speed at home plate: {v_home:.2f} m/s ({v_home * 3.6:.1f} km/h)")

    print("\n=== Home plate: with spin vs no spin ===")
    print(f"  With spin: X = {x_with:.4f} m, Z = {z_with:.4f} m")
    print(f"  No spin:   X = {x_no:.4f} m, Z = {z_no:.4f} m")
    print(f"  Delta (with - no): dX = {delta_x:.4f} m, dZ = {delta_z:.4f} m")

    plot_spin_comparison(traj_with_spin, traj_no_spin, home_with, home_no, home_plate_y)
    plot_spin_comparison_3d(traj_with_spin, traj_no_spin, home_with, home_no, home_plate_y)
    anim = animate_spin_comparison_3d(traj_with_spin, traj_no_spin, interval_ms=24)

    sim.trajectory = traj_with_spin
    sim.home_plate_crossing = home_with
    sim.export_to_csv('trajectory_output2.csv')
    sim.plot_time_series()
    sim.plot_all_projections()
    sim.plot_trajectory_3d()
    
    return sim


if __name__ == "__main__":
    sim = main()

