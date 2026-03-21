"""
Rapsodo 計測値 → Nathan Excel 用入力への変換

実装本体は code/ フォルダ（Git 用の最小依存セット）。
このファイルは後方互換のため同じ API を再エクスポートする。
"""

import importlib.util
from pathlib import Path

_CODE_RAPSODO = Path(__file__).resolve().parent / "code" / "rapsodo_to_nathan.py"
_spec = importlib.util.spec_from_file_location("_rapsodo_nathan_code", _CODE_RAPSODO)
_impl = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_impl)

# code/ から再エクスポート（MyBallTrajectorySim には依存しない）
PitchParameters = _impl.PitchParameters
angular_velocity_xyz_to_backspin_sidespin_wg = _impl.angular_velocity_xyz_to_backspin_sidespin_wg
clock_time_to_angle_deg = _impl.clock_time_to_angle_deg
rapsodo_velocity_to_theta_phi = _impl.rapsodo_velocity_to_theta_phi
rapsodo_spin_to_omega_xyz = _impl.rapsodo_spin_to_omega_xyz
rapsodo_to_nathan = _impl.rapsodo_to_nathan
pitch_parameters_to_nathan_excel_units = _impl.pitch_parameters_to_nathan_excel_units
format_nathan_excel_line = _impl.format_nathan_excel_line
RPM_TO_RAD_S = _impl.RPM_TO_RAD_S
KMH_TO_MPS = _impl.KMH_TO_MPS
MPS_TO_MPH = _impl.MPS_TO_MPH
M_TO_FT = _impl.M_TO_FT

__all__ = [
    "PitchParameters",
    "angular_velocity_xyz_to_backspin_sidespin_wg",
    "clock_time_to_angle_deg",
    "rapsodo_velocity_to_theta_phi",
    "rapsodo_spin_to_omega_xyz",
    "rapsodo_to_nathan",
    "pitch_parameters_to_nathan_excel_units",
    "format_nathan_excel_line",
    "RPM_TO_RAD_S",
    "KMH_TO_MPS",
    "MPS_TO_MPH",
    "M_TO_FT",
]


def main():
    _impl.main()


if __name__ == "__main__":
    main()
