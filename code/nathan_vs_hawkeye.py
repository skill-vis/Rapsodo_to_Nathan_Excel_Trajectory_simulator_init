"""HawkEye 実測 ω (pitch.spin.direction) vs Nathan Pass1/Pass2 推定の比較。

手順:
  1. .action ファイルを読み込み
  2. `pitch.spin.direction` (RPM ベクトル) を HawkEye 実測 ω として取得
  3. `samples.ball` から 9 パラメータ定加速度フィット (release 直後の区間で)
  4. fit 値を ft 単位へ変換し、nathan_two_pass_spin で Pass1/Pass2 を推定
  5. 3 者 (HawkEye / Pass1 / Pass2) を並べて出力
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from nathan_two_pass_spin import (
    estimate_spin_two_pass, FPS_TO_MS, RPM_TO_RADS, RADS_TO_RPM,
)

MS_TO_FPS = 1.0 / FPS_TO_MS  # 3.28084


# 6 既存セッション (baseball.skill-vis.com 永続化分) の .action マップ
ACTION_FILES = {
    '232b1545': '/Users/ohta/Library/CloudStorage/Dropbox/．sync/．NTT/_HawkEye/20240412-L-H-1-180000-Json-Tag-V1/2024_31_5532_136_2024-04-12--19-01-34.533.baseball.action',
    '9779f979': '/Users/ohta/Library/CloudStorage/Dropbox/．sync/．NTT/_HawkEye/20240412-L-H-1-180000-Json-Tag-V1/2024_31_5532_208_2024-04-12--19-39-00.866.baseball.action',
    '986a97f0': '/Users/ohta/Library/CloudStorage/Dropbox/．sync/．NTT/_HawkEye/20240412-L-H-1-180000-Json-Tag-V1/2024_31_5532_119_2024-04-12--18-54-00.687.baseball.action',
    '60712783': '/Users/ohta/Library/CloudStorage/Dropbox/．sync/．NTT/_HawkEye/20240412-L-H-1-180000-Json-Tag-V1/2024_31_5532_30_2024-04-12--18-13-24.150.baseball.action',
    'a02f82e2': '/Users/ohta/Library/CloudStorage/Dropbox/．sync/．NTT/_HawkEye/20240412-L-H-1-180000-Json-Tag-V1/2024_31_5532_15_2024-04-12--18-05-29.691.baseball.action',
    'a26cc7cc': '/Users/ohta/Library/CloudStorage/Dropbox/．sync/．NTT/_HawkEye/20240412-L-H-1-180000-Json-Tag-V1/2024_31_5532_17_2024-04-12--18-06-25.832.baseball.action',
}


def _fit_9param_from_samples(samples_ball, t_release):
    """
    samples.ball [{time, pos:[x,y,z], vel:[vx,vy,vz]}, ...] から Statcast 風
    9 パラメータ定加速度フィットを作る。出力単位は ft / ft/s / ft/s^2。

    release ~ home plate 通過までの区間に限定。
    """
    pts = [p for p in samples_ball if p.get('pos') and p.get('time') is not None]
    if len(pts) < 5:
        return None
    t0 = pts[0]['time']
    t_rel = (t_release if t_release is not None
             else t0)  # fallback
    # ball 軌道は ~0.4 秒で home plate に届く。全区間を使う。
    t = np.array([p['time'] - t_rel for p in pts])
    x_m = np.array([p['pos'][0] for p in pts])
    y_m = np.array([p['pos'][1] for p in pts])
    z_m = np.array([p['pos'][2] for p in pts])
    # 2 次多項式フィット: p(t) = p0 + v0 t + (a/2) t^2
    def fit_axis(ts, ps):
        A = np.stack([np.ones_like(ts), ts, 0.5 * ts**2], axis=1)
        coef, *_ = np.linalg.lstsq(A, ps, rcond=None)
        p0, v0, a = coef
        return float(p0), float(v0), float(a)
    x0_m, vx0_m, ax_m = fit_axis(t, x_m)
    y0_m, vy0_m, ay_m = fit_axis(t, y_m)
    z0_m, vz0_m, az_m = fit_axis(t, z_m)
    # Convert m → ft for statcast-style input
    return {
        'vx0': vx0_m * MS_TO_FPS,
        'vy0': vy0_m * MS_TO_FPS,
        'vz0': vz0_m * MS_TO_FPS,
        'ax':  ax_m  * MS_TO_FPS,
        'ay':  ay_m  * MS_TO_FPS,
        'az':  az_m  * MS_TO_FPS,
        'release_pos_x': x0_m * MS_TO_FPS,
        'release_pos_y': y0_m * MS_TO_FPS,
        'release_pos_z': z0_m * MS_TO_FPS,
    }


def _hawkeye_omega(raw: Dict):
    """HawkEye .action から ω ベクトル (rad/s) を取り出す。
    pitch.spin.direction は RPM 成分。"""
    pitches = [e for e in raw.get('events', []) if e.get('type') == 'Pitch']
    if not pitches:
        return None, None, None
    sp = pitches[0].get('spin') or {}
    dirn = sp.get('direction')
    if not dirn or len(dirn) != 3:
        return None, None, None
    # HawkEye direction 単位 = RPM (magnitude = rpm と一致することを確認済)
    wx = dirn[0] * RPM_TO_RADS
    wy = dirn[1] * RPM_TO_RADS
    wz = dirn[2] * RPM_TO_RADS
    total_rpm = float(sp.get('rpm') or math.sqrt(
        dirn[0]**2 + dirn[1]**2 + dirn[2]**2))
    return (wx, wy, wz), total_rpm, pitches[0].get('refinedReleaseTimeUTC')


def compare_pitch(action_path: str, session_id: str = '') -> Dict:
    with open(action_path) as f:
        raw = json.load(f)
    pitcher_throws = raw.get('pitcherThrows', 'R')
    label = f'#{raw.get("sequences", {}).get("pitch", "?")} {raw.get("pitcher", "?")} vs {raw.get("batter", "?")}'

    # HawkEye 実測
    he_omega, he_total_rpm, _ = _hawkeye_omega(raw)
    if he_omega is None:
        return {'session_id': session_id, 'label': label, 'error': 'no spin data'}

    # 9-param fit
    samples_ball = raw.get('samples', {}).get('ball', [])
    pitch_evt = [e for e in raw['events'] if e['type'] == 'Pitch'][0]
    t_rel_utc = pitch_evt.get('refinedReleaseTimeUTC')
    # ball sample の time は相対秒。release time は 0 付近。
    # samples.ball[0].time が release 直後なのでそのまま origin に
    t0_rel = samples_ball[0]['time'] if samples_ball else 0
    # 実際は refinedReleaseTime で合わせるべきだが、fit は相対時刻で OK
    fit = _fit_9param_from_samples(samples_ball, t0_rel)
    if fit is None:
        return {'session_id': session_id, 'label': label, 'error': 'fit failed'}

    # Nathan 2-pass 推定 — Pass2 でエラーが出ても Pass1 は保持
    result = {'pass1': None, 'pass2': None}
    pass2_err = None
    try:
        result = estimate_spin_two_pass(fit, spin_rate_rpm=he_total_rpm,
                                        pitcher_hand=pitcher_throws[0])
    except Exception as e:
        # Pass 2 で例外 → Pass 1 のみフォールバック
        try:
            from nathan_two_pass_spin import _pass1
            p1 = _pass1(fit, he_total_rpm, pitcher_throws[0])
            result = {'pass1': p1.to_dict() if p1 else None, 'pass2': None}
        except Exception as e2:
            pass2_err = f'{type(e).__name__}: {e}'
        else:
            pass2_err = f'{type(e).__name__}: {e}'
    return {
        'session_id': session_id,
        'label': label,
        'pitcher_throws': pitcher_throws,
        'he_total_rpm': he_total_rpm,
        'he_omega_rads': he_omega,
        'pass1': result.get('pass1'),
        'pass2': result.get('pass2'),
        'pass2_err': pass2_err,
    }


def _fmt_omega(o_rads, label='ω'):
    """(wx, wy, wz) を rad/s と rpm で表示."""
    wx, wy, wz = o_rads
    mag_rpm = math.sqrt(wx**2 + wy**2 + wz**2) * RADS_TO_RPM
    return (f'{label}: ({wx:+7.1f}, {wy:+7.1f}, {wz:+7.1f}) rad/s  '
            f'|ω|={mag_rpm:.0f} rpm')


def _diff_angle_deg(o1, o2):
    """2 つの ω ベクトル間の角度差 (度)."""
    a = np.array(o1); b = np.array(o2)
    if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
        return None
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def main():
    for sid, fp in ACTION_FILES.items():
        if not os.path.isfile(fp):
            print(f'{sid}: file not found  {fp}')
            continue
        res = compare_pitch(fp, sid)
        print(f'=== {sid}  {res.get("label")}  (pitcher: {res.get("pitcher_throws")}) ===')
        if 'error' in res:
            print(f'  SKIP: {res["error"]}'); print(); continue
        he = res['he_omega_rads']
        print('  ' + _fmt_omega(he, 'HawkEye'))
        if res['pass1']:
            p1 = res['pass1']
            o1 = (p1['omega_x_rads'], p1['omega_y_rads'], p1['omega_z_rads'])
            print('  ' + _fmt_omega(o1, 'Pass 1 '))
            ang1 = _diff_angle_deg(he, o1)
            d1 = math.sqrt(sum((a - b)**2 for a, b in zip(he, o1))) * RADS_TO_RPM
            print(f'    Pass1 axis-angle err: {ang1:.1f}°,  |Δω|={d1:.0f} rpm,  efficiency={p1["efficiency"]:.2f}')
        if res['pass2']:
            p2 = res['pass2']
            o2 = (p2['omega_x_rads'], p2['omega_y_rads'], p2['omega_z_rads'])
            print('  ' + _fmt_omega(o2, 'Pass 2 '))
            ang2 = _diff_angle_deg(he, o2)
            d2 = math.sqrt(sum((a - b)**2 for a, b in zip(he, o2))) * RADS_TO_RPM
            print(f'    Pass2 axis-angle err: {ang2:.1f}°,  |Δω|={d2:.0f} rpm,  efficiency={p2["efficiency"]:.2f}')
        elif res.get('pass2_err'):
            print(f'  Pass 2 : FAILED  ({res["pass2_err"]})')
        print()


if __name__ == '__main__':
    main()
