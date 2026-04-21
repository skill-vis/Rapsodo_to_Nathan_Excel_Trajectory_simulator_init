"""Migrate persisted HawkEye session JSONs to the windowed computation.

Spec:
  - 計算窓 = [swing_start_frame, impact_frame)  (impact 自身を除外)
      * ヒット swing: impact = ボール contact
      * 空振り:        impact = ホームベース通過時刻 (既に保存済み)
  - spline weight:
      * grip 速度:  w = 0.00005 (user 指定)
      * head 速度:  w = 0.0005  (従来どおり)
      * bat_vec 角速度: w = 0.05 (従来どおり)
  - swing_start は従来どおりグリップ bat-axial 速度の zero-cross で検出
  - vel_time / isa_time は bat_time と同じ長さに保ち、
    window 外のフレームは null とする (Chart.js が gap として描画)
  - release_time マーカーは変更しない (ユーザ要件 #2, #5)
  - 新フィールド impact_minus_nr_ms = (impact - NR) × 1000 を追加 (要件 #8)

Run:
    python migrate_hawkeye_sessions.py [--dry-run] [session_id ...]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Optional, List

import numpy as np
from scipy.interpolate import UnivariateSpline


SESSION_DIR = os.path.join(os.path.dirname(__file__), "hawkeye_sessions")
SF = 300.0
BAT_LENGTH = 0.83

# Local-only: 6 既存セッション → .action ファイル (HawkEye 公式の bat speed 値抽出用)
ACTION_FILE_MAP = {
    '232b1545': '/Users/ohta/Library/CloudStorage/Dropbox/．sync/．NTT/_HawkEye/20240412-L-H-1-180000-Json-Tag-V1/2024_31_5532_136_2024-04-12--19-01-34.533.baseball.action',
    '9779f979': '/Users/ohta/Library/CloudStorage/Dropbox/．sync/．NTT/_HawkEye/20240412-L-H-1-180000-Json-Tag-V1/2024_31_5532_208_2024-04-12--19-39-00.866.baseball.action',
    '986a97f0': '/Users/ohta/Library/CloudStorage/Dropbox/．sync/．NTT/_HawkEye/20240412-L-H-1-180000-Json-Tag-V1/2024_31_5532_119_2024-04-12--18-54-00.687.baseball.action',
    '60712783': '/Users/ohta/Library/CloudStorage/Dropbox/．sync/．NTT/_HawkEye/20240412-L-H-1-180000-Json-Tag-V1/2024_31_5532_30_2024-04-12--18-13-24.150.baseball.action',
    'a02f82e2': '/Users/ohta/Library/CloudStorage/Dropbox/．sync/．NTT/_HawkEye/20240412-L-H-1-180000-Json-Tag-V1/2024_31_5532_15_2024-04-12--18-05-29.691.baseball.action',
    'a26cc7cc': '/Users/ohta/Library/CloudStorage/Dropbox/．sync/．NTT/_HawkEye/20240412-L-H-1-180000-Json-Tag-V1/2024_31_5532_17_2024-04-12--18-06-25.832.baseball.action',
}


def _extract_hawkeye_bat_speeds(action_file: str):
    """HawkEye .action から Hit event の sweetSpot / impactPoint バット速度を抽出。
    Hit event が無ければ (None, None) を返す (空振り)。"""
    if not action_file or not os.path.isfile(action_file):
        return None, None
    try:
        with open(action_file) as f:
            raw = json.load(f)
        hit_events = [e for e in raw.get('events', []) if e.get('type') == 'Hit']
        if not hit_events:
            return None, None
        bat = hit_events[0].get('before', {}).get('bat', {})
        sweet = bat.get('sweetSpot', {}).get('speed', {}).get('kph')
        impactpt = bat.get('impactPoint', {}).get('speed', {}).get('kph')
        return (float(sweet) if sweet is not None else None,
                float(impactpt) if impactpt is not None else None)
    except Exception:
        return None, None

# spline 重み
W_BAT_VEL = 0.05
W_GRIP = 0.00005     # ← 新方針 (要件 #7)
W_HV = 0.0005


def _norm(a): return np.linalg.norm(a, axis=-1)


def _unit_vec(a):
    n = _norm(a)[:, np.newaxis]
    n = np.where(n == 0, 1.0, n)
    return a / n


def _spline(data: np.ndarray, weight: float, sf: float,
            order: int = 1, k: int = 3) -> np.ndarray:
    """s = weight * n 形式 (send_to_simulator.py 互換)。
    data は (n, 3) を想定。n (フレーム数) は窓内のみ。"""
    n = len(data)
    t = np.arange(n) / sf
    out = np.zeros_like(data, dtype=float)
    for d in range(data.shape[1]):
        out[:, d] = UnivariateSpline(t, data[:, d], s=weight * n, k=k) \
            .derivative(order)(t)
    return out


def _detect_swing_start(grip: np.ndarray, bat_unit: np.ndarray,
                        impact_frame: int, time_arr: np.ndarray,
                        sf: float) -> Optional[float]:
    """グリップ bat-axial 速度の最後の zero-cross (+→−) を検出。

    impact から時間を遡り、最初に「負 → 非負」へ変化する境界を swing_start
    と定義する (= forward time で見れば最も impact に近い +→− zero-cross)。
    Setup 期の小さなゆらぎに引きずられず、真の downswing のリリース点を捉える。
    impact 手前 0.1s は振動ゾーンとして除外。
    """
    if impact_frame < 20:
        return None
    grip_pre = grip[:impact_frame]
    bat_unit_pre = bat_unit[:impact_frame]
    t_s = np.arange(len(grip_pre)) / sf
    vel = np.zeros_like(grip_pre, dtype=float)
    for ax in range(3):
        spl = UnivariateSpline(t_s, grip_pre[:, ax], s=0.02, k=4)
        vel[:, ax] = spl.derivative(n=1)(t_s)
    axial = np.sum(vel * bat_unit_pre, axis=1)
    search_limit = max(1, int(impact_frame - 0.1 * sf))
    search_limit = min(search_limit, len(axial))
    if search_limit <= 1:
        return None
    # Scan backward: 最後の +→− zero-cross を採用
    for i in range(search_limit - 1, 0, -1):
        if axial[i - 1] >= 0 and axial[i] < 0:
            t0_ = time_arr[i - 1]
            t1_ = time_arr[i]
            v0_ = axial[i - 1]
            v1_ = axial[i]
            return float(t0_ + (0 - v0_) / (v1_ - v0_) * (t1_ - t0_))
    return None


def _closest_approach_time(session: dict) -> Optional[float]:
    """Whiff の最接近時刻を計算。

    ボール軌跡 (ball_time/ball_pos) を末端で線形補外しつつ、
    バット線分 (handle→head, bat_time で補間) との最小距離を与える時刻 t* を返す。
    ヒットセッション (hit_ball_pos が非空) では None を返す (呼び出し側で無視)。
    """
    hit_ball = session.get('hit_ball_pos') or []
    if len(hit_ball) > 0:
        return None

    ball_time = np.asarray(session.get('ball_time') or [], dtype=float)
    ball_pos = np.asarray(session.get('ball_pos') or [], dtype=float)
    bat_time = np.asarray(session.get('bat_time') or [], dtype=float)
    bat_head = np.asarray(session.get('bat_head') or [], dtype=float)
    bat_handle = np.asarray(session.get('bat_handle') or [], dtype=float)

    if ball_time.size < 2 or ball_pos.shape[0] < 2:
        return None
    if bat_time.size < 2 or bat_head.shape[0] < 2:
        return None

    # ボール末端速度 (線形補外用)
    dt_last = ball_time[-1] - ball_time[-2]
    if dt_last <= 0:
        return None
    ball_vel = (ball_pos[-1] - ball_pos[-2]) / dt_last

    def ball_at(t: float) -> np.ndarray:
        if t <= ball_time[-1]:
            return np.array([np.interp(t, ball_time, ball_pos[:, i])
                             for i in range(3)])
        return ball_pos[-1] + ball_vel * (t - ball_time[-1])

    # バット補間 (bat_time は等間隔仮定で OK, np.interp を使用)
    def bat_seg_at(t: float):
        handle = np.array([np.interp(t, bat_time, bat_handle[:, i])
                           for i in range(3)])
        head = np.array([np.interp(t, bat_time, bat_head[:, i])
                         for i in range(3)])
        return handle, head

    # 探索区間 [max(release, bat 開始), min(bat 終了, ball 終了+300ms)]
    t_start = max(ball_time[0], bat_time[0])
    t_end = min(bat_time[-1], ball_time[-1] + 0.3)
    if t_end <= t_start:
        return None

    # 1500Hz (0.67ms 刻み) で総当り — 3m/s の相対運動で 2mm 分解能
    ts = np.arange(t_start, t_end, 1.0 / 1500.0)

    best_t = None
    best_d = np.inf
    for t in ts:
        handle, head = bat_seg_at(float(t))
        d_vec = head - handle
        dd = float(d_vec @ d_vec)
        b = ball_at(float(t))
        if dd < 1e-10:
            closest = handle
        else:
            u = float((b - handle) @ d_vec / dd)
            u = max(0.0, min(1.0, u))
            closest = handle + u * d_vec
        dist = float(np.linalg.norm(b - closest))
        if dist < best_d:
            best_d = dist
            best_t = float(t)

    return best_t


def _fill_windowed(n_total: int, start: int, end: int,
                   values: np.ndarray) -> List:
    """長さ n_total のリストを作り、[start:end] を values で埋め、他は None。"""
    out: List = [None] * n_total
    for i, v in enumerate(values):
        out[start + i] = (float(v)
                          if np.isscalar(v) or isinstance(v, np.generic)
                          else [float(x) for x in v])
    return out


def migrate_session(session: dict, sf: float = SF,
                    session_id: Optional[str] = None) -> dict:
    # Whiff の場合は impact_time をボール-バット最接近時刻に更新
    whiff_impact = _closest_approach_time(session)
    if whiff_impact is not None:
        # 記録: 元の値をログに残すために impact_time_event として保存
        if 'impact_time_event' not in session:
            session = dict(session)
            session['impact_time_event'] = session.get('impact_time')
        session['impact_time'] = whiff_impact

    bat_time = np.asarray(session['bat_time'], dtype=float)
    bat_head = np.asarray(session['bat_head'], dtype=float)
    bat_handle = np.asarray(session['bat_handle'], dtype=float)
    impact_time = float(session['impact_time'])
    n_total = len(bat_time)

    impact_frame = int(np.argmin(np.abs(bat_time - impact_time)))

    # --- フル範囲 (swing_start 検出用) ---
    bat_vec_full = bat_head - bat_handle
    bat_unit_full = _unit_vec(bat_vec_full)
    grip_full = bat_handle + 0.1 * bat_unit_full

    swing_start_time = _detect_swing_start(grip_full, bat_unit_full,
                                            impact_frame, bat_time, sf)
    if swing_start_time is None:
        raise RuntimeError('swing_start detection failed')
    swing_start_frame = int(np.argmin(np.abs(bat_time - swing_start_time)))

    # --- 窓内 [swing_start, impact) で spline fit ---
    start = swing_start_frame
    end = impact_frame  # exclusive
    if end - start < 10:
        raise RuntimeError(f'window too short: {end-start} frames')

    head_win = bat_head[start:end]
    handle_win = bat_handle[start:end]
    time_win = bat_time[start:end]

    bat_vec_win = head_win - handle_win
    bat_unit_win = _unit_vec(bat_vec_win)
    grip_win = handle_win + 0.1 * bat_unit_win

    bat_vel = _spline(bat_vec_win, W_BAT_VEL, sf)
    av_unit = _unit_vec(np.cross(bat_vec_win, bat_vel))
    av_vec = _norm(bat_vel)[:, np.newaxis] * av_unit / BAT_LENGTH
    av_norm = _norm(av_vec)

    grip_vel = _spline(grip_win, W_GRIP, sf)
    head_vel = _spline(head_win, W_HV, sf)

    grip_speed_kmh = _norm(grip_vel) * 3.6
    head_speed_kmh = _norm(head_vel) * 3.6

    # --- Head velocity を FULL 範囲でも計算 (表示は全区間、数値比較用) ---
    head_vel_full = _spline(bat_head, W_HV, sf)
    head_speed_kmh_full = _norm(head_vel_full) * 3.6
    our_head_at_impact_kph = float(head_speed_kmh_full[impact_frame])
    # peak は swing 窓内でのみ探索 (外のノイズ除外)
    search_hi = min(impact_frame + 1, len(head_speed_kmh_full))
    search_lo = max(0, start)
    if search_hi > search_lo:
        max_rel = int(np.argmax(head_speed_kmh_full[search_lo:search_hi]))
        our_head_max_frame = search_lo + max_rel
        our_head_max_kph = float(head_speed_kmh_full[our_head_max_frame])
        our_head_max_time = float(bat_time[our_head_max_frame])
    else:
        our_head_max_kph = our_head_at_impact_kph
        our_head_max_time = impact_time

    # --- HawkEye 公式 bat speed (hits のみ) ---
    action_file = ACTION_FILE_MAP.get(session_id) if session_id else None
    hawkeye_sweet_kph, hawkeye_impactpt_kph = _extract_hawkeye_bat_speeds(action_file)

    # ISA 位置 (窓内)
    isa_pos = np.zeros_like(grip_win)
    for i in range(len(grip_win)):
        if av_norm[i] > 0.1:
            isa_pos[i] = (np.cross(av_vec[i], grip_vel[i])
                          / (av_norm[i] ** 2) + grip_win[i])
        else:
            isa_pos[i] = grip_win[i]

    # --- NR (窓内でのグリップ速度最大) ---
    grip_speed = _norm(grip_vel)
    nr_rel = int(np.argmax(grip_speed))
    grip_max_frame = nr_rel + start
    grip_max_time = float(bat_time[grip_max_frame])

    impact_minus_nr_ms = (impact_time - grip_max_time) * 1000.0

    # --- 出力 ---
    new = dict(session)
    new['vel_time'] = bat_time.tolist()
    # Head 速度は full 範囲 (null 無し) — 境界影響少ないため全区間で綺麗
    new['vel_head'] = [float(v) for v in head_speed_kmh_full]
    # Grip 速度は窓外 null (post-impact 除外が計算誤差低減に効くため)
    new['vel_grip'] = _fill_windowed(n_total, start, end, grip_speed_kmh)
    new['isa_time'] = bat_time.tolist()
    new['isa_pos'] = _fill_windowed(n_total, start, end, isa_pos)
    new['isa_axis'] = _fill_windowed(n_total, start, end, av_vec)
    new['isa_omega'] = _fill_windowed(n_total, start, end, av_norm)
    new['grip_max_time'] = grip_max_time
    new['swing_start_time'] = swing_start_time
    new['impact_minus_nr_ms'] = impact_minus_nr_ms
    # バット速度比較用フィールド (hits: HawkEye 公式 / 全セッション: ours)
    new['hawkeye_bat_sweet_kph'] = hawkeye_sweet_kph
    new['hawkeye_bat_impactpt_kph'] = hawkeye_impactpt_kph
    new['our_head_at_impact_kph'] = our_head_at_impact_kph
    new['our_head_max_kph'] = our_head_max_kph
    new['our_head_max_time'] = our_head_max_time
    # impact_time, release_time, 他は保持
    return new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('session_ids', nargs='*')
    args = ap.parse_args()

    if not os.path.isdir(SESSION_DIR):
        print(f'Session dir not found: {SESSION_DIR}', file=sys.stderr)
        return 1

    if args.session_ids:
        files = [f'{sid}.json' for sid in args.session_ids]
    else:
        files = sorted(f for f in os.listdir(SESSION_DIR)
                       if f.endswith('.json'))

    backup_dir = os.path.join(
        SESSION_DIR, f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    changed = 0

    for fname in files:
        path = os.path.join(SESSION_DIR, fname)
        with open(path) as f:
            sess = json.load(f)
        label = (sess.get('label') or '')[:55]
        sid = fname.replace('.json', '')

        try:
            new_sess = migrate_session(sess, session_id=sid)
        except Exception as e:
            print(f'  SKIP {fname}: {type(e).__name__}: {e}')
            continue

        impact_t = new_sess['impact_time']
        ss_new = new_sess['swing_start_time']
        gm_new = new_sess['grip_max_time']
        gap_ms = new_sess['impact_minus_nr_ms']
        sweet = new_sess.get('hawkeye_bat_sweet_kph')
        impactpt = new_sess.get('hawkeye_bat_impactpt_kph')
        our_i = new_sess.get('our_head_at_impact_kph')
        our_m = new_sess.get('our_head_max_kph')
        print(f'  {fname}  [{label}]')
        print(f'    swing_start = {ss_new:.3f}s  NR = {gm_new:.3f}s  '
              f'impact = {impact_t:.3f}s  |  impact−NR = {gap_ms:+.1f} ms')
        print(f'    HawkEye sweet = {sweet and f"{sweet:.1f}" or "n/a"} kph, '
              f'impactPt = {impactpt and f"{impactpt:.1f}" or "n/a"} kph | '
              f'ours@impact = {our_i:.1f} kph, ours_max = {our_m:.1f} kph')

        if args.dry_run:
            continue

        if not os.path.isdir(backup_dir):
            os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(backup_dir, fname))
        with open(path, 'w') as f:
            json.dump(new_sess, f, ensure_ascii=False)
        changed += 1

    if args.dry_run:
        print('\n(dry-run: no files written)')
    else:
        print(f'\nMigrated {changed} sessions. Backups in: {backup_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
