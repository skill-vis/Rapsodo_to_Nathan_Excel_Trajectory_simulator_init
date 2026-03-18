"""
Rapsodo 時計表記（HH:MM）→ 角度 [deg] 変換（利き腕による鏡映オプション付き）。
"""

# 00:00 = -90 deg, 03:00 = 0 deg, 06:00 = 90 deg, 09:00 = 180 deg, 12:00 = -90
CLOCK_HOURS_TO_DEG = 30.0
CLOCK_REFERENCE_HOUR = 3.0


def clock_time_to_angle_deg(time_str: str, *, pitcher_hand: str | None = None) -> float:
    """
    12時間制の時計時刻を角度 [deg] に変換する。

    変換ルール: 00:00 = -90 deg, 03:00 = 0 deg
    pitcher_hand が 'L' のとき、時計を左右反転（3:00↔9:00）してから変換する。
    """
    parts = time_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError(
            f"時刻は 'HH:MM' 形式である必要があります（例: 03:00, 12:30）: {time_str!r}"
        )
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
    except ValueError as e:
        raise ValueError(f"時刻の解析に失敗しました: {time_str!r}") from e

    if not (0 <= hours <= 12):
        raise ValueError(f"時間は 0–12 の範囲である必要があります: {hours}")
    if not (0 <= minutes < 60):
        raise ValueError(f"分は 0–59 の範囲である必要があります: {minutes}")

    h = 0 if hours == 12 else int(hours)
    total_minutes = h * 60 + int(minutes)

    if pitcher_hand is not None and str(pitcher_hand).upper().startswith("L"):
        total_minutes = (12 * 60 - total_minutes) % (12 * 60)

    total_hours = total_minutes / 60.0
    return (total_hours - CLOCK_REFERENCE_HOUR) * CLOCK_HOURS_TO_DEG
