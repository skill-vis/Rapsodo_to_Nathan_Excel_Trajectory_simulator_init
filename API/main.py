"""
FastAPI server for baseball pitch trajectory simulation.

Wraps BallTrajectorySimulator2 from MyBallTrajectorySim_E.py and exposes
endpoints for trajectory simulation and Statcast data browsing.

Deploy on AWS (EC2 / Lambda + API Gateway / ECS) with:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import json
from datetime import datetime
import math
import time
import os
import subprocess
import tempfile
import uuid

from MyBallTrajectorySim_E import (
    BallTrajectorySimulator2,
    PitchParameters,
    EnvironmentParameters,
    IntegrationMethod,
    LiftModel,
)
from statcast_to_sim import statcast_to_sim_params, statcast_to_release, vectorized_bsg_summary

app = FastAPI(
    title="Baseball Trajectory Simulator API",
    description="Pitch trajectory (x, y, z, t) based on Nathan's physics model + Statcast integration",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PitchRequest(BaseModel):
    """Pitch release parameters."""
    x0: float = Field(-0.47, description="Release X position (m)")
    y0: float = Field(16.48, description="Release Y position (m)")
    z0: float = Field(1.5, description="Release Z position (m)")
    v0_mps: float = Field(37.611111, description="Release speed (m/s)")
    theta_deg: float = Field(0.1, description="Vertical release angle (deg)")
    phi_deg: float = Field(2.6, description="Horizontal release angle (deg)")
    backspin_rpm: float = Field(1062.74, description="Backspin (rpm)")
    sidespin_rpm: float = Field(-1377.89, description="Sidespin (rpm)")
    wg_rpm: float = Field(451.0, description="Gyrospin (rpm)")
    batter_hand: str = Field("R", description="Batter handedness (R/L)")


class EnvRequest(BaseModel):
    """Environment parameters (optional)."""
    temp_F: float = Field(70.0, description="Temperature (deg F)")
    elev_m: float = Field(4.572, description="Elevation (m)")
    relative_humidity: float = Field(50.0, description="Relative humidity (%)")
    pressure_mmHg: float = Field(760.0, description="Atmospheric pressure (mmHg)")
    vwind_mph: float = Field(0.0, description="Wind speed (mph)")
    phiwind_deg: float = Field(0.0, description="Wind direction (deg)")
    hwind_m: float = Field(0.0, description="Wind height threshold (m)")


class SimulationRequest(BaseModel):
    """Full simulation request."""
    pitch: PitchRequest = Field(default_factory=PitchRequest)
    env: Optional[EnvRequest] = Field(default_factory=EnvRequest)
    integration_method: str = Field("rk4", description="euler / nathan / rk4")
    lift_model: str = Field("nathan_exp", description="nathan_exp / rational")
    max_time: float = Field(1.0, description="Max simulation time (s)")
    save_interval: int = Field(1, description="Save every N steps")
    fixed_cd: Optional[float] = Field(None, description="Constant Cd (overrides Nathan model)")
    fixed_cl: Optional[float] = Field(None, description="Constant Cl (overrides Nathan model)")


class TrajectoryPoint(BaseModel):
    t: float
    x: float
    y: float
    z: float
    vx: Optional[float] = None
    vy: Optional[float] = None
    vz: Optional[float] = None
    ax: Optional[float] = None
    ay: Optional[float] = None
    az: Optional[float] = None
    drag_x: Optional[float] = None
    drag_y: Optional[float] = None
    drag_z: Optional[float] = None
    magnus_x: Optional[float] = None
    magnus_y: Optional[float] = None
    magnus_z: Optional[float] = None
    gravity_z: Optional[float] = None
    cd: Optional[float] = None
    cl: Optional[float] = None


class SimulationResponse(BaseModel):
    trajectory: list[TrajectoryPoint]
    home_plate: Optional[TrajectoryPoint] = None
    num_points: int


# ---------------------------------------------------------------------------
# Statcast models
# ---------------------------------------------------------------------------

class StatcastSearchRequest(BaseModel):
    last_name: str
    first_name: Optional[str] = None
    year: int = 2024


class PlayerInfo(BaseModel):
    index: int
    first_name: str
    last_name: str
    mlbam_id: int
    years: str


class StatcastSearchResponse(BaseModel):
    players: list[PlayerInfo]


class StatcastFetchRequest(BaseModel):
    mlbam_id: int
    year: int = 2024


class GameDateInfo(BaseModel):
    date: str
    pitch_count: int
    pitch_types: str


class StatcastGamesResponse(BaseModel):
    games: list[GameDateInfo]
    total_pitches: int


class StatcastPitchesRequest(BaseModel):
    mlbam_id: int
    year: int = 2024
    date: str


class PitchInfo(BaseModel):
    index: int
    pitch_type: str
    release_speed: Optional[float]
    release_spin_rate: Optional[float]
    spin_axis: Optional[float]
    pfx_x: Optional[float] = None       # spin-induced movement (feet)
    pfx_z: Optional[float] = None
    effective_speed: Optional[float] = None
    arm_angle: Optional[float] = None
    inning: Optional[int]
    at_bat_number: Optional[int]
    pitch_number: Optional[int]
    description: Optional[str]
    events: Optional[str] = None        # at-bat result (strikeout, single, etc.)
    plate_x: Optional[float]
    plate_z: Optional[float]
    stand: Optional[str]
    batter_id: Optional[int] = None     # batter MLBAM ID
    batter_name: Optional[str] = None   # batter name
    balls: Optional[int] = None
    strikes: Optional[int] = None
    outs: Optional[int] = None
    launch_speed: Optional[float] = None
    launch_angle: Optional[float] = None


class StatcastPitchesResponse(BaseModel):
    pitches: list[PitchInfo]


class StatcastSimRequest(BaseModel):
    mlbam_id: int
    year: int = 2024
    date: str
    pitch_index: int
    spin_method: str = "bsg"  # "bsg" or "direct"
    cl_mode: str = "nathan_exp"  # "nathan_exp" / "rational" / "nathan" (legacy: cl2=1.12)
    accel_method: bool = False  # True: Nathan (2020) accel-based ω_T estimation


class StatcastSimResponse(BaseModel):
    trajectory: list[TrajectoryPoint]
    trajectory_nospin: list[TrajectoryPoint] = []
    home_plate: Optional[TrajectoryPoint] = None
    home_plate_nospin: Optional[TrajectoryPoint] = None
    home_plate_statcast: Optional[dict] = None
    statcast_pfx: Optional[dict] = None
    release_point: Optional[dict] = None
    pitch_info: dict
    sim_params: dict
    num_points: int


class ComparePitchItem(BaseModel):
    mlbam_id: int
    year: int
    date: str
    pitch_index: int
    label: Optional[str] = None


class CompareRequest(BaseModel):
    pitches: list[ComparePitchItem]
    spin_method: str = "bsg"
    cl_mode: str = "nathan_exp"
    accel_method: bool = False


class CompareResultItem(BaseModel):
    label: Optional[str] = None
    pitcher_name: Optional[str] = None
    pitch_type: Optional[str] = None
    release_speed_mph: Optional[float] = None
    release_spin_rate: Optional[float] = None
    spin_axis: Optional[float] = None
    pfx_x_in: Optional[float] = None
    pfx_z_in: Optional[float] = None
    spin_efficiency: Optional[float] = None
    backspin_rpm: Optional[float] = None
    sidespin_rpm: Optional[float] = None
    wg_rpm: Optional[float] = None
    home_plate_sim: Optional[dict] = None
    home_plate_statcast: Optional[dict] = None
    error_mm: Optional[float] = None
    sim_url: Optional[str] = None


class CompareResponse(BaseModel):
    results: list[CompareResultItem]
    count: int


# ---------------------------------------------------------------------------
# Statcast data cache (avoid re-fetching within same session)
# ---------------------------------------------------------------------------
_statcast_cache: dict = {}
_STATCAST_CACHE_MAX = 3  # max cached pitcher/year combinations


def _get_statcast_data(mlbam_id: int, year: int):
    """Fetch and cache Statcast data for a pitcher/year."""
    import pandas as pd
    cache_key = (mlbam_id, year)
    if cache_key in _statcast_cache:
        return _statcast_cache[cache_key]

    # Evict oldest entries if cache is full
    while len(_statcast_cache) >= _STATCAST_CACHE_MAX:
        oldest_key = next(iter(_statcast_cache))
        del _statcast_cache[oldest_key]

    from pybaseball import statcast_pitcher
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    df = statcast_pitcher(start, end, mlbam_id)
    if df.empty:
        _statcast_cache[cache_key] = df
        return df

    from statcast_fetcher import StatcastFetcher
    cols = [c for c in StatcastFetcher.SIM_COLUMNS if c in df.columns]
    df = df[cols].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    _statcast_cache[cache_key] = df
    return df


# ---------------------------------------------------------------------------
# Player name autocomplete cache
# ---------------------------------------------------------------------------
_player_register = None


def _get_player_register():
    """Load and cache the Chadwick player register for autocomplete."""
    global _player_register
    if _player_register is not None:
        return _player_register
    try:
        from pybaseball import chadwick_register
        _player_register = chadwick_register()
    except Exception:
        _player_register = None
    return _player_register


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/log/memory")
def log_memory():
    """Return current memory usage and cache info."""
    import resource
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    cache_keys = [f"{k[0]}_{k[1]}" for k in _statcast_cache.keys()]
    return {
        "rss_mb": round(rss_mb, 1),
        "statcast_cache_entries": len(_statcast_cache),
        "statcast_cache_keys": cache_keys,
        "player_register_loaded": _player_register is not None,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/statcast/autocomplete")
def statcast_autocomplete(q: str = ""):
    """Return pitcher name suggestions for autocomplete."""
    if len(q) < 2:
        return {"suggestions": []}
    reg = _get_player_register()
    if reg is None:
        return {"suggestions": []}
    q_lower = q.lower()
    matches = reg[
        reg["name_last"].str.lower().str.startswith(q_lower)
        & reg["key_mlbam"].notna()
        & reg["mlb_played_last"].notna()
    ].copy()
    # Sort by most recent players first
    matches = matches.sort_values("mlb_played_last", ascending=False).head(15)
    suggestions = []
    for _, row in matches.iterrows():
        suggestions.append({
            "first_name": str(row.get("name_first", "")),
            "last_name": str(row.get("name_last", "")),
            "mlbam_id": int(row["key_mlbam"]),
            "years": f"{row.get('mlb_played_first', '?')}-{row.get('mlb_played_last', '?')}",
        })
    return {"suggestions": suggestions}


@app.get("/")
def root():
    """Serve frontend HTML."""
    html_path = os.path.join(_BASE_DIR, "static", "index.html")
    return FileResponse(html_path)


# =========================================================================
# Event logging
# =========================================================================
_EVENT_LOG_PATH = os.path.join(_BASE_DIR, "event_log.json")

def _load_events():
    if os.path.exists(_EVENT_LOG_PATH):
        with open(_EVENT_LOG_PATH, "r") as f:
            return json.load(f)
    return []

def _save_events(events):
    with open(_EVENT_LOG_PATH, "w") as f:
        json.dump(events, f, ensure_ascii=False)


class EventInput(BaseModel):
    event: str  # e.g. "csv_import", "csv_export", "simulate", etc.


@app.post("/log/event")
def log_event(data: EventInput, request: Request):
    """Log a UI event with timestamp and client IP."""
    events = _load_events()
    events.append({
        "event": data.event,
        "time": datetime.now().isoformat(),
        "ip": request.client.host if request.client else "unknown",
    })
    _save_events(events)
    return {"ok": True}


@app.get("/log/stats")
def log_stats():
    """Return event counts."""
    events = _load_events()
    counts = {}
    for e in events:
        name = e.get("event", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return {"total": len(events), "counts": counts, "recent": events[-20:]}


# SEO: serve robots.txt and sitemap.xml at root
@app.get("/robots.txt")
async def robots_txt():
    return FileResponse(os.path.join(_BASE_DIR, "static", "robots.txt"), media_type="text/plain")

@app.get("/sitemap.xml")
async def sitemap_xml():
    return FileResponse(os.path.join(_BASE_DIR, "static", "sitemap.xml"), media_type="application/xml")

# Mocap viewer page
@app.get("/mocap")
async def mocap_viewer_page():
    return FileResponse(os.path.join(_BASE_DIR, "static", "mocap_viewer.html"), media_type="text/html")

# Static files (images, etc.)
app.mount("/static", StaticFiles(directory=os.path.join(_BASE_DIR, "static")), name="static")


# ---------------------------------------------------------------------------
# Simulation endpoint (existing)
# ---------------------------------------------------------------------------

@app.post("/simulate", response_model=SimulationResponse)
def simulate(req: SimulationRequest):
    """Run trajectory simulation and return (t, x, y, z) points."""

    method_map = {
        "euler": IntegrationMethod.EULER,
        "nathan": IntegrationMethod.NATHAN,
        "rk4": IntegrationMethod.RK4,
    }
    method = method_map.get(req.integration_method.lower(), IntegrationMethod.RK4)

    lift_map = {
        "nathan_exp": LiftModel.NATHAN_EXP,
        "rational": LiftModel.RATIONAL,
    }
    lift = lift_map.get(req.lift_model.lower(), LiftModel.NATHAN_EXP)

    sim = BallTrajectorySimulator2(integration_method=method, lift_model=lift,
                                   fixed_cd=req.fixed_cd, fixed_cl=req.fixed_cl)

    pitch = PitchParameters(
        x0=req.pitch.x0,
        y0=req.pitch.y0,
        z0=req.pitch.z0,
        v0_mps=req.pitch.v0_mps,
        theta_deg=req.pitch.theta_deg,
        phi_deg=req.pitch.phi_deg,
        backspin_rpm=req.pitch.backspin_rpm,
        sidespin_rpm=req.pitch.sidespin_rpm,
        wg_rpm=req.pitch.wg_rpm,
        batter_hand=req.pitch.batter_hand,
    )

    env = EnvironmentParameters()
    if req.env:
        env = EnvironmentParameters(
            temp_F=req.env.temp_F,
            elev_m=req.env.elev_m,
            relative_humidity=req.env.relative_humidity,
            pressure_mmHg=req.env.pressure_mmHg,
            vwind_mph=req.env.vwind_mph,
            phiwind_deg=req.env.phiwind_deg,
            hwind_m=req.env.hwind_m,
        )

    traj = sim.simulate(pitch=pitch, env=env,
                        max_time=req.max_time, save_interval=req.save_interval)

    points = [TrajectoryPoint(t=p["t"], x=p["x"], y=p["y"], z=p["z"],
                              vx=p.get("vx"), vy=p.get("vy"), vz=p.get("vz"),
                              ax=p.get("ax"), ay=p.get("ay"), az=p.get("az"),
                              drag_x=p.get("drag_x"), drag_y=p.get("drag_y"), drag_z=p.get("drag_z"),
                              magnus_x=p.get("magnus_x"), magnus_y=p.get("magnus_y"), magnus_z=p.get("magnus_z"),
                              gravity_z=p.get("gravity_z"),
                              cd=p.get("cd"), cl=p.get("cl")) for p in traj]

    home = None
    if sim.home_plate_crossing:
        hp = sim.home_plate_crossing
        home = TrajectoryPoint(t=hp["t"], x=hp["x"], y=hp["y"], z=hp["z"],
                               vx=hp.get("vx"), vy=hp.get("vy"), vz=hp.get("vz"))

    return SimulationResponse(
        trajectory=points,
        home_plate=home,
        num_points=len(points),
    )


# ---------------------------------------------------------------------------
# HawkEye session endpoints
# ---------------------------------------------------------------------------

# In-memory session store (UUID → data). Auto-evict after TTL, max 50 sessions.
# Persistent sessions (saved to disk) are never evicted.
_hawkeye_sessions: Dict[str, dict] = {}
_HAWKEYE_MAX = 50
_HAWKEYE_TTL = 7 * 24 * 3600  # 1 week
_HAWKEYE_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "hawkeye_sessions")


def _hawkeye_load_persistent():
    """Load persistent sessions from disk at startup."""
    if not os.path.isdir(_HAWKEYE_PERSIST_DIR):
        return
    for fname in os.listdir(_HAWKEYE_PERSIST_DIR):
        if not fname.endswith(".json"):
            continue
        sid = fname[:-5]
        try:
            with open(os.path.join(_HAWKEYE_PERSIST_DIR, fname), "r") as f:
                data = json.load(f)
            data["_persistent"] = True
            _hawkeye_sessions[sid] = data
        except Exception:
            pass


def _hawkeye_save_persistent(session_id: str, data: dict):
    """Save a session to disk for persistence."""
    os.makedirs(_HAWKEYE_PERSIST_DIR, exist_ok=True)
    fpath = os.path.join(_HAWKEYE_PERSIST_DIR, f"{session_id}.json")
    with open(fpath, "w") as f:
        json.dump({k: v for k, v in data.items() if not k.startswith("_")}, f)
    data["_persistent"] = True


def _hawkeye_cleanup():
    """Remove expired sessions (persistent sessions are exempt)."""
    now = time.time()
    expired = [k for k, v in _hawkeye_sessions.items()
               if not v.get("_persistent") and now - v.get("_ts", 0) > _HAWKEYE_TTL]
    for k in expired:
        del _hawkeye_sessions[k]
    while len(_hawkeye_sessions) > _HAWKEYE_MAX:
        non_persistent = {k: v for k, v in _hawkeye_sessions.items() if not v.get("_persistent")}
        if not non_persistent:
            break
        oldest = min(non_persistent, key=lambda k: non_persistent[k].get("_ts", 0))
        del _hawkeye_sessions[oldest]


_hawkeye_load_persistent()


class HawkEyeSessionRequest(BaseModel):
    """HawkEye session data upload."""
    pitch: PitchRequest
    env: Optional[EnvRequest] = Field(default_factory=EnvRequest)
    label: str = ""
    persistent: bool = False  # Save to disk for survival across server restarts
    # Ball trajectory from HawkEye (measured)
    ball_time: Optional[List[float]] = None
    ball_pos: Optional[List[List[float]]] = None   # [[x,y,z], ...]
    # Bat trajectory from HawkEye
    bat_time: Optional[List[float]] = None
    bat_head: Optional[List[List[float]]] = None   # [[x,y,z], ...]
    bat_handle: Optional[List[List[float]]] = None  # [[x,y,z], ...]
    # ISA (Instantaneous Screw Axis)
    isa_time: Optional[List[float]] = None
    isa_pos: Optional[List[List[float]]] = None    # [[x,y,z], ...]
    isa_axis: Optional[List[List[float]]] = None   # [[wx,wy,wz], ...]
    isa_omega: Optional[List[float]] = None        # angular velocity magnitude
    grip_max_time: Optional[float] = None          # NR start time (grip speed max)
    impact_time: Optional[float] = None            # impact time
    vel_time: Optional[List[float]] = None         # time axis for velocity chart
    vel_head: Optional[List[float]] = None         # head speed (km/h)
    vel_grip: Optional[List[float]] = None         # grip speed (km/h)
    # Pitcher / Batter skeleton
    pitcher_skel: Optional[dict] = None            # {time, joints, bones}
    batter_skel: Optional[dict] = None             # {time, joints, bones}
    # Hit ball trajectory (post-impact measured)
    hit_ball_time: Optional[List[float]] = None
    hit_ball_pos: Optional[List[List[float]]] = None  # [[x,y,z], ...]
    # Release point from HawkEye events
    release_time: Optional[float] = None
    release_pos: Optional[List[float]] = None         # [x,y,z]


class HawkEyeSessionResponse(BaseModel):
    session_id: str
    url: str


@app.post("/hawkeye/session", response_model=HawkEyeSessionResponse)
def hawkeye_create_session(req: HawkEyeSessionRequest):
    """Create a HawkEye session with pitch + bat trajectory data."""
    _hawkeye_cleanup()

    session_id = str(uuid.uuid4())[:8]

    # Run simulation
    method = IntegrationMethod.RK4
    sim = BallTrajectorySimulator2(integration_method=method)
    pitch = PitchParameters(
        x0=req.pitch.x0, y0=req.pitch.y0, z0=req.pitch.z0,
        v0_mps=req.pitch.v0_mps, theta_deg=req.pitch.theta_deg,
        phi_deg=req.pitch.phi_deg, backspin_rpm=req.pitch.backspin_rpm,
        sidespin_rpm=req.pitch.sidespin_rpm, wg_rpm=req.pitch.wg_rpm,
        batter_hand=req.pitch.batter_hand,
    )
    env = EnvironmentParameters()
    if req.env:
        env = EnvironmentParameters(
            temp_F=req.env.temp_F, elev_m=req.env.elev_m,
            relative_humidity=req.env.relative_humidity,
            pressure_mmHg=req.env.pressure_mmHg,
            vwind_mph=req.env.vwind_mph, phiwind_deg=req.env.phiwind_deg,
            hwind_m=req.env.hwind_m,
        )
    traj = sim.simulate(pitch=pitch, env=env, max_time=1.0, save_interval=1)

    sim_points = [{"t": p["t"], "x": p["x"], "y": p["y"], "z": p["z"]} for p in traj]
    home = None
    if sim.home_plate_crossing:
        hp = sim.home_plate_crossing
        home = {"t": hp["t"], "x": hp["x"], "y": hp["y"], "z": hp["z"]}

    _hawkeye_sessions[session_id] = {
        "_ts": time.time(),
        "label": req.label,
        "sim_trajectory": sim_points,
        "home_plate": home,
        "ball_time": req.ball_time,
        "ball_pos": req.ball_pos,
        "bat_time": req.bat_time,
        "bat_head": req.bat_head,
        "bat_handle": req.bat_handle,
        "isa_time": req.isa_time,
        "isa_pos": req.isa_pos,
        "isa_axis": req.isa_axis,
        "isa_omega": req.isa_omega,
        "grip_max_time": req.grip_max_time,
        "impact_time": req.impact_time,
        "vel_time": req.vel_time,
        "vel_head": req.vel_head,
        "vel_grip": req.vel_grip,
        "pitcher_skel": req.pitcher_skel,
        "batter_skel": req.batter_skel,
        "hit_ball_time": req.hit_ball_time,
        "hit_ball_pos": req.hit_ball_pos,
        "release_time": req.release_time,
        "release_pos": req.release_pos,
        "pitch_params": req.pitch.model_dump(),
    }

    if req.persistent:
        _hawkeye_save_persistent(session_id, _hawkeye_sessions[session_id])

    return HawkEyeSessionResponse(
        session_id=session_id,
        url=f"/?hawkeye={session_id}",
    )


@app.get("/hawkeye/session/{session_id}")
def hawkeye_get_session(session_id: str):
    """Retrieve HawkEye session data."""
    if session_id not in _hawkeye_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    data = _hawkeye_sessions[session_id]
    # Return everything except internal timestamp
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Statcast endpoints
# ---------------------------------------------------------------------------

@app.post("/statcast/search", response_model=StatcastSearchResponse)
def statcast_search(req: StatcastSearchRequest):
    """Search for a pitcher by name.

    First tries pybaseball's Chadwick lookup table.  If that returns nothing,
    falls back to the MLB Stats API ``/people/search`` endpoint so that
    recently-debuted players (e.g. Japanese NPB transfers) can still be found.
    """
    from pybaseball import playerid_lookup
    if req.first_name:
        lookup = playerid_lookup(req.last_name, req.first_name)
    else:
        lookup = playerid_lookup(req.last_name)

    players = []
    for i, row in lookup.iterrows():
        players.append(PlayerInfo(
            index=i,
            first_name=str(row.get("name_first", "")),
            last_name=str(row.get("name_last", "")),
            mlbam_id=int(row["key_mlbam"]),
            years=f"{row.get('mlb_played_first', '?')}-{row.get('mlb_played_last', '?')}",
        ))

    # Fallback: MLB Stats API for players not yet in Chadwick table
    if not players:
        import requests as _requests
        query = f"{req.last_name} {req.first_name}" if req.first_name else req.last_name
        try:
            resp = _requests.get(
                "https://statsapi.mlb.com/api/v1/people/search",
                params={"names": query, "sportIds": "1"},
                timeout=10,
            )
            resp.raise_for_status()
            for p in resp.json().get("people", []):
                debut = p.get("mlbDebutDate", "?")[:4] if p.get("mlbDebutDate") else "?"
                last_year = p.get("lastPlayedDate", "?")[:4] if p.get("lastPlayedDate") else "?"
                players.append(PlayerInfo(
                    index=len(players),
                    first_name=p.get("firstName", ""),
                    last_name=p.get("lastName", ""),
                    mlbam_id=int(p["id"]),
                    years=f"{debut}-{last_year}",
                ))
        except Exception:
            pass  # fall through with empty list

    return StatcastSearchResponse(players=players)


class StatcastIdSearchRequest(BaseModel):
    mlbam_id: int
    year: int = 2026


@app.post("/statcast/search_by_id", response_model=StatcastSearchResponse)
def statcast_search_by_id(req: StatcastIdSearchRequest):
    """Search for a pitcher by MLBAM ID using MLB Stats API."""
    import requests as _requests
    url = f"https://statsapi.mlb.com/api/v1/people/{req.mlbam_id}"
    try:
        resp = _requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        people = data.get("people", [])
        if not people:
            return StatcastSearchResponse(players=[])
        p = people[0]
        first_name = p.get("firstName", "")
        last_name = p.get("lastName", "")
        debut = p.get("mlbDebutDate", "?")[:4] if p.get("mlbDebutDate") else "?"
        last_year = p.get("lastPlayedDate", "?")[:4] if p.get("lastPlayedDate") else "?"
        return StatcastSearchResponse(players=[PlayerInfo(
            index=0,
            first_name=first_name,
            last_name=last_name,
            mlbam_id=req.mlbam_id,
            years=f"{debut}-{last_year}",
        )])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MLB API error: {e}")


@app.post("/statcast/games", response_model=StatcastGamesResponse)
def statcast_games(req: StatcastFetchRequest):
    """Get game dates for a pitcher/year."""
    df = _get_statcast_data(req.mlbam_id, req.year)
    if df.empty:
        return StatcastGamesResponse(games=[], total_pitches=0)

    summary = (
        df.groupby("game_date")
        .agg(
            pitches=("pitch_type", "count"),
            types=("pitch_type", lambda x: ", ".join(sorted(x.dropna().unique())))
        )
        .reset_index()
        .sort_values("game_date")
    )

    games = []
    for _, row in summary.iterrows():
        games.append(GameDateInfo(
            date=row["game_date"].strftime("%Y-%m-%d"),
            pitch_count=int(row["pitches"]),
            pitch_types=row["types"],
        ))

    return StatcastGamesResponse(games=games, total_pitches=len(df))


@app.post("/statcast/pitches", response_model=StatcastPitchesResponse)
def statcast_pitches(req: StatcastPitchesRequest):
    """Get pitches for a specific game date."""
    import pandas as pd
    df = _get_statcast_data(req.mlbam_id, req.year)
    if df.empty:
        return StatcastPitchesResponse(pitches=[])

    date = pd.to_datetime(req.date)
    df_game = df[df["game_date"] == date].reset_index(drop=True)

    # Resolve batter names from MLBAM IDs
    batter_ids = df_game["batter"].dropna().unique().astype(int).tolist() if "batter" in df_game else []
    batter_names = {}
    if batter_ids:
        import requests as _requests
        # MLB Stats API supports comma-separated IDs
        try:
            ids_str = ",".join(str(bid) for bid in batter_ids[:50])  # limit to 50
            resp = _requests.get(f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}", timeout=10)
            if resp.ok:
                for p in resp.json().get("people", []):
                    batter_names[p["id"]] = p.get("fullName", "?")
        except Exception:
            pass

    pitches = []
    for i, row in df_game.iterrows():
        batter_id = _safe_int(row.get("batter"))
        pitches.append(PitchInfo(
            index=i,
            pitch_type=str(row.get("pitch_type", "?")),
            release_speed=_safe_float(row.get("release_speed")),
            release_spin_rate=_safe_float(row.get("release_spin_rate")),
            spin_axis=_safe_float(row.get("spin_axis")),
            pfx_x=_safe_float(row.get("pfx_x")),
            pfx_z=_safe_float(row.get("pfx_z")),
            effective_speed=_safe_float(row.get("effective_speed")),
            arm_angle=_safe_float(row.get("arm_angle")),
            inning=_safe_int(row.get("inning")),
            at_bat_number=_safe_int(row.get("at_bat_number")),
            pitch_number=_safe_int(row.get("pitch_number")),
            description=str(row.get("description", "")),
            events=str(row.get("events", "")) if row.get("events") is not None else None,
            plate_x=_safe_float(row.get("plate_x")),
            plate_z=_safe_float(row.get("plate_z")),
            stand=str(row.get("stand", "")) if row.get("stand") is not None else None,
            batter_id=batter_id,
            batter_name=batter_names.get(batter_id) if batter_id else None,
            balls=_safe_int(row.get("balls")),
            strikes=_safe_int(row.get("strikes")),
            outs=_safe_int(row.get("outs_when_up")),
            launch_speed=_safe_float(row.get("launch_speed")),
            launch_angle=_safe_float(row.get("launch_angle")),
        ))

    return StatcastPitchesResponse(pitches=pitches)


class SeasonSummaryRequest(BaseModel):
    mlbam_id: int
    year: int = 2025
    include_bsg: bool = False
    include_monthly: bool = True


class PitchTypeSummary(BaseModel):
    pitch_type: str
    count: int
    avg_speed_mph: Optional[float] = None
    avg_spin_rate: Optional[float] = None
    avg_spin_axis: Optional[float] = None
    avg_pfx_x_in: Optional[float] = None
    avg_pfx_z_in: Optional[float] = None
    avg_ivb_in: Optional[float] = None      # induced vertical break
    avg_hb_in: Optional[float] = None       # horizontal break (arm-side)
    avg_arm_angle: Optional[float] = None
    avg_effective_speed: Optional[float] = None
    # Plate discipline
    whiff_pct: Optional[float] = None       # swinging strikes / swings
    chase_pct: Optional[float] = None       # swings at pitches outside zone / pitches outside zone
    zone_pct: Optional[float] = None        # pitches in zone / total pitches
    csw_pct: Optional[float] = None         # (called strikes + whiffs) / total pitches
    # Batted ball (when contact made)
    avg_launch_speed: Optional[float] = None
    avg_launch_angle: Optional[float] = None
    avg_xwoba: Optional[float] = None       # estimated_woba_using_speedangle
    avg_woba: Optional[float] = None
    # BSG (opt-in)
    avg_spin_efficiency: Optional[float] = None
    avg_backspin_rpm: Optional[float] = None
    avg_sidespin_rpm: Optional[float] = None
    avg_gyrospin_rpm: Optional[float] = None


class MonthlyTrend(BaseModel):
    month: str
    pitch_type: str
    count: int
    avg_speed_mph: Optional[float] = None
    avg_spin_rate: Optional[float] = None
    avg_spin_axis: Optional[float] = None
    avg_pfx_x_in: Optional[float] = None
    avg_pfx_z_in: Optional[float] = None


class SeasonSummaryResponse(BaseModel):
    mlbam_id: int
    year: int
    total_pitches: int
    pitch_type_summaries: list[PitchTypeSummary]
    monthly_trends: list[MonthlyTrend] = []
    bsg_computed: bool = False


@app.post("/statcast/season_summary", response_model=SeasonSummaryResponse)
def statcast_season_summary(req: SeasonSummaryRequest):
    """Season-wide pitch-type summary with optional BSG decomposition and monthly trends."""
    import pandas as pd
    import numpy as np

    df = _get_statcast_data(req.mlbam_id, req.year)
    if df.empty:
        raise HTTPException(status_code=404, detail="No Statcast data found")

    # Circular mean helper for spin_axis
    def circular_mean(angles_deg):
        valid = angles_deg.dropna()
        if len(valid) == 0:
            return None
        rads = np.radians(valid.values.astype(float))
        return float(np.degrees(np.arctan2(np.sin(rads).mean(), np.cos(rads).mean())) % 360)

    # --- BSG (opt-in) ---
    bsg_df = None
    if req.include_bsg:
        bsg_df = vectorized_bsg_summary(df)

    # --- Plate discipline helpers ---
    def _plate_discipline(g):
        """Compute whiff%, chase%, zone%, csw% from description column."""
        if "description" not in g or "zone" not in g:
            return {}
        desc = g["description"].dropna().str.lower()
        total = len(desc)
        if total == 0:
            return {}

        swings = desc.isin(["swinging_strike", "swinging_strike_blocked",
                            "foul", "foul_tip", "foul_bunt", "hit_into_play",
                            "hit_into_play_no_out", "hit_into_play_score",
                            "missed_bunt"])
        whiffs = desc.isin(["swinging_strike", "swinging_strike_blocked"])
        called_strikes = desc.isin(["called_strike"])

        n_swings = swings.sum()
        n_whiffs = whiffs.sum()
        n_called = called_strikes.sum()

        # Zone: zone 1-9 = in zone, others = out
        zone_vals = pd.to_numeric(g["zone"], errors="coerce")
        in_zone = (zone_vals >= 1) & (zone_vals <= 9)
        out_zone = zone_vals.notna() & ~in_zone
        n_in_zone = in_zone.sum()
        n_out_zone = out_zone.sum()

        # Swings at pitches outside zone
        swings_outside = (swings & out_zone).sum()

        result = {}
        result["whiff_pct"] = round(n_whiffs / n_swings * 100, 1) if n_swings > 0 else None
        result["chase_pct"] = round(swings_outside / n_out_zone * 100, 1) if n_out_zone > 0 else None
        result["zone_pct"] = round(n_in_zone / total * 100, 1) if total > 0 else None
        result["csw_pct"] = round((n_called + n_whiffs) / total * 100, 1) if total > 0 else None
        return result

    # --- Per pitch-type summary ---
    summaries = []
    for pt, g in df.groupby("pitch_type"):
        disc = _plate_discipline(g)
        s = PitchTypeSummary(
            pitch_type=str(pt),
            count=len(g),
            avg_speed_mph=_safe_round(g["release_speed"].mean(), 1) if "release_speed" in g else None,
            avg_spin_rate=_safe_round(g["release_spin_rate"].mean(), 0) if "release_spin_rate" in g else None,
            avg_spin_axis=_safe_round(circular_mean(g["spin_axis"]), 1) if "spin_axis" in g else None,
            avg_pfx_x_in=_safe_round(g["pfx_x"].mean() * 12, 1) if "pfx_x" in g else None,
            avg_pfx_z_in=_safe_round(g["pfx_z"].mean() * 12, 1) if "pfx_z" in g else None,
            avg_ivb_in=_safe_round(g["api_break_z_with_gravity"].mean(), 1) if "api_break_z_with_gravity" in g else None,
            avg_hb_in=_safe_round(g["api_break_x_arm"].mean(), 1) if "api_break_x_arm" in g else None,
            avg_arm_angle=_safe_round(g["arm_angle"].mean(), 1) if "arm_angle" in g else None,
            avg_effective_speed=_safe_round(g["effective_speed"].mean(), 1) if "effective_speed" in g else None,
            whiff_pct=disc.get("whiff_pct"),
            chase_pct=disc.get("chase_pct"),
            zone_pct=disc.get("zone_pct"),
            csw_pct=disc.get("csw_pct"),
            avg_launch_speed=_safe_round(g["launch_speed"].mean(), 1) if "launch_speed" in g else None,
            avg_launch_angle=_safe_round(g["launch_angle"].mean(), 1) if "launch_angle" in g else None,
            avg_xwoba=_safe_round(g["estimated_woba_using_speedangle"].mean(), 3) if "estimated_woba_using_speedangle" in g else None,
            avg_woba=_safe_round(g["woba_value"].mean(), 3) if "woba_value" in g else None,
        )
        if bsg_df is not None:
            bsg_g = bsg_df.loc[g.index]
            s.avg_spin_efficiency = _safe_round(bsg_g["spin_efficiency"].mean(), 3)
            s.avg_backspin_rpm = _safe_round(bsg_g["backspin_rpm"].mean(), 0)
            s.avg_sidespin_rpm = _safe_round(bsg_g["sidespin_rpm"].mean(), 0)
            s.avg_gyrospin_rpm = _safe_round(bsg_g["wg_rpm"].mean(), 0)
        summaries.append(s)

    # --- Monthly trends (opt-in) ---
    trends = []
    if req.include_monthly:
        df["_month"] = df["game_date"].dt.strftime("%Y-%m")
        for (month, pt), g in df.groupby(["_month", "pitch_type"]):
            trends.append(MonthlyTrend(
                month=str(month),
                pitch_type=str(pt),
                count=len(g),
                avg_speed_mph=_safe_round(g["release_speed"].mean(), 1) if "release_speed" in g else None,
                avg_spin_rate=_safe_round(g["release_spin_rate"].mean(), 0) if "release_spin_rate" in g else None,
                avg_spin_axis=_safe_round(circular_mean(g["spin_axis"]), 1) if "spin_axis" in g else None,
                avg_pfx_x_in=_safe_round(g["pfx_x"].mean() * 12, 1) if "pfx_x" in g else None,
                avg_pfx_z_in=_safe_round(g["pfx_z"].mean() * 12, 1) if "pfx_z" in g else None,
            ))
        df.drop(columns=["_month"], inplace=True)

    return SeasonSummaryResponse(
        mlbam_id=req.mlbam_id,
        year=req.year,
        total_pitches=len(df),
        pitch_type_summaries=summaries,
        monthly_trends=trends,
        bsg_computed=req.include_bsg,
    )


@app.post("/statcast/compare", response_model=CompareResponse)
def statcast_compare(req: CompareRequest):
    """Compare multiple pitches side by side. Max 6 pitches per request."""
    import pandas as pd
    import requests as _requests

    if len(req.pitches) > 6:
        raise HTTPException(status_code=400, detail="Max 6 pitches per comparison")

    FT_TO_M = 0.3048
    results = []

    # Resolve pitcher names
    pitcher_names = {}
    unique_ids = set(p.mlbam_id for p in req.pitches)
    for pid in unique_ids:
        try:
            resp = _requests.get(f"https://statsapi.mlb.com/api/v1/people/{pid}", timeout=5)
            if resp.ok:
                people = resp.json().get("people", [])
                if people:
                    pitcher_names[pid] = people[0].get("fullName", str(pid))
        except Exception:
            pitcher_names[pid] = str(pid)

    for item in req.pitches:
        try:
            df = _get_statcast_data(item.mlbam_id, item.year)
            if df.empty:
                continue

            date = pd.to_datetime(item.date)
            df_game = df[df["game_date"] == date].reset_index(drop=True)
            if item.pitch_index < 0 or item.pitch_index >= len(df_game):
                continue

            row = df_game.iloc[item.pitch_index]
            pitch_data = {}
            for k, v in row.to_dict().items():
                pitch_data[k] = v if pd.notna(v) else None

            if req.cl_mode == "nathan_exp":
                _lift_c = LiftModel.NATHAN_EXP
                _cl2_c = None
            elif req.cl_mode == "rational":
                _lift_c = LiftModel.RATIONAL
                _cl2_c = None
            elif req.cl_mode == "nathan":
                _lift_c = LiftModel.RATIONAL
                _cl2_c = BallTrajectorySimulator2.NATHAN_CL2
            else:
                _lift_c = LiftModel.RATIONAL
                _cl2_c = None

            sim_params = statcast_to_sim_params(pitch_data, spin_method=req.spin_method,
                                                lift_model=_lift_c.value,
                                                accel_method=req.accel_method)
            batter_hand = str(row.get("stand", "R")) if pd.notna(row.get("stand")) else "R"

            sim = BallTrajectorySimulator2(integration_method=IntegrationMethod.RK4,
                                           lift_model=_lift_c, cl2=_cl2_c)

            if req.spin_method == "direct" and "wx_rad_s" in sim_params:
                pitch = PitchParameters(
                    x0=sim_params["x0"], y0=sim_params["y0"], z0=sim_params["z0"],
                    v0_mps=sim_params["v0_mps"],
                    theta_deg=sim_params["theta_deg"], phi_deg=sim_params["phi_deg"],
                    batter_hand=batter_hand,
                    wx_rad_s=sim_params["wx_rad_s"], wy_rad_s=sim_params["wy_rad_s"],
                    wz_rad_s=sim_params["wz_rad_s"],
                )
            else:
                pitch = PitchParameters(
                    x0=sim_params["x0"], y0=sim_params["y0"], z0=sim_params["z0"],
                    v0_mps=sim_params["v0_mps"],
                    theta_deg=sim_params["theta_deg"], phi_deg=sim_params["phi_deg"],
                    backspin_rpm=sim_params["backspin_rpm"], sidespin_rpm=sim_params["sidespin_rpm"],
                    wg_rpm=sim_params["wg_rpm"], batter_hand=batter_hand,
                )

            traj = sim.simulate(pitch=pitch, env=EnvironmentParameters(), max_time=1.0)

            home_sim = None
            if sim.home_plate_crossing:
                hp = sim.home_plate_crossing
                home_sim = {"x": hp["x"], "z": hp["z"]}

            plate_x = _safe_float(row.get("plate_x"))
            plate_z = _safe_float(row.get("plate_z"))
            home_sc = None
            error_mm = None
            if plate_x is not None and plate_z is not None:
                home_sc = {"x": plate_x * FT_TO_M, "z": plate_z * FT_TO_M}
                if home_sim:
                    dx = (home_sim["x"] - home_sc["x"]) * 1000
                    dz = (home_sim["z"] - home_sc["z"]) * 1000
                    error_mm = round(math.sqrt(dx**2 + dz**2), 1)

            pfx_x = _safe_float(row.get("pfx_x"))
            pfx_z = _safe_float(row.get("pfx_z"))

            sim_url = f"https://baseball.skill-vis.com/?mlbam_id={item.mlbam_id}&year={item.year}&date={item.date}&pitch={item.pitch_index}"

            results.append(CompareResultItem(
                label=item.label or f"{pitcher_names.get(item.mlbam_id, '?')} {str(pitch_data.get('pitch_type', '?'))}",
                pitcher_name=pitcher_names.get(item.mlbam_id),
                pitch_type=str(pitch_data.get("pitch_type", "?")),
                release_speed_mph=_safe_float(pitch_data.get("release_speed")),
                release_spin_rate=_safe_float(pitch_data.get("release_spin_rate")),
                spin_axis=_safe_float(pitch_data.get("spin_axis")),
                pfx_x_in=round(pfx_x * 12, 1) if pfx_x else None,
                pfx_z_in=round(pfx_z * 12, 1) if pfx_z else None,
                spin_efficiency=sim_params.get("spin_efficiency"),
                backspin_rpm=sim_params.get("backspin_rpm"),
                sidespin_rpm=sim_params.get("sidespin_rpm"),
                wg_rpm=sim_params.get("wg_rpm"),
                home_plate_sim=home_sim,
                home_plate_statcast=home_sc,
                error_mm=error_mm,
                sim_url=sim_url,
            ))
        except Exception:
            continue

    return CompareResponse(results=results, count=len(results))


@app.post("/statcast/simulate", response_model=StatcastSimResponse)
def statcast_simulate(req: StatcastSimRequest):
    """Select a Statcast pitch, convert to sim params, and run trajectory."""
    import pandas as pd
    df = _get_statcast_data(req.mlbam_id, req.year)
    if df.empty:
        raise HTTPException(status_code=404, detail="No Statcast data found")

    date = pd.to_datetime(req.date)
    df_game = df[df["game_date"] == date].reset_index(drop=True)

    if req.pitch_index < 0 or req.pitch_index >= len(df_game):
        raise HTTPException(status_code=400, detail=f"Invalid pitch index: {req.pitch_index}")

    row = df_game.iloc[req.pitch_index]
    pitch_data = {}
    for k, v in row.to_dict().items():
        if pd.notna(v):
            pitch_data[k] = v
        else:
            pitch_data[k] = None

    # Resolve lift model from cl_mode
    if req.cl_mode == "nathan_exp":
        _lift = LiftModel.NATHAN_EXP
        _cl2 = None
    elif req.cl_mode == "rational":
        _lift = LiftModel.RATIONAL
        _cl2 = None  # uses default ADJUSTED_CL2
    elif req.cl_mode == "nathan":
        _lift = LiftModel.RATIONAL
        _cl2 = BallTrajectorySimulator2.NATHAN_CL2
    else:  # "adjusted" (legacy)
        _lift = LiftModel.RATIONAL
        _cl2 = None

    # Convert to sim params
    sim_params = statcast_to_sim_params(pitch_data, spin_method=req.spin_method,
                                       lift_model=_lift.value,
                                       accel_method=req.accel_method)

    # Batter handedness
    batter_hand = str(row.get("stand", "R")) if pd.notna(row.get("stand")) else "R"

    # Run simulation
    sim = BallTrajectorySimulator2(integration_method=IntegrationMethod.RK4,
                                   lift_model=_lift, cl2=_cl2)
    if req.spin_method == "direct" and "wx_rad_s" in sim_params:
        pitch = PitchParameters(
            x0=sim_params["x0"],
            y0=sim_params["y0"],
            z0=sim_params["z0"],
            v0_mps=sim_params["v0_mps"],
            theta_deg=sim_params["theta_deg"],
            phi_deg=sim_params["phi_deg"],
            batter_hand=batter_hand,
            wx_rad_s=sim_params["wx_rad_s"],
            wy_rad_s=sim_params["wy_rad_s"],
            wz_rad_s=sim_params["wz_rad_s"],
        )
    else:
        pitch = PitchParameters(
            x0=sim_params["x0"],
            y0=sim_params["y0"],
            z0=sim_params["z0"],
            v0_mps=sim_params["v0_mps"],
            theta_deg=sim_params["theta_deg"],
            phi_deg=sim_params["phi_deg"],
            backspin_rpm=sim_params["backspin_rpm"],
            sidespin_rpm=sim_params["sidespin_rpm"],
            wg_rpm=sim_params["wg_rpm"],
            batter_hand=batter_hand,
        )
    traj = sim.simulate(pitch=pitch, env=EnvironmentParameters(), max_time=1.0, save_interval=1)

    points = [TrajectoryPoint(t=p["t"], x=p["x"], y=p["y"], z=p["z"],
                              vx=p.get("vx"), vy=p.get("vy"), vz=p.get("vz"),
                              ax=p.get("ax"), ay=p.get("ay"), az=p.get("az"),
                              drag_x=p.get("drag_x"), drag_y=p.get("drag_y"), drag_z=p.get("drag_z"),
                              magnus_x=p.get("magnus_x"), magnus_y=p.get("magnus_y"), magnus_z=p.get("magnus_z"),
                              gravity_z=p.get("gravity_z"),
                              cd=p.get("cd"), cl=p.get("cl")) for p in traj]

    home = None
    if sim.home_plate_crossing:
        hp = sim.home_plate_crossing
        home = TrajectoryPoint(t=hp["t"], x=hp["x"], y=hp["y"], z=hp["z"],
                               vx=hp.get("vx"), vy=hp.get("vy"), vz=hp.get("vz"))

    # No-spin simulation (same release but zero spin)
    sim_nospin = BallTrajectorySimulator2(integration_method=IntegrationMethod.RK4,
                                         lift_model=_lift, cl2=_cl2)
    pitch_nospin = PitchParameters(
        x0=sim_params["x0"],
        y0=sim_params["y0"],
        z0=sim_params["z0"],
        v0_mps=sim_params["v0_mps"],
        theta_deg=sim_params["theta_deg"],
        phi_deg=sim_params["phi_deg"],
        backspin_rpm=0.0,
        sidespin_rpm=0.0,
        wg_rpm=0.0,
        batter_hand=batter_hand,
    )
    traj_nospin = sim_nospin.simulate(pitch=pitch_nospin, env=EnvironmentParameters(),
                                      max_time=1.0, save_interval=1)
    nospin_points = [TrajectoryPoint(t=p["t"], x=p["x"], y=p["y"], z=p["z"],
                                    vx=p.get("vx"), vy=p.get("vy"), vz=p.get("vz"),
                                    ax=p.get("ax"), ay=p.get("ay"), az=p.get("az"),
                                    drag_x=p.get("drag_x"), drag_y=p.get("drag_y"), drag_z=p.get("drag_z"),
                                    magnus_x=p.get("magnus_x"), magnus_y=p.get("magnus_y"), magnus_z=p.get("magnus_z"),
                                    gravity_z=p.get("gravity_z"),
                                    cd=p.get("cd"), cl=p.get("cl"))
                     for p in traj_nospin]
    home_nospin = None
    if sim_nospin.home_plate_crossing:
        hp_ns = sim_nospin.home_plate_crossing
        home_nospin = TrajectoryPoint(t=hp_ns["t"], x=hp_ns["x"], y=hp_ns["y"], z=hp_ns["z"],
                                      vx=hp_ns.get("vx"), vy=hp_ns.get("vy"), vz=hp_ns.get("vz"))

    # Statcast home plate crossing (plate_x, plate_z are in feet)
    FT_TO_M = 0.3048
    home_statcast = None
    plate_x = _safe_float(row.get("plate_x"))
    plate_z = _safe_float(row.get("plate_z"))
    if plate_x is not None and plate_z is not None:
        home_statcast = {
            "x": plate_x * FT_TO_M,
            "y": 0.0,
            "z": plate_z * FT_TO_M,
            "x_ft": plate_x,
            "z_ft": plate_z,
        }

    # Statcast pfx (spin-induced movement — pybaseball returns feet)
    pfx_x = _safe_float(row.get("pfx_x"))
    pfx_z = _safe_float(row.get("pfx_z"))
    statcast_pfx = None
    if pfx_x is not None and pfx_z is not None:
        statcast_pfx = {
            "pfx_x_ft": pfx_x,
            "pfx_z_ft": pfx_z,
            "pfx_x_in": pfx_x * 12.0,
            "pfx_z_in": pfx_z * 12.0,
            "pfx_x_m": pfx_x * FT_TO_M,
            "pfx_z_m": pfx_z * FT_TO_M,
        }

    # Release point
    release_point = {
        "x": sim_params["x0"],
        "y": sim_params["y0"],
        "z": sim_params["z0"],
    }

    # Pitch info for display
    pitch_info = {
        "pitch_type": str(pitch_data.get("pitch_type", "?")),
        "release_speed_mph": _safe_float(pitch_data.get("release_speed")),
        "release_spin_rate": _safe_float(pitch_data.get("release_spin_rate")),
        "spin_axis": _safe_float(pitch_data.get("spin_axis")),
        "inning": _safe_int(pitch_data.get("inning")),
        "description": str(pitch_data.get("description", "")),
        "batter_hand": batter_hand,
        "p_throws": str(pitch_data.get("p_throws", "R")),
        "pitch_number": _safe_int(pitch_data.get("pitch_number")),
        "at_bat_number": _safe_int(pitch_data.get("at_bat_number")),
        "sz_top_ft": _safe_float(pitch_data.get("sz_top")),
        "sz_bot_ft": _safe_float(pitch_data.get("sz_bot")),
    }

    return StatcastSimResponse(
        trajectory=points,
        trajectory_nospin=nospin_points,
        home_plate=home,
        home_plate_nospin=home_nospin,
        home_plate_statcast=home_statcast,
        statcast_pfx=statcast_pfx,
        release_point=release_point,
        pitch_info=pitch_info,
        sim_params={
            "x0": sim_params["x0"],
            "y0": sim_params["y0"],
            "z0": sim_params["z0"],
            "v0_mps": sim_params["v0_mps"],
            "theta_deg": sim_params["theta_deg"],
            "phi_deg": sim_params["phi_deg"],
            "backspin_rpm": sim_params["backspin_rpm"],
            "sidespin_rpm": sim_params["sidespin_rpm"],
            "wg_rpm": sim_params["wg_rpm"],
            "release_speed_mph": sim_params.get("release_speed_mph"),
            "spin_efficiency": sim_params.get("spin_efficiency"),
            "theta_eff_deg": sim_params.get("theta_eff_deg"),
            "spin_method": req.spin_method,
            "cl_mode": req.cl_mode,
            "cl2": sim.cl2,
            "wx_rad_s": sim_params.get("wx_rad_s"),
            "wy_rad_s": sim_params.get("wy_rad_s"),
            "wz_rad_s": sim_params.get("wz_rad_s"),
        },
        num_points=len(points),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_round(v, ndigits: int = 1) -> Optional[float]:
    """Round a value safely, returning None for NaN/None."""
    if v is None:
        return None
    try:
        val = float(v)
        return None if math.isnan(val) else round(val, ndigits)
    except (TypeError, ValueError):
        return None


def _safe_float(v) -> Optional[float]:
    import pandas as pd
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        val = float(v)
        return None if math.isnan(val) else val
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    import pandas as pd
    if v is None:
        return None
    try:
        val = float(v)
        return None if math.isnan(val) else int(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# WebM -> MP4 conversion endpoint
# ---------------------------------------------------------------------------
@app.post("/convert-to-mp4")
async def convert_to_mp4(file: UploadFile = File(...)):
    """Convert uploaded WebM video to MP4 (H.264) using ffmpeg."""
    import logging
    logger = logging.getLogger("convert-to-mp4")

    tmp_dir = tempfile.mkdtemp()
    uid = uuid.uuid4().hex[:8]
    webm_path = os.path.join(tmp_dir, f"{uid}.webm")
    mp4_path = os.path.join(tmp_dir, f"{uid}.mp4")

    try:
        # Save uploaded WebM
        content = await file.read()
        logger.warning(f"Received file: name={file.filename}, size={len(content)} bytes")
        if len(content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        with open(webm_path, "wb") as f:
            f.write(content)

        # Convert with ffmpeg (pad to even dimensions for yuv420p)
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", webm_path,
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                mp4_path,
            ],
            capture_output=True, text=True, timeout=120,
        )

        logger.warning(f"ffmpeg returncode={result.returncode}, "
                       f"mp4 exists={os.path.exists(mp4_path)}, "
                       f"stderr_tail={result.stderr[-300:] if result.stderr else 'none'}")

        if result.returncode != 0 or not os.path.exists(mp4_path):
            raise HTTPException(
                status_code=500,
                detail=f"ffmpeg error (rc={result.returncode}): {result.stderr[-500:] if result.stderr else 'no stderr'}")

        mp4_size = os.path.getsize(mp4_path)
        logger.warning(f"MP4 ready: {mp4_size} bytes at {mp4_path}")

        # Log the download event
        try:
            events = _load_events()
            events.append({
                "event": "video_download",
                "time": datetime.now().isoformat(),
                "ip": "server",
                "detail": f"{file.filename}, {len(content)}B→{mp4_size}B",
            })
            _save_events(events)
        except Exception:
            pass

        # Read MP4 into memory before cleanup
        with open(mp4_path, "rb") as f:
            mp4_data = f.read()

        # Clean up temp files
        os.unlink(webm_path)
        os.unlink(mp4_path)
        os.rmdir(tmp_dir)

        from fastapi.responses import Response
        out_filename = file.filename.replace(".webm", ".mp4") if file.filename else "trajectory.mp4"
        return Response(
            content=mp4_data,
            media_type="video/mp4",
            headers={"Content-Disposition": f'attachment; filename="{out_filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Exception: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Shaft deflection analysis
# ---------------------------------------------------------------------------
from shaft_analysis import analyze as shaft_analyze, generate_mp4 as shaft_generate_mp4

@app.post("/shaft/analyze")
async def shaft_deflection_analyze(
    calib: UploadFile = File(...),
    swing: UploadFile = File(...),
    defl_scale: float = Form(5.0),
    frame_step: int = Form(2),
):
    """Analyze shaft deflection from calibration and swing CSV files."""
    try:
        calib_text = (await calib.read()).decode('utf-8')
        swing_text = (await swing.read()).decode('utf-8')
        result = shaft_analyze(calib_text, swing_text,
                               defl_scale=defl_scale, frame_step=frame_step)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/shaft/mp4")
async def shaft_deflection_mp4(
    calib: UploadFile = File(...),
    swing: UploadFile = File(...),
    defl_scale: float = Form(5.0),
    frame_step: int = Form(2),
    cam_x: float = Form(-2.0),
    cam_y: float = Form(0.0),
    cam_z: float = Form(0.8),
):
    """Generate MP4 animation of shaft deflection."""
    try:
        calib_text = (await calib.read()).decode('utf-8')
        swing_text = (await swing.read()).decode('utf-8')
        out_path = f"/tmp/shaft_{uuid.uuid4().hex[:8]}.mp4"
        shaft_generate_mp4(calib_text, swing_text,
                           defl_scale=defl_scale, frame_step=frame_step,
                           output_path=out_path,
                           cam_eye=(cam_x, cam_y, cam_z))
        return FileResponse(
            out_path, media_type="video/mp4",
            headers={"Content-Disposition": 'attachment; filename="shaft_deflection.mp4"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Mocap Session API  (JSON file-based storage)
# ---------------------------------------------------------------------------
import hashlib

_MOCAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")
_MOCAP_INDEX = os.path.join(_MOCAP_DIR, "index.json")
# API Key: set via environment variable BALL_API_KEY (required for write operations)
_MOCAP_API_KEY = os.environ.get("BALL_API_KEY", "")


def _mocap_verify_key(request: Request):
    """Verify admin API key from X-API-Key header."""
    key = request.headers.get("X-API-Key", "")
    if not _MOCAP_API_KEY:
        raise HTTPException(status_code=500, detail="BALL_API_KEY not configured on server")
    if not key or key != _MOCAP_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


def _mocap_load_index() -> list:
    """Load session index. Returns [] if not found."""
    if not os.path.exists(_MOCAP_INDEX):
        return []
    with open(_MOCAP_INDEX, "r", encoding="utf-8") as f:
        return json.load(f)


def _mocap_save_index(index: list):
    """Save session index."""
    os.makedirs(_MOCAP_DIR, exist_ok=True)
    with open(_MOCAP_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


@app.post("/mocap/session")
async def mocap_create_session(request: Request):
    """Upload a mocap session (admin only)."""
    _mocap_verify_key(request)

    data = await request.json()
    session_id = str(uuid.uuid4())[:8]

    # Save full session data
    os.makedirs(_MOCAP_DIR, exist_ok=True)
    session_path = os.path.join(_MOCAP_DIR, f"{session_id}.json")
    data["id"] = session_id
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    # Update index
    index = _mocap_load_index()
    index.append({
        "id": session_id,
        "player_name": data.get("player_name", ""),
        "pitch_type": data.get("pitch_type", ""),
        "visibility": data.get("visibility", "private"),
        "release_speed_kmh": data.get("summary", {}).get("release_speed_kmh"),
        "spin_rate_rpm": data.get("summary", {}).get("spin_rate_rpm"),
        "created_at": data.get("created_at", ""),
    })
    _mocap_save_index(index)

    return {"session_id": session_id, "url": f"/?mocap={session_id}"}


@app.get("/mocap/sessions")
def mocap_list_sessions():
    """List public/limited sessions."""
    index = _mocap_load_index()
    return [s for s in index if s.get("visibility") in ("public", "limited")]


@app.get("/mocap/session/{session_id}")
def mocap_get_session(session_id: str):
    """Get session data (public/limited only, or all for admin)."""
    session_path = os.path.join(_MOCAP_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")

    with open(session_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("visibility") == "private":
        raise HTTPException(status_code=404, detail="Session not found")

    return data


@app.get("/mocap/session/{session_id}/admin")
async def mocap_get_session_admin(session_id: str, request: Request):
    """Get any session data (admin only, including private)."""
    _mocap_verify_key(request)
    session_path = os.path.join(_MOCAP_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")
    with open(session_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.delete("/mocap/session/{session_id}")
async def mocap_delete_session(session_id: str, request: Request):
    """Delete a session (admin only)."""
    _mocap_verify_key(request)

    session_path = os.path.join(_MOCAP_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")

    os.remove(session_path)

    # Update index
    index = _mocap_load_index()
    index = [s for s in index if s.get("id") != session_id]
    _mocap_save_index(index)

    return {"deleted": session_id}


@app.post("/mocap/session/{session_id}/simulate")
def mocap_simulate_from_session(session_id: str):
    """Run spin + no-spin trajectory simulation. Returns trajectory with Magnus force components."""
    session_path = os.path.join(_MOCAP_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")

    with open(session_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sp = data.get("sim_params", {})
    if "error" in sp or "v0_mps" not in sp:
        raise HTTPException(status_code=400, detail="Session has no valid sim_params")

    env = EnvironmentParameters()

    # Spin trajectory (with Magnus force components)
    pitch_spin = PitchParameters(
        x0=sp.get("release_x_m", -0.47),
        y0=sp.get("release_y_m", 16.48),
        z0=sp.get("release_z_m", 1.50),
        v0_mps=sp["v0_mps"],
        theta_deg=sp["theta_deg"],
        phi_deg=sp["phi_deg"],
        backspin_rpm=sp["backspin_rpm"],
        sidespin_rpm=sp["sidespin_rpm"],
        wg_rpm=sp.get("gyrospin_rpm", 0),
    )
    # Note: session's wx/wy/wz_rad_s are in mocap coordinates, not Nathan.
    # BSG (backspin/sidespin/gyrospin) are already in Nathan frame, so use BSG mode.

    sim_spin = BallTrajectorySimulator2(
        integration_method=IntegrationMethod.RK4,
        lift_model=LiftModel.NATHAN_EXP,
    )
    # home_plate_y=0: simulate until rear edge of home plate
    traj_spin = sim_spin.simulate(pitch=pitch_spin, env=env, max_time=1.0, save_interval=1, home_plate_y=0.0)

    spin_points = [{
        "t": p["t"], "x": p["x"], "y": p["y"], "z": p["z"],
        "vx": p.get("vx", 0), "vy": p.get("vy", 0), "vz": p.get("vz", 0),
        "magnus_x": p.get("magnus_x", 0), "magnus_y": p.get("magnus_y", 0), "magnus_z": p.get("magnus_z", 0),
    } for p in traj_spin]

    home = None
    if sim_spin.home_plate_crossing:
        hp = sim_spin.home_plate_crossing
        home = {"t": hp["t"], "x": hp["x"], "y": hp["y"], "z": hp["z"]}

    # No-spin trajectory (same release, zero spin)
    pitch_nospin = PitchParameters(
        x0=sp.get("release_x_m", -0.47),
        y0=sp.get("release_y_m", 16.48),
        z0=sp.get("release_z_m", 1.50),
        v0_mps=sp["v0_mps"],
        theta_deg=sp["theta_deg"],
        phi_deg=sp["phi_deg"],
        backspin_rpm=0, sidespin_rpm=0, wg_rpm=0,
    )
    sim_nospin = BallTrajectorySimulator2(
        integration_method=IntegrationMethod.RK4,
        lift_model=LiftModel.NATHAN_EXP,
    )
    traj_nospin = sim_nospin.simulate(pitch=pitch_nospin, env=env, max_time=1.0, save_interval=1, home_plate_y=0.0)

    nospin_points = [{"t": p["t"], "x": p["x"], "y": p["y"], "z": p["z"]} for p in traj_nospin]

    nospin_home = None
    if sim_nospin.home_plate_crossing:
        hp = sim_nospin.home_plate_crossing
        nospin_home = {"t": hp["t"], "x": hp["x"], "y": hp["y"], "z": hp["z"]}

    return {
        "session_id": session_id,
        "trajectory": spin_points,
        "home_plate": home,
        "nospin_trajectory": nospin_points,
        "nospin_home_plate": nospin_home,
    }


# ---------------------------------------------------------------------------
# Pitch sequence analysis endpoints
# ---------------------------------------------------------------------------

class SequenceAnalysisRequest(BaseModel):
    mlbam_id: int
    year: int = 2025
    date: str                           # YYYY-MM-DD
    analyze: bool = True                # run simulation + metrics


class SequenceAnalysisResponse(BaseModel):
    pitcher_name: str
    game_date: str
    total_at_bats: int
    total_pitches: int
    at_bats: list                       # serialized AtBat objects


@app.post("/sequence/analyze", response_model=SequenceAnalysisResponse)
def sequence_analyze(req: SequenceAnalysisRequest):
    """Full pitch-sequence analysis for one game."""
    from pitch_sequence.queries import SequenceQueryEngine

    engine = SequenceQueryEngine()
    at_bats = engine.load_statcast_game(
        mlbam_id=req.mlbam_id,
        year=req.year,
        date=req.date,
        analyze=req.analyze,
    )

    def _serialize_at_bat(ab):
        return {
            "at_bat_id": ab.at_bat_id,
            "at_bat_number": ab.at_bat_number,
            "batter_name": ab.batter_name,
            "stand": ab.stand,
            "inning": ab.inning,
            "result": ab.result.value if ab.result else None,
            "result_category": ab.result_category.value if ab.result_category else None,
            "pitches": [
                {
                    "pitch_number": p.pitch_number_in_ab,
                    "pitch_type": p.pitch_type,
                    "balls": p.balls,
                    "strikes": p.strikes,
                    "plate_x_m": p.plate_x,
                    "plate_z_m": p.plate_z,
                    "release_speed_mps": p.release_speed_mps,
                    "description": p.description.value if p.description else None,
                    "is_whiff": p.is_whiff,
                    "arrival_time_ms": (p.sim_result.arrival_time_s * 1000
                                        if p.sim_result else None),
                    "arrival_speed_mps": (p.sim_result.arrival_speed_mps
                                          if p.sim_result else None),
                }
                for p in ab.pitches
            ],
            "metrics": _serialize_metrics(ab.sequence_metrics) if ab.sequence_metrics else None,
        }

    def _serialize_metrics(sm):
        return {
            "tempo_differentials": [
                {
                    "pair": f"{td.pitch_a_type}->{td.pitch_b_type}",
                    "differential_ms": round(td.differential_ms, 1),
                }
                for td in sm.tempo_differentials
            ],
            "tunnel_analyses": [
                {
                    "pair": f"{t.pitch_a_type}->{t.pitch_b_type}",
                    "tunnel_distance_m": round(t.tunnel_distance_m, 3),
                    "tunnel_time_ms": round(t.tunnel_time_s * 1000, 1),
                    "plate_separation_cm": round(t.plate_separation_m * 100, 1),
                }
                for t in sm.tunnel_analyses
            ],
            "movement_vectors": [
                {
                    "dx_cm": round(mv.dx_m * 100, 1),
                    "dz_cm": round(mv.dz_m * 100, 1),
                    "magnitude_cm": round(mv.magnitude_m * 100, 1),
                }
                for mv in sm.movement_vectors
            ],
            "reaction_mismatches": [
                {
                    "mismatch_ms": round(rm.timing_mismatch_ms, 1),
                    "direction": rm.mismatch_direction,
                    "pitch_b_result": rm.pitch_b_result.value if rm.pitch_b_result else None,
                }
                for rm in sm.reaction_mismatches
            ],
        }

    pitcher_name = at_bats[0].pitcher_name if at_bats else ""
    total_pitches = sum(len(ab.pitches) for ab in at_bats)

    return SequenceAnalysisResponse(
        pitcher_name=pitcher_name,
        game_date=req.date,
        total_at_bats=len(at_bats),
        total_pitches=total_pitches,
        at_bats=[_serialize_at_bat(ab) for ab in at_bats],
    )


# ---------------------------------------------------------------------------
# AWS Lambda handler (API Gateway + Lambda deploy via Mangum)
# ---------------------------------------------------------------------------
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    pass
