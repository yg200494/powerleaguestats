# Powerleague Stats – clean, mobile-first build
# Streamlit 1.37+, Python 3.10+
# -------------------------------------------------------------

from __future__ import annotations
import os
import io
import math
import uuid
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as html_component

# Supabase
from supabase import create_client, Client

# Optional HEIC support
_HEIC_AVAILABLE = False
try:
    import pillow_heif  # type: ignore
    _HEIC_AVAILABLE = True
except Exception:
    _HEIC_AVAILABLE = False

from PIL import Image

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="Powerleague Stats",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Safe access to secrets
def _secret(k: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[k]
    except Exception:
        return default

SUPABASE_URL = _secret("SUPABASE_URL", "")
SUPABASE_ANON_KEY = _secret("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = _secret("SUPABASE_SERVICE_KEY", "")
ADMIN_PASSWORD = _secret("ADMIN_PASSWORD", "")
AVATAR_BUCKET = _secret("AVATAR_BUCKET", "avatars")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing Supabase credentials in secrets. Add SUPABASE_URL and SUPABASE_ANON_KEY.")
    st.stop()

sb: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
sb_write: Optional[Client] = None
if SUPABASE_SERVICE_KEY:
    sb_write = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# -------------------------------------------------------------
# Helpers & Cache
# -------------------------------------------------------------
@st.cache_data(ttl=90, show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    try:
        data = sb.table(name).select("*").execute().data
        df = pd.DataFrame(data or [])
        return df
    except Exception as e:
        st.warning(f"Failed to load `{name}`: {e}")
        return pd.DataFrame()

def refresh_all():
    load_table.clear()
    st.experimental_rerun()

def parts_of(formation: str) -> List[int]:
    """Parse '1-2-1' or '2-1-2-1' -> [1,2,1] etc. Returns only outfield line sizes.
       We model GK as a separate single slot.
    """
    s = (formation or "").strip()
    if not s:
        return [1,2,1]
    try:
        parts = [int(x) for x in s.split("-") if x.strip().isdigit()]
        parts = [x for x in parts if x >= 0]
        if not parts:
            return [1,2,1]
        return parts
    except Exception:
        return [1,2,1]

def default_formation(side_count: int) -> str:
    return "2-1-2-1" if side_count >= 7 else "1-2-1"

def name_or_initials(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "?"
    parts = name.split()
    if len(parts) == 1 and len(parts[0]) <= 3:
        return parts[0].upper()
    initials = "".join([p[0] for p in parts[:2]]).upper()
    return initials or name[:2].upper()

def _coerce_int(x, fallback: int = 0) -> int:
    try:
        if pd.isna(x):
            return fallback
        if isinstance(x, bool):
            return int(x)
        return int(x)
    except Exception:
        return fallback

# -------------------------------------------------------------
# Pitch SVG (responsive, both teams, GK visible, mobile-first)
# -------------------------------------------------------------
def _layout_side(team_rows: pd.DataFrame, parts: List[int], left_side: bool,
                 W: int, H: int, margin: int) -> List[str]:
    """Return positioned player nodes for one side.
       GK at line=0, slot=0 by convention here.
    """
    out: List[str] = []

    # Safety columns
    for col in ["name", "goals", "assists", "is_gk", "line", "slot", "photo_url"]:
        if col not in team_rows.columns:
            team_rows[col] = None

    # Geometry
    mid_x = W / 2
    half_w = (W - 2 * margin) / 2
    top = margin
    bottom = H - margin
    usable_h = bottom - top

    # Distribute lines: GK + len(parts)
    total_lines = 1 + len(parts)

    # Where is GK on each side?
    # Left team defends left goal: GK near left box center
    # Right team defends right goal: GK near right box center
    gk_y = H / 2
    gk_x = margin + 10 if left_side else (W - margin - 10)

    # GK first
    gk = team_rows[team_rows["is_gk"] == True]
    if not gk.empty:
        gkr = gk.iloc[0].to_dict()
        out += _player_node(
            x=gk_x, y=gk_y,
            name=str(gkr.get("name") or gkr.get("player_name") or ""),
            goals=_coerce_int(gkr.get("goals"), 0),
            assists=_coerce_int(gkr.get("assists"), 0),
            motm=False,  # set later at merge stage if needed
            left_side=left_side,
            photo_url=str(gkr.get("photo_url") or ""),
        )

    # Outfield lines: spread evenly from GK towards midfield and attack end
    # y positions: distribute across usable_h centered, leave space under labels
    # For readability, bias a tiny bit toward midfield
    rows_only = team_rows[team_rows["is_gk"] != True].copy()
    # lines are 1..len(parts)
    for li, count in enumerate(parts, start=1):
        # y: evenly spaced lines
        y = top + (usable_h * (li / (len(parts) + 1)))
        # x band for this side
        band_left = margin
        band_right = margin + half_w
        if not left_side:
            band_left = W - margin - half_w
            band_right = W - margin
        # place players evenly across the band
        if count <= 0:
            continue
        xs = [band_left + (band_right - band_left) * (i + 1) / (count + 1) for i in range(count)]
        # Pull players for this line (slot order)
        line_players = rows_only[rows_only["line"] == li].sort_values("slot")
        # If data missing, fallback to first N rows remaining
        if line_players.shape[0] < count:
            need = count - line_players.shape[0]
            extra = rows_only[~rows_only.index.isin(line_players.index)].head(need)
            line_players = pd.concat([line_players, extra], ignore_index=True)

        for i, (idx, pr) in enumerate(line_players.head(count).iterrows()):
            px = xs[i]
            py = y
            out += _player_node(
                x=px, y=py,
                name=str(pr.get("name") or pr.get("player_name") or ""),
                goals=_coerce_int(pr.get("goals"), 0),
                assists=_coerce_int(pr.get("assists"), 0),
                motm=False,  # inject later
                left_side=left_side,
                photo_url=str(pr.get("photo_url") or ""),
            )
    return out

def _player_node(x: float, y: float, name: str, goals: int, assists: int,
                 motm: bool, left_side: bool, photo_url: str) -> List[str]:
    """Return SVG for a single player node. No emoji; clean pills; initials if no photo."""
    initials = name_or_initials(name)
    radius = 20  # avatar
    pad_y = 36   # room below avatar for name + chips
    g_badge = f"<g><rect rx='8' ry='8' x='-18' y='-8' width='36' height='16' fill='#111'/><text x='0' y='4' text-anchor='middle' font-size='10' fill='#f5d042'>{goals}g</text></g>" if goals > 0 else ""
    a_badge = f"<g><rect rx='8' ry='8' x='-18' y='-8' width='36' height='16' fill='#111'/><text x='0' y='4' text-anchor='middle' font-size='10' fill='#4fd1c5'>{assists}a</text></g>" if assists > 0 else ""
    motm_star = "<circle cx='15' cy='-15' r='8' fill='#f5d042'/>" if motm else ""

    # Avatar (photo or initials)
    if photo_url:
        avatar = (
            f"<clipPath id='clip_{uuid.uuid4().hex}'>"
            f"<circle cx='{x}' cy='{y}' r='{radius}'/></clipPath>"
            f"<image href='{photo_url}' x='{x - radius}' y='{y - radius}' width='{2*radius}' height='{2*radius}' preserveAspectRatio='xMidYMid slice' clip-path='url(#clip_{uuid.uuid4().hex})' />"
        )
    else:
        avatar = (
            f"<circle cx='{x}' cy='{y}' r='{radius}' fill='#222' stroke='#555' stroke-width='1'/>"
            f"<text x='{x}' y='{y+5}' text-anchor='middle' font-weight='700' font-size='12' fill='#eaeaea'>{initials}</text>"
        )

    chips = ""
    if goals > 0 and assists > 0:
        chips = (
            f"<g transform='translate({x-22},{y+radius+6})'>{g_badge}</g>"
            f"<g transform='translate({x+22},{y+radius+6})'>{a_badge}</g>"
        )
    elif goals > 0:
        chips = f"<g transform='translate({x},{y+radius+6})'>{g_badge}</g>"
    elif assists > 0:
        chips = f"<g transform='translate({x},{y+radius+6})'>{a_badge}</g>"

    name_text = f"<text x='{x}' y='{y+radius+24}' text-anchor='middle' font-size='12' fill='#ffffff'>{name}</text>"

    node = [
        f"<g class='player'>",
        motm_star.replace("cx='15'", f"cx='{x+15}'").replace("cy='-15'", f"cy='{y-15}'"),
        avatar,
        chips,
        name_text,
        "</g>",
    ]
    return node

def render_pitch_svg(a_rows: pd.DataFrame, b_rows: pd.DataFrame,
                     formation_a: str, formation_b: str,
                     motm_name: Optional[str]) -> str:
    """Produce the full SVG pitch with both teams arranged cleanly for mobile."""
    # Base geometry
    W, H = 1000, 600
    margin = 40
    mid_x = W / 2

    # Boxes
    box_top = 150
    box_bot = H - 150
    left_box_w = 180
    six_w = 80
    six_top = H/2 - 80/2
    six_bot = H/2 + 80/2
    goal_depth = 20

    # Background + lines
    pitch = []
    pitch.append(f"<svg viewBox='0 0 {W} {H}' width='100%' height='100%' preserveAspectRatio='xMidYMid meet'>")
    pitch.append(
        "<defs>"
        "<linearGradient id='g' x1='0' y1='0' x2='0' y2='1'>"
        "<stop offset='0%' stop-color='#2f7a43'/>"
        "<stop offset='60%' stop-color='#2a6f3c'/>"
        "<stop offset='100%' stop-color='#235f34'/>"
        "</linearGradient>"
        "</defs>"
    )
    pitch.append(f"<rect x='0' y='0' width='{W}' height='{H}' fill='url(#g)'/>")
    pitch.append(f"<rect x='{margin}' y='{margin}' width='{W - 2*margin}' height='{H - 2*margin}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<line x1='{mid_x}' y1='{margin}' x2='{mid_x}' y2='{H - margin}' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<circle cx='{mid_x}' cy='{H/2}' r='65' fill='none' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<circle cx='{mid_x}' cy='{H/2}' r='4' fill='#ffffff'/>")

    # Left box
    pitch.append(f"<rect x='{margin}' y='{box_top}' width='{left_box_w}' height='{box_bot - box_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<rect x='{margin}' y='{six_top}' width='{six_w}' height='{six_bot - six_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<line x1='{margin - goal_depth}' y1='{H/2 - 8}' x2='{margin}' y2='{H/2 - 8}' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<line x1='{margin - goal_depth}' y1='{H/2 + 8}' x2='{margin}' y2='{H/2 + 8}' stroke='#ffffff' stroke-width='3'/>")

    # Right box
    pitch.append(f"<rect x='{W - margin - left_box_w}' y='{box_top}' width='{left_box_w}' height='{box_bot - box_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<rect x='{W - margin - six_w}' y='{six_top}' width='{six_w}' height='{six_bot - six_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<line x1='{W - margin}' y1='{H/2 - 8}' x2='{W - margin + goal_depth}' y2='{H/2 - 8}' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<line x1='{W - margin}' y1='{H/2 + 8}' x2='{W - margin + goal_depth}' y2='{H/2 + 8}' stroke='#ffffff' stroke-width='3'/>")

    # Determine MOTM on each side
    motm = (motm_name or "").strip().lower()
    if "name" not in a_rows.columns: a_rows["name"] = a_rows.get("player_name")
    if "name" not in b_rows.columns: b_rows["name"] = b_rows.get("player_name")
    a_rows = a_rows.copy()
    b_rows = b_rows.copy()
    a_rows["motm"] = a_rows["name"].astype(str).str.lower().eq(motm)
    b_rows["motm"] = b_rows["name"].astype(str).str.lower().eq(motm)

    # Pull team photos by joining players table (for avatars)
    players = load_table("players")
    if not players.empty:
        a_rows = a_rows.merge(players[["name", "photo_url"]], on="name", how="left", suffixes=("",""))
        b_rows = b_rows.merge(players[["name", "photo_url"]], on="name", how="left", suffixes=("",""))
    else:
        a_rows["photo_url"] = ""
        b_rows["photo_url"] = ""

    pa = parts_of(formation_a)
    pb = parts_of(formation_b)

    pitch += _layout_side(a_rows, pa, left_side=True,  W=W, H=H, margin=margin)
    pitch += _layout_side(b_rows, pb, left_side=False, W=W, H=H, margin=margin)

    pitch.append("</svg>")

    svg = "".join(pitch)
    wrapper = (
        "<div style='width:100%;max-width:1000px;margin-inline:auto;'>"
        f"{svg}"
        "</div>"
    )
    return wrapper

def render_match_pitch(a_rows: pd.DataFrame, b_rows: pd.DataFrame,
                       formation_a: str, formation_b: str,
                       motm_name: Optional[str] = None,
                       height: int = 520):
    inner = render_pitch_svg(a_rows, b_rows, formation_a, formation_b, motm_name)
    # Responsive wrapper; height adapt—let browser scale to width
    html_component(
        f"""
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1"/>
          <style>
            html,body{{margin:0;padding:0;background:transparent}}
          </style>
        </head>
        <body>{inner}</body>
        </html>
        """,
        height=height, scrolling=False
    )

# -------------------------------------------------------------
# Data ops (safe upserts)
# -------------------------------------------------------------
def _service() -> Client:
    if not sb_write:
        st.error("ADMIN write requires SUPABASE_SERVICE_KEY in secrets.")
        st.stop()
    return sb_write

def upsert_match_row(row: Dict):
    _service().table("matches").upsert(row, on_conflict="id").execute()

def delete_team_lineups(match_id: str, team: str):
    _service().table("lineups").delete().eq("match_id", match_id).eq("team", team).execute()

def insert_lineups(rows: List[Dict]):
    # Ensure id present; ints/booleans correct
    prepared = []
    for r in rows:
        rr = dict(r)
        rr["id"] = rr.get("id") or str(uuid.uuid4())
        rr["is_gk"] = bool(rr.get("is_gk", False))
        rr["line"] = _coerce_int(rr.get("line"), 0)
        rr["slot"] = _coerce_int(rr.get("slot"), 0)
        rr["goals"] = _coerce_int(rr.get("goals"), 0)
        rr["assists"] = _coerce_int(rr.get("assists"), 0)
        prepared.append(rr)
    # chunked
    for i in range(0, len(prepared), 500):
        _service().table("lineups").insert(prepared[i:i+500]).execute()

def upsert_player_name_photo(player_id: Optional[str], name: str, notes: str, photo_url: str):
    s = _service()
    if player_id:
        s.table("players").update({"name": name, "notes": notes, "photo_url": photo_url}).eq("id", player_id).execute()
    else:
        # Upsert by unique name
        s.table("players").upsert({"name": name, "notes": notes, "photo_url": photo_url}, on_conflict="name").execute()

# -------------------------------------------------------------
# Fact table for Stats
# -------------------------------------------------------------
@st.cache_data(ttl=120)
def build_fact(players: pd.DataFrame, matches: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    if lineups.empty or matches.empty:
        return pd.DataFrame()

    l = lineups.copy()
    m = matches.copy()

    # normalize columns
    for c in ["season", "gw", "goals", "assists", "line", "slot"]:
        if c in l.columns:
            l[c] = pd.to_numeric(l[c], errors="coerce").fillna(0).astype(int)
    for c in ["score_a", "score_b", "season", "gw"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0).astype(int)

    # standardize player column
    if "name" not in l.columns:
        l["name"] = l.get("player_name")
    l["name"] = l["name"].fillna("").astype(str)

    # join to get team score side
    mm = m[["id","season","gw","team_a","team_b","score_a","score_b","is_draw","motm_name","formation_a","formation_b"]].rename(columns={"id":"match_id"})
    fact = l.merge(mm, on=["match_id","season","gw"], how="left")

    # win/draw/loss by team
    def wdl(row):
        if row["is_draw"]:
            return "D"
        if row["team"] == "Non-bibs":
            return "W" if row["score_a"] > row["score_b"] else "L"
        else:
            return "W" if row["score_b"] > row["score_a"] else "L"
    fact["result"] = fact.apply(wdl, axis=1)

    # team goals for contribution%
    tg = fact.groupby(["match_id","team"], as_index=False).agg(team_goals=("goals","sum"))
    fact = fact.merge(tg, on=["match_id","team"], how="left")
    fact["ga"] = fact["goals"] + fact["assists"]
    fact["contrib_pct"] = (fact["ga"] / fact["team_goals"].replace(0, pd.NA) * 100).fillna(0.0).round(1)

    return fact

# -------------------------------------------------------------
# UI: Admin & Header
# -------------------------------------------------------------
def header():
    st.markdown(
        """
        <style>
          .topbar { position: sticky; top: 0; z-index: 100; background: #0f172a; color: #fff;
                    padding: 10px 12px; border-bottom: 1px solid #1f2937; }
          .topbar .brand { font-weight: 700; letter-spacing: 0.3px; }
          .brand small { opacity: .7; font-weight: 400; margin-left: 6px; }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.container():
        st.markdown("<div class='topbar'><span class='brand'>Powerleague Stats</span><small>mobile-first</small></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3,1,1])
    with col1:
        pass
    with col2:
        if st.session_state.get("is_admin"):
            st.success("Admin mode")
        else:
            st.info("Read-only")

    with col3:
        if not st.session_state.get("is_admin"):
            with st.expander("Admin login", expanded=False):
                pwd = st.text_input("Password", type="password", key="admin_pwd")
                if st.button("Login", key="btn_login"):
                    if pwd and pwd == ADMIN_PASSWORD:
                        st.session_state["is_admin"] = True
                        st.success("Logged in as admin.")
                        st.rerun()
                    else:
                        st.error("Wrong password.")
        else:
            if st.button("Logout", key="btn_logout"):
                st.session_state["is_admin"] = False
                st.experimental_rerun()

# -------------------------------------------------------------
# Page: Matches (summary + lineup editor)
# -------------------------------------------------------------
def auto_assign_from_order(names: List[str], side_count: int) -> List[Tuple[int,int]]:
    """Return list of (line,slot) for each name in order: GK first, then formation lines."""
    parts = parts_of(default_formation(side_count))
    out: List[Tuple[int,int]] = []
    if not names:
        return out
    # GK
    out.append((0,0))
    idx = 1
    for li, count in enumerate(parts, start=1):
        for sj in range(count):
            if idx >= len(names): break
            out.append((li, sj))
            idx += 1
    return out

def team_lineup_editor(team_label: str, mid: str, side_count: int,
                       formation: str, team_rows: pd.DataFrame,
                       all_players: pd.DataFrame, keypref: str):
    """Simple slot editor: choose formation, then assign each slot from available names (this team only)."""
    # Available names: those already in this match for this team
    existing_names = team_rows["name"].fillna(team_rows.get("player_name")).dropna().astype(str).unique().tolist()
    # Fallback: suggest from players table
    if not existing_names and not all_players.empty:
        existing_names = all_players["name"].dropna().astype(str).unique().tolist()

    st.markdown(f"**{team_label}** – lineup editor")

    # Formation picker (5s / 7s presets)
    presets_5 = ["1-2-1", "2-2", "1-3"]
    presets_7 = ["2-1-2-1", "1-2-3", "3-2-1"]
    preset_list = presets_7 if side_count >= 7 else presets_5
    fcol1, fcol2 = st.columns([2,1])
    with fcol1:
        new_form = st.selectbox(
            "Formation",
            preset_list,
            index=(preset_list.index(formation) if formation in preset_list else 0),
            key=f"{keypref}_formation"
        )
    with fcol2:
        if st.session_state.get("is_admin"):
            if st.button("Save Formation", key=f"{keypref}_save_form"):
                if team_label == "Non-bibs":
                    upsert_match_row({"id": mid, "formation_a": new_form})
                else:
                    upsert_match_row({"id": mid, "formation_b": new_form})
                refresh_all()

    parts = parts_of(new_form)

    # Build slot table: GK + per line slots
    slot_defs: List[Tuple[int,int,str]] = [(0,0,"GK")]
    for li, cnt in enumerate(parts, start=1):
        for sj in range(cnt):
            slot_defs.append((li, sj, f"L{li} S{sj+1}"))

    # Editors
    chosen: Dict[Tuple[int,int], str] = {}
    names_for_pick = ["—"] + existing_names
    for li, sj, label in slot_defs:
        # Pre-fill current
        cur = team_rows[(team_rows["line"]==li) & (team_rows["slot"]==sj)]
        if li == 0 and sj == 0:
            cur = team_rows[team_rows["is_gk"]==True]
        cur_name = ""
        if not cur.empty:
            cur_name = str(cur.iloc[0].get("name") or cur.iloc[0].get("player_name") or "")
        sel = st.selectbox(label, names_for_pick,
                           index=(names_for_pick.index(cur_name) if cur_name in names_for_pick else 0),
                           key=f"{keypref}_slot_{li}_{sj}")
        chosen[(li,sj)] = "" if sel == "—" else sel

    # Autopopulate from order button
    if st.button("Auto-assign from current order", key=f"{keypref}_auto"):
        order = existing_names[:len(slot_defs)]
        mapping = auto_assign_from_order(order, side_count)
        for i, (li, sj) in enumerate(mapping):
            try:
                nm = order[i]
            except IndexError:
                nm = ""
            st.session_state[f"{keypref}_slot_{li}_{sj}"] = nm
        st.rerun()

    # Save
    if st.session_state.get("is_admin"):
        if st.button("Save this team lineup", key=f"{keypref}_save"):
            # Delete existing rows for this match+team then insert selected
            delete_team_lineups(mid, team_label)
            # lookup player_id by name
            pl = load_table("players")[["id","name"]] if not load_table("players").empty else pd.DataFrame()
            rows_to_insert: List[Dict] = []
            for (li, sj), nm in chosen.items():
                if not nm:
                    continue
                pid = None
                if not pl.empty:
                    m = pl[pl["name"]==nm]
                    if not m.empty:
                        pid = m.iloc[0]["id"]
                rows_to_insert.append({
                    "id": str(uuid.uuid4()),
                    "season": int(st.session_state.get("season", 0) or 0),  # optional
                    "gw": int(st.session_state.get("gw", 0) or 0),          # optional
                    "match_id": mid,
                    "team": team_label,
                    "player_id": pid,
                    "player_name": nm,
                    "is_gk": (li == 0 and sj == 0),
                    "goals": 0,
                    "assists": 0,
                    "line": li,
                    "slot": sj,
                    "position": ""
                })
            if rows_to_insert:
                insert_lineups(rows_to_insert)
                st.success("Saved lineup.")
                refresh_all()
            else:
                st.warning("No slots selected to save.")

def page_matches():
    header()

    matches = load_table("matches")
    lineups = load_table("lineups")
    players = load_table("players")

    if matches.empty:
        st.info("No matches yet.")
        return

    # Select match (season + GW label)
    matches["_label"] = matches.apply(lambda r: f"S{int(r.get('season') or 0)} – GW{int(r.get('gw') or 0)}", axis=1)
    matches = matches.sort_values(["season","gw"], ascending=[True, True])
    sel_label = st.selectbox("Select match", matches["_label"].tolist(), key="match_sel")
    m = matches[matches["_label"]==sel_label].iloc[0]
    mid = m["id"]
    st.session_state["season"] = int(m.get("season") or 0)
    st.session_state["gw"] = int(m.get("gw") or 0)

    # Summary header
    left, right = st.columns([1,1])
    with left:
        st.subheader(f"{m.get('team_a','Non-bibs')} {int(m.get('score_a') or 0)} – {int(m.get('score_b') or 0)} {m.get('team_b','Bibs')}")
        if str(m.get("motm_name") or "").strip():
            st.caption(f"⭐ MOTM: {m['motm_name']}")
        st.caption(f"Formation A: {m.get('formation_a') or default_formation(int(m.get('side_count') or 5))} | Formation B: {m.get('formation_b') or default_formation(int(m.get('side_count') or 5))}")
        if str(m.get("notes") or "").strip():
            st.write(m["notes"])

    # Build team dfs for pitch
    a_rows = lineups[(lineups["match_id"]==mid) & (lineups["team"]=="Non-bibs")].copy()
    b_rows = lineups[(lineups["match_id"]==mid) & (lineups["team"]=="Bibs")].copy()
    # ensure base columns
    for df in (a_rows, b_rows):
        for c in ["name","player_name","goals","assists","is_gk","line","slot"]:
            if c not in df.columns:
                df[c] = None
        df["name"] = df["name"].fillna(df["player_name"]).fillna("").astype(str)
        for c in ["goals","assists","line","slot"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        df["is_gk"] = df["is_gk"].fillna(False).astype(bool)

    # Pitch render
    fa = m.get("formation_a") or default_formation(int(m.get("side_count") or 5))
    fb = m.get("formation_b") or default_formation(int(m.get("side_count") or 5))
    render_match_pitch(a_rows, b_rows, fa, fb, m.get("motm_name"), height=520)

    st.divider()
    st.subheader("Arrange lineup (Admin)")
    cA, cB = st.columns(2)
    with cA:
        team_lineup_editor("Non-bibs", mid, int(m.get("side_count") or 5), fa, a_rows, players, keypref=f"A_{mid}")
    with cB:
        team_lineup_editor("Bibs", mid, int(m.get("side_count") or 5), fb, b_rows, players, keypref=f"B_{mid}")

# -------------------------------------------------------------
# Page: Stats (filters + leaderboards + duos/nemesis)
# -------------------------------------------------------------
def leaderboard_table(df: pd.DataFrame, metric: str, min_games: int, top_n: int):
    if df.empty:
        st.info("No data.")
        return
    # Aggregate per player
    agg = df.groupby("name", as_index=False).agg(
        GP=("match_id","nunique"),
        W=("result", lambda s: (s=="W").sum()),
        D=("result", lambda s: (s=="D").sum()),
        L=("result", lambda s: (s=="L").sum()),
        Goals=("goals","sum"),
        Assists=("assists","sum"),
        GA=("ga","sum"),
        WinPct=("result", lambda s: (s=="W").mean()*100.0),
        ContribPct=("contrib_pct","mean"),
    )
    agg = agg[agg["GP"] >= min_games]

    metric_map = {
        "Goals": "Goals",
        "Assists": "Assists",
        "G+A": "GA",
        "Win %": "WinPct",
        "Contribution %": "ContribPct",
    }
    col = metric_map.get(metric, "GA")
    agg = agg.sort_values([col,"GA"], ascending=[False,False]).head(top_n)
    st.dataframe(agg.reset_index(drop=True), use_container_width=True)

def duos_table(df: pd.DataFrame, min_games: int, top_n: int, best: bool):
    # Duo = two-player pair same team, evaluate pair GA and Win%
    if df.empty:
        st.info("No data.")
        return
    g = df.groupby(["match_id","team"])
    pairs = []
    for (mid, team), grp in g:
        names = grp["name"].dropna().unique().tolist()
        stats = grp.set_index("name")[["goals","assists","result"]]
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                n1, n2 = names[i], names[j]
                r = stats.loc[[n1,n2]]
                ga = int(r["goals"].sum() + r["assists"].sum())
                res = r["result"]
                # same match -> both have same W/D/L; take first
                pairs.append({"pair": " & ".join(sorted([n1,n2])),
                              "match_id": mid, "team": team,
                              "ga": ga,
                              "win": int((res.iloc[0]=="W"))})
    if not pairs:
        st.info("No pairs yet.")
        return
    dfp = pd.DataFrame(pairs)
    agg = dfp.groupby("pair", as_index=False).agg(
        GP=("match_id","nunique"),
        GA=("ga","sum"),
        Wins=("win","sum"),
        WinPct=("win", lambda s: s.mean()*100.0)
    )
    agg = agg[agg["GP"] >= min_games]
    agg = agg.sort_values(["WinPct","GA"], ascending=[not best, not best]).head(top_n)
    st.dataframe(agg.reset_index(drop=True), use_container_width=True)

def nemesis_table(df: pd.DataFrame, min_games: int, top_n: int):
    # Nemesis = opponent causing lowest Win% for the player
    if df.empty:
        st.info("No data.")
        return
    # Build per-match opponent set for each player
    # Approach: for each match_id, cross team A names vs team B names
    ag = []
    for mid, grp in df.groupby("match_id"):
        a = grp[grp["team"]=="Non-bibs"]["name"].dropna().unique().tolist()
        b = grp[grp["team"]=="Bibs"]["name"].dropna().unique().tolist()
        # result for A to derive
        resA = grp[grp["team"]=="Non-bibs"]["result"].iloc[0] if not grp[grp["team"]=="Non-bibs"].empty else "D"
        resB = grp[grp["team"]=="Bibs"]["result"].iloc[0] if not grp[grp["team"]=="Bibs"].empty else "D"
        for x in a:
            for y in b:
                ag.append({"p": x, "opp": y, "win": int(resA=="W"), "mid": mid})
        for x in b:
            for y in a:
                ag.append({"p": x, "opp": y, "win": int(resB=="W"), "mid": mid})
    if not ag:
        st.info("No opponent pairs.")
        return
    df2 = pd.DataFrame(ag)
    agg = df2.groupby(["p","opp"], as_index=False).agg(
        GP=("mid","nunique"),
        Wins=("win","sum"),
        WinPct=("win", lambda s: s.mean()*100.0)
    )
    # Nemesis: for each p, the opponent with lowest Win% (min GP filter)
    out = []
    for p, grp in agg.groupby("p"):
        g2 = grp[grp["GP"] >= min_games]
        if g2.empty:
            continue
        worst = g2.sort_values(["WinPct","GP"], ascending=[True,False]).head(top_n)
        worst = worst.assign(Player=p)
        out.append(worst)
    if not out:
        st.info("No nemesis pass GP filter.")
        return
    df_out = pd.concat(out, ignore_index=True)[["Player","opp","GP","WinPct","Wins"]].rename(columns={"opp":"Nemesis"})
    st.dataframe(df_out, use_container_width=True)

def page_stats():
    header()
    players = load_table("players")
    matches = load_table("matches")
    lineups = load_table("lineups")
    fact = build_fact(players, matches, lineups)

    if fact.empty:
        st.info("No data.")
        return

    # Filters
    seasons = sorted(fact["season"].dropna().unique().tolist())
    seasons = [int(x) for x in seasons if pd.notna(x)]
    seasons = ([-1] + seasons) if seasons else [-1]
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        sel_season = st.selectbox("Season (or All)", seasons, format_func=lambda x: "All" if x==-1 else x, index=len(seasons)-1)
    with c2:
        min_games = st.number_input("Min games", min_value=0, max_value=200, value=2, step=1)
    with c3:
        last_n = st.number_input("Last N GWs (0 = all)", min_value=0, max_value=500, value=0, step=1)
    with c4:
        top_n = st.number_input("Rows", min_value=5, max_value=200, value=20, step=5)

    f = fact.copy()
    if sel_season != -1:
        f = f[f["season"]==sel_season]
    if last_n and "gw" in f.columns:
        max_gw = f["gw"].max()
        f = f[f["gw"] >= max(1, max_gw - last_n + 1)]

    st.subheader("Leaderboards")
    metric = st.selectbox("Metric", ["G+A", "Goals", "Assists", "Win %", "Contribution %"], index=0)
    leaderboard_table(f, metric, min_games, top_n)

    st.subheader("Best Duos")
    duos_table(f, min_games=max(1, min_games), top_n=top_n, best=True)

    st.subheader("Nemesis")
    nemesis_table(f, min_games=max(1, min_games), top_n=3)

# -------------------------------------------------------------
# Page: Players
# -------------------------------------------------------------
def page_players():
    header()
    players = load_table("players")
    matches = load_table("matches")
    lineups = load_table("lineups")
    fact = build_fact(players, matches, lineups)

    if players.empty:
        st.info("No players.")
        return

    names = players["name"].dropna().astype(str).sort_values().tolist()
    sel = st.selectbox("Player", names, key="player_sel")
    p = players[players["name"]==sel].iloc[0]

    st.subheader(sel)
    cols = st.columns([1,2])
    with cols[0]:
        photo_url = p.get("photo_url") or ""
        if photo_url:
            st.image(photo_url, width=220)
        else:
            st.markdown(f"<div style='width:220px;height:220px;border-radius:16px;background:#111;color:#eee;display:flex;align-items:center;justify-content:center;font-size:42px;font-weight:700'>{name_or_initials(sel)}</div>", unsafe_allow_html=True)
        st.caption(p.get("notes") or "")

    with cols[1]:
        mine = fact[fact["name"]==sel]
        if mine.empty:
            st.info("No appearances yet.")
            return
        gp = mine["match_id"].nunique()
        w = (mine["result"]=="W").sum()
        d = (mine["result"]=="D").sum()
        l = (mine["result"]=="L").sum()
        g = mine["goals"].sum()
        a = mine["assists"].sum()
        ga = g + a
        winpct = (w / gp * 100.0) if gp else 0.0
        contrib = mine["contrib_pct"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GP", gp)
        c2.metric("W-D-L", f"{w}-{d}-{l}")
        c3.metric("Win %", f"{winpct:.1f}%")
        c4.metric("G + A", ga)

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Goals", g)
        c6.metric("Assists", a)
        c7.metric("Avg Contrib %", f"{contrib:.1f}%")
        c8.metric("G+A / game", f"{(ga/gp):.2f}" if gp else "0")

        # Last N games W/L/D string
        N = st.number_input("Last N games for form", min_value=3, max_value=30, value=5, step=1)
        last = mine.sort_values(["season","gw"], ascending=[True, True]).tail(int(N))
        form = "".join(last["result"].map({"W":"W","D":"D","L":"L"}).tolist())
        st.write(f"Recent form: **{form}**")

        # Recent games list
        st.markdown("**Recent games**")
        for _, r in mine.sort_values(["season","gw"], ascending=[False, False]).head(10).iterrows():
            st.caption(f"S{int(r['season'])} GW{int(r['gw'])} – {r['team']}: {int(r['goals'])}g {int(r['assists'])}a ({r['result']})")

# -------------------------------------------------------------
# Page: Awards
# -------------------------------------------------------------
def page_awards():
    header()
    matches = load_table("matches")
    awards = load_table("awards")

    st.subheader("Man of the Match (from matches)")
    if matches.empty:
        st.info("No matches.")
    else:
        motm = matches[["season","gw","motm_name"]].copy()
        motm = motm[motm["motm_name"].notna() & (motm["motm_name"].astype(str)!="")]
        if motm.empty:
            st.info("No MOTM recorded.")
        else:
            motm = motm.sort_values(["season","gw"])
            st.dataframe(motm.reset_index(drop=True), use_container_width=True)

    st.subheader("Player of the Month (manual)")
    if not st.session_state.get("is_admin"):
        st.info("Login as admin to add POTM.")
        if not awards.empty:
            potm = awards[awards["type"]=="POTM"][["season","month","player_name","notes"]].sort_values(["season","month"])
            if potm.empty:
                st.info("No POTM yet.")
            else:
                st.dataframe(potm, use_container_width=True)
        return

    # admin add POTM
    col1, col2, col3 = st.columns(3)
    with col1:
        season = st.number_input("Season", min_value=2000, max_value=2100, value=date.today().year, step=1, key="aw_season")
    with col2:
        month = st.selectbox("Month", list(range(1,13)), index=(date.today().month-1), key="aw_month")
    players = load_table("players")
    with col3:
        name = st.selectbox("Player", players["name"].dropna().astype(str).sort_values().tolist(), key="aw_name")
    notes = st.text_input("Notes (optional)", key="aw_notes")
    if st.button("Add POTM"):
        _service().table("awards").insert({
            "id": str(uuid.uuid4()),
            "season": int(season),
            "month": int(month),
            "type": "POTM",
            "gw": None,
            "player_id": None,
            "player_name": name,
            "notes": notes
        }).execute()
        st.success("POTM added.")
        refresh_all()

    # show table
    awards = load_table("awards")
    if not awards.empty:
        potm = awards[awards["type"]=="POTM"][["season","month","player_name","notes"]].sort_values(["season","month"])
        st.dataframe(potm, use_container_width=True)

# -------------------------------------------------------------
# Page: Player Manager (add/edit + avatar upload)
# -------------------------------------------------------------
def page_player_manager():
    header()
    if not st.session_state.get("is_admin"):
        st.info("Admin only.")
        return

    players = load_table("players")
    st.subheader("Edit player")
    col1, col2 = st.columns([2,1])

    with col1:
        current = None
        if not players.empty:
            sel = st.selectbox("Select player", players["name"].dropna().astype(str).sort_values().tolist(), key="pm_sel")
            current = players[players["name"]==sel].iloc[0]
        else:
            st.info("No players yet.")

        new_name = st.text_input("Name", value=(current["name"] if current is not None else ""), key="pm_name")
        notes = st.text_area("Notes", value=(current.get("notes","") if current is not None else ""), key="pm_notes", height=120)

        uploaded = st.file_uploader("Avatar (JPG/PNG/HEIC)", type=["jpg","jpeg","png","heic","HEIC"], key="pm_file")
        photo_url = current.get("photo_url","") if current is not None else ""
        if uploaded is not None:
            img: Optional[Image.Image] = None
            fmt = (uploaded.name.split(".")[-1] or "").lower()
            try:
                if fmt in ("heic",) and _HEIC_AVAILABLE:
                    heif = pillow_heif.read_heif(uploaded.read())
                    img = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
                elif fmt in ("heic",) and not _HEIC_AVAILABLE:
                    st.error("HEIC support not available on this host. Please upload JPG/PNG.")
                else:
                    img = Image.open(uploaded).convert("RGB")
            except Exception as e:
                st.error(f"Failed to read image: {e}")
                img = None

            if img is not None:
                # Normalize & upload to avatars bucket
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                buf.seek(0)
                key = f"{uuid.uuid4().hex}.png"
                try:
                    _service().storage.from_(AVATAR_BUCKET).upload(key, buf, {"content-type": "image/png", "x-upsert": "true"})
                    # public URL
                    public = _service().storage.from_(AVATAR_BUCKET).get_public_url(key)
                    if isinstance(public, dict) and "publicUrl" in public:
                        photo_url = public["publicUrl"]
                    elif isinstance(public, str):
                        photo_url = public
                    st.image(photo_url, width=160, caption="Uploaded")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

        if st.button("Save player"):
            pid = current.get("id") if current is not None else None
            upsert_player_name_photo(pid, new_name.strip(), notes.strip(), photo_url.strip())
            st.success("Saved.")
            refresh_all()

    with col2:
        if current is not None:
            st.caption("Current avatar")
            if current.get("photo_url"):
                st.image(current["photo_url"], width=140)
            else:
                st.markdown(f"<div style='width:140px;height:140px;border-radius:14px;background:#111;color:#eee;display:flex;align-items:center;justify-content:center;font-size:32px;font-weight:700'>{name_or_initials(current['name'])}</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Router
# -------------------------------------------------------------
def run_app():
    pages = ["Matches", "Stats", "Players", "Awards", "Player Manager"]
    page = st.sidebar.radio("Go to", pages, index=0, key="nav_page")
    if page == "Matches": page_matches()
    elif page == "Stats": page_stats()
    elif page == "Players": page_players()
    elif page == "Awards": page_awards()
    elif page == "Player Manager": page_player_manager()

if __name__ == "__main__":
    # Initialize admin state flag
    st.session_state.setdefault("is_admin", False)
    run_app()
