# Powerleague Stats — clean, mobile-first, premium player profile
# Streamlit >=1.27 (tested up to 1.37), Python 3.10+

from __future__ import annotations
import io
import uuid
from datetime import date
from typing import Optional, Dict, List, Tuple

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as html_component
from supabase import create_client, Client

# Optional HEIC support (best-effort; app works without it)
_HEIC = False
try:
    import pillow_heif  # type: ignore
    _HEIC = True
except Exception:
    _HEIC = False
from PIL import Image

# -----------------------------------------------------------------------------
# Config & Secrets
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Powerleague Stats",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def _sec(k, default=None):
    try:
        return st.secrets[k]
    except Exception:
        return default

SUPABASE_URL        = _sec("SUPABASE_URL", "")
SUPABASE_ANON_KEY   = _sec("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY= _sec("SUPABASE_SERVICE_KEY", "")
ADMIN_PASSWORD      = _sec("ADMIN_PASSWORD", "")
AVATAR_BUCKET       = _sec("AVATAR_BUCKET", "avatars")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing Supabase credentials. Add SUPABASE_URL & SUPABASE_ANON_KEY to secrets.")
    st.stop()

sb: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
sb_write: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_SERVICE_KEY else None

# -----------------------------------------------------------------------------
# Theme / CSS (Black & Gold)
# -----------------------------------------------------------------------------
CSS = """
<style>
:root{
  --bg:#0c1015; --panel:#101822; --muted:#1f2a37; --text:#e7eaf0; --sub:#98a2b3;
  --gold:#f5d042; --teal:#4fd1c5; --chip:#12161c; --border:#1b2430;
}
html,body,.stApp{background:var(--bg); color:var(--text)}
.block-container{padding-top:0.6rem; padding-bottom:2rem}
.topbar{position:sticky; top:0; z-index:50; background:var(--panel); border-bottom:1px solid var(--border);
  padding:10px 12px;}
.brand{font-weight:800; letter-spacing:.2px}
.brand small{font-weight:400; opacity:.75; margin-left:6px}
.pitchWrap{width:100%; max-width:1000px; margin:0 auto}
.card{background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:12px 14px}
.statCard{background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:14px; text-align:center}
.statLabel{font-size:12px; color:var(--sub)}
.statValue{font-size:20px; font-weight:800; color:var(--gold)}
.badge{display:inline-flex; gap:6px; align-items:center; font-size:12px; background:var(--chip);
  border:1px solid var(--border); border-radius:999px; padding:2px 8px}
.small{color:var(--sub); font-size:12px}
.stTabs [data-baseweb="tab-list"]{gap:6px}
.stTabs [data-baseweb="tab"]{background:var(--panel); border-radius:10px 10px 0 0}
.stDataFrame, .stDataFrame div{color:var(--text) !important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Utilities & Cache
# -----------------------------------------------------------------------------
ALIAS_MAP = {
    "Ani": "Anirudh",
    "Abdullah Y13": "Mohammad Abdullah",
}

def normalize_name(s: str) -> str:
    s = (s or "").strip()
    return ALIAS_MAP.get(s, s)

@st.cache_data(ttl=120, show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    try:
        data = sb.table(name).select("*").execute().data
        return pd.DataFrame(data or [])
    except Exception as e:
        st.warning(f"Failed to load `{name}`: {e}")
        return pd.DataFrame()

def refresh_all():
    load_table.clear()
    st.rerun()

def name_initials(name: str) -> str:
    name = (str(name or "").strip())
    if not name: return "?"
    parts = name.split()
    if len(parts)==1: return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()

def parts_of(formation: str) -> List[int]:
    s = (formation or "").strip()
    if not s: return [1,2,1]
    try:
        p = [int(x) for x in s.split("-") if x.strip().isdigit()]
        p = [x for x in p if x >= 0]
        return p if p else [1,2,1]
    except Exception:
        return [1,2,1]

def default_formation(side_count: int) -> str:
    return "2-1-2-1" if int(side_count or 5) >= 7 else "1-2-1"

def _int(x, d=0) -> int:
    try:
        if pd.isna(x): return d
        if isinstance(x, bool): return int(x)
        return int(x)
    except Exception:
        return d

def _service() -> Client:
    if not sb_write:
        st.error("Writes require SUPABASE_SERVICE_KEY in secrets.")
        st.stop()
    return sb_write

def hide_index(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    try:
        return df.style.hide(axis="index")
    except Exception:
        return df  # fallback (older pandas)

# -----------------------------------------------------------------------------
# Pitch SVG (compact, readable, no overlap, GK visible)
# -----------------------------------------------------------------------------
def _player_node(x: float, y: float, name: str, goals: int, assists: int,
                 motm: bool, photo_url: str) -> List[str]:
    """SVG node: circular avatar, chips '1g'/'2a', name below."""
    initials = name_initials(name)
    r = 22
    chip_gap = 6
    clip_id = uuid.uuid4().hex

    def chip(text, fill):
        return f"<g><rect rx='8' ry='8' x='-16' y='-8' width='32' height='16' fill='#0e1319' stroke='#1b2430'/>" \
               f"<text x='0' y='4' text-anchor='middle' font-size='10' fill='{fill}'>{text}</text></g>"

    g_chip = chip(f"{goals}g", "#f5d042") if goals>0 else ""
    a_chip = chip(f"{assists}a", "#4fd1c5") if assists>0 else ""
    chips = ""
    if goals>0 and assists>0:
        chips = f"<g transform='translate({x-20},{y+r+chip_gap})'>{g_chip}</g><g transform='translate({x+20},{y+r+chip_gap})'>{a_chip}</g>"
    elif goals>0:
        chips = f"<g transform='translate({x},{y+r+chip_gap})'>{g_chip}</g>"
    elif assists>0:
        chips = f"<g transform='translate({x},{y+r+chip_gap})'>{a_chip}</g>"

    star = f"<circle cx='{x+16}' cy='{y-16}' r='8' fill='#f5d042'/>" if motm else ""

    if photo_url:
        avatar = (
            f"<clipPath id='clip_{clip_id}'><circle cx='{x}' cy='{y}' r='{r}'/></clipPath>"
            f"<image href='{photo_url}' x='{x-r}' y='{y-r}' width='{2*r}' height='{2*r}' preserveAspectRatio='xMidYMid slice' clip-path='url(#clip_{clip_id})' />"
        )
    else:
        avatar = (
            f"<circle cx='{x}' cy='{y}' r='{r}' fill='#1a2230' stroke='#2a3647' stroke-width='1'/>"
            f"<text x='{x}' y='{y+5}' text-anchor='middle' font-size='12' font-weight='700' fill='#e7eaf0'>{initials}</text>"
        )

    name_text = f"<text x='{x}' y='{y+r+24}' text-anchor='middle' font-size='12' fill='#ffffff'>{name}</text>"
    return [star, avatar, chips, name_text]

def _layout_side(team_rows: pd.DataFrame, parts: List[int], left_side: bool,
                 W: int, H: int, margin: int) -> List[str]:
    out: List[str] = []
    base_cols = ["name","player_name","goals","assists","is_gk","line","slot","photo_url","motm"]
    for c in base_cols:
        if c not in team_rows.columns:
            team_rows[c] = None
    team_rows["player_name"] = team_rows["player_name"].fillna("").astype(str)
    team_rows["name"] = team_rows["player_name"].map(normalize_name).fillna("").astype(str)
    for c in ["goals","assists","line","slot"]:
        team_rows[c] = pd.to_numeric(team_rows[c], errors="coerce").fillna(0).astype(int)
    team_rows["is_gk"] = team_rows["is_gk"].fillna(False).astype(bool)
    team_rows["motm"] = team_rows["motm"].fillna(False).astype(bool)
    team_rows["photo_url"] = team_rows["photo_url"].fillna("").astype(str)

    mid_x = W/2
    usable_w = (W - 2*margin) / 2
    left_x = margin
    right_x = margin + usable_w
    if not left_side:
        left_x = W - margin - usable_w
        right_x = W - margin
    top = margin
    bottom = H - margin
    usable_h = bottom - top

    # GK near box center
    gk_x = left_x + 14 if left_side else right_x - 14
    gk_y = H / 2

    gk_row = team_rows[team_rows["is_gk"] == True]
    if not gk_row.empty:
        r = gk_row.iloc[0]
        out += _player_node(gk_x, gk_y, r["name"], _int(r["goals"]), _int(r["assists"]), bool(r["motm"]), r["photo_url"])

    # Outfield lines
    for li, count in enumerate(parts, start=1):
        y = top + usable_h * (li/(len(parts)+1))
        xs = [left_x + (usable_w*(i+1)/(count+1)) for i in range(count)]
        grp = team_rows[team_rows["is_gk"] != True]
        line_players = grp[grp["line"]==li].sort_values("slot")
        if line_players.shape[0] < count:
            need = count - line_players.shape[0]
            extra = grp[~grp.index.isin(line_players.index)].head(need)
            line_players = pd.concat([line_players, extra], ignore_index=True)
        for i, (_, pr) in enumerate(line_players.head(count).iterrows()):
            x = xs[i]
            out += _player_node(x, y, pr["name"], _int(pr["goals"]), _int(pr["assists"]), bool(pr["motm"]), pr["photo_url"])
    return out

def render_pitch_svg(a_rows: pd.DataFrame, b_rows: pd.DataFrame,
                     formation_a: str, formation_b: str,
                     motm_name: Optional[str]) -> str:
    W, H = 1000, 560
    margin = 36
    mid_x = W/2

    # Normalize names & set motm flags
    def ensure(df: pd.DataFrame) -> pd.DataFrame:
        if "player_name" not in df.columns: df["player_name"] = ""
        df["player_name"] = df["player_name"].fillna("").astype(str)
        df["name"] = df["player_name"].map(normalize_name)
        return df

    a_rows = ensure(a_rows.copy())
    b_rows = ensure(b_rows.copy())
    motm = (motm_name or "").strip().lower()
    a_rows["motm"] = a_rows["name"].str.lower().eq(motm)
    b_rows["motm"] = b_rows["name"].str.lower().eq(motm)

    players_df = load_table("players")
    if not players_df.empty:
        players_df["name"] = players_df["name"].fillna("").astype(str).map(normalize_name)
        a_rows = a_rows.merge(players_df[["name","photo_url"]], on="name", how="left")
        b_rows = b_rows.merge(players_df[["name","photo_url"]], on="name", how="left")
    else:
        a_rows["photo_url"] = ""
        b_rows["photo_url"] = ""

    pa = parts_of(formation_a)
    pb = parts_of(formation_b)

    svg = []
    svg.append(f"<svg viewBox='0 0 {W} {H}' width='100%' height='100%' preserveAspectRatio='xMidYMid meet'>")
    svg.append(
        "<defs>"
        "<linearGradient id='g' x1='0' y1='0' x2='0' y2='1'>"
        "<stop offset='0%' stop-color='#215a33'/>"
        "<stop offset='100%' stop-color='#173b22'/>"
        "</linearGradient>"
        "</defs>"
    )
    svg.append(f"<rect x='0' y='0' width='{W}' height='{H}' fill='url(#g)'/>")
    svg.append(f"<rect x='{margin}' y='{margin}' width='{W-2*margin}' height='{H-2*margin}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<line x1='{mid_x}' y1='{margin}' x2='{mid_x}' y2='{H-margin}' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<circle cx='{mid_x}' cy='{H/2}' r='60' fill='none' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<circle cx='{mid_x}' cy='{H/2}' r='4' fill='#ffffff'/>")
    # Left goal/boxes
    box_top, box_bot = 120, H-120
    box_w = 170; six_w = 80; six_h = 80
    six_top, six_bot = H/2-six_h/2, H/2+six_h/2
    goal_depth = 20
    svg.append(f"<rect x='{margin}' y='{box_top}' width='{box_w}' height='{box_bot-box_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<rect x='{margin}' y='{six_top}' width='{six_w}' height='{six_bot-six_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<line x1='{margin-goal_depth}' y1='{H/2-8}' x2='{margin}' y2='{H/2-8}' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<line x1='{margin-goal_depth}' y1='{H/2+8}' x2='{margin}' y2='{H/2+8}' stroke='#ffffff' stroke-width='3'/>")
    # Right goal/boxes
    svg.append(f"<rect x='{W-margin-box_w}' y='{box_top}' width='{box_w}' height='{box_bot-box_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<rect x='{W-margin-six_w}' y='{six_top}' width='{six_w}' height='{six_bot-six_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<line x1='{W-margin}' y1='{H/2-8}' x2='{W-margin+goal_depth}' y2='{H/2-8}' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<line x1='{W-margin}' y1='{H/2+8}' x2='{W-margin+goal_depth}' y2='{H/2+8}' stroke='#ffffff' stroke-width='3'/>")

    # Players
    svg += _layout_side(a_rows, pa, True,  W, H, margin)
    svg += _layout_side(b_rows, pb, False, W, H, margin)

    svg.append("</svg>")
    return "<div class='pitchWrap'>" + "".join(svg) + "</div>"

def render_match_pitch(a_rows: pd.DataFrame, b_rows: pd.DataFrame,
                       formation_a: str, formation_b: str,
                       motm_name: Optional[str] = None,
                       height: int = 520):
    inner = render_pitch_svg(a_rows, b_rows, formation_a, formation_b, motm_name)
    html_component(
        "<html><head><meta name='viewport' content='width=device-width, initial-scale=1, maximum-scale=1'/>"
        "<style>html,body{margin:0;padding:0;background:transparent}</style></head>"
        "<body>"+inner+"</body></html>",
        height=height, scrolling=False
    )

# -----------------------------------------------------------------------------
# Write Ops
# -----------------------------------------------------------------------------
def upsert_match(row: Dict):
    _service().table("matches").upsert(row, on_conflict="id").execute()

def delete_lineups(match_id: str, team: Optional[str] = None):
    q = _service().table("lineups").delete().eq("match_id", match_id)
    if team:
        q = q.eq("team", team)
    q.execute()

def insert_lineups(rows: List[Dict]):
    if not rows: return
    batched = []
    for r in rows:
        rr = dict(r)
        rr["id"] = rr.get("id") or str(uuid.uuid4())
        rr["is_gk"] = bool(rr.get("is_gk", False))
        for c in ["line","slot","goals","assists","season","gw"]:
            rr[c] = _int(rr.get(c), 0)
        batched.append(rr)
    for i in range(0, len(batched), 500):
        _service().table("lineups").insert(batched[i:i+500]).execute()

# -----------------------------------------------------------------------------
# Fact table (robust name normalization; no DB 'name' dependency)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=120)
def build_fact(players: pd.DataFrame, matches: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    if lineups.empty or matches.empty:
        return pd.DataFrame()

    l = lineups.copy()
    if "player_name" not in l.columns: l["player_name"] = ""
    l["player_name"] = l["player_name"].fillna("").astype(str).str.strip()
    l["name"] = l["player_name"].map(normalize_name)

    for c in ["season","gw","goals","assists","line","slot"]:
        if c in l.columns:
            l[c] = pd.to_numeric(l[c], errors="coerce").fillna(0).astype(int)
        else:
            l[c] = 0

    m = matches.copy()
    for c in ["season","gw","score_a","score_b"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0).astype(int)

    mm = m.rename(columns={"id":"match_id"})[
        ["match_id","season","gw","team_a","team_b","score_a","score_b","is_draw","motm_name","formation_a","formation_b"]
    ]
    fact = l.merge(mm, on=["match_id","season","gw"], how="left")

    def wdl(r):
        if bool(r.get("is_draw", False)):
            return "D"
        ta, tb = int(r.get("score_a",0)), int(r.get("score_b",0))
        return "W" if ((r.get("team")=="Non-bibs" and ta>tb) or (r.get("team")=="Bibs" and tb>ta)) else "L"
    fact["result"] = fact.apply(wdl, axis=1)

    tg = fact.groupby(["match_id","team"], as_index=False).agg(team_goals=("goals","sum"))
    fact = fact.merge(tg, on=["match_id","team"], how="left")
    fact["ga"] = fact["goals"] + fact["assists"]
    fact["contrib_pct"] = (fact["ga"] / fact["team_goals"].replace(0, pd.NA) * 100).fillna(0.0).round(1)

    fact = fact[fact["name"].astype(str).str.strip() != ""].copy()
    return fact

# -----------------------------------------------------------------------------
# Header / Admin
# -----------------------------------------------------------------------------
def header():
    st.markdown("<div class='topbar'><span class='brand'>Powerleague <small>stats</small></span></div>", unsafe_allow_html=True)
    with st.expander("Admin"):
        if st.session_state.get("is_admin"):
            st.success("Logged in")
            if st.button("Logout"):
                st.session_state["is_admin"] = False
                st.rerun()
        else:
            pw = st.text_input("Password", type="password")
            if st.button("Login"):
                if pw == ADMIN_PASSWORD and pw:
                    st.session_state["is_admin"] = True
                    st.rerun()
                else:
                    st.error("Wrong password.")

# -----------------------------------------------------------------------------
# Add / Edit Match helpers (slot editor UX)
# -----------------------------------------------------------------------------
def _uuid_match(season: int, gw: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pl-match-{season}-{gw}"))

def _squad_picker(label: str, players: pd.DataFrame, key: str, prefill: List[str]) -> List[str]:
    names = players["name"].dropna().astype(str).map(normalize_name).sort_values().tolist()
    pre = [n for n in prefill if n in names]
    return st.multiselect(label, names, default=pre, key=key)

def _slot_table(team_label: str, formation: str, squad: List[str], current_rows: pd.DataFrame, keypref: str):
    parts = parts_of(formation)
    slots: List[Tuple[int,int,str]] = [(0,0,"GK")]
    for li, cnt in enumerate(parts, start=1):
        for sj in range(cnt):
            slots.append((li, sj, f"L{li} S{sj+1}"))

    # map existing assignments
    current_rows = current_rows.copy()
    current_rows["player_name"] = current_rows["player_name"].fillna("").astype(str)
    current_rows["name"] = current_rows["player_name"].map(normalize_name)
    cur: Dict[Tuple[int,int], str] = {}
    for _, r in current_rows.iterrows():
        nm = str(r.get("name") or "")
        if r.get("is_gk"):
            cur[(0,0)] = nm
        else:
            cur[(int(r.get("line") or 0), int(r.get("slot") or 0))] = nm

    chosen: Dict[Tuple[int,int], str] = {}
    goals_map: Dict[str, int] = {}
    assists_map: Dict[str, int] = {}

    for line, slot, label in slots:
        nm_list = ["—"] + squad
        cur_nm = cur.get((line, slot), "")
        sel = st.selectbox(f"{team_label} • {label}", nm_list,
                           index=(nm_list.index(cur_nm) if cur_nm in nm_list else 0),
                           key=f"{keypref}_slot_{line}_{slot}")
        nm = "" if sel == "—" else sel
        chosen[(line, slot)] = nm
        gc = st.number_input(f"{label} goals", min_value=0, max_value=99, value=0, step=1, key=f"{keypref}_g_{line}_{slot}")
        ac = st.number_input(f"{label} assists", min_value=0, max_value=99, value=0, step=1, key=f"{keypref}_a_{line}_{slot}")
        if nm:
            goals_map[nm] = gc
            assists_map[nm] = ac
    return chosen, goals_map, assists_map

def _save_team_lineup(match_id: str, team_label: str, season: int, gw: int,
                      chosen: Dict[Tuple[int,int], str], goals_map: Dict[str,int], assists_map: Dict[str,int]):
    rows: List[Dict] = []
    for (li, sj), nm in chosen.items():
        if not nm: continue
        rows.append({
            "id": str(uuid.uuid4()), "season": season, "gw": gw,
            "match_id": match_id, "team": team_label,
            "player_id": None, "player_name": nm,
            "is_gk": (li==0 and sj==0),
            "goals": int(goals_map.get(nm, 0)), "assists": int(assists_map.get(nm, 0)),
            "line": int(li), "slot": int(sj), "position": ""
        })
    delete_lineups(match_id, team_label)
    insert_lineups(rows)

# -----------------------------------------------------------------------------
# Page: Matches
# -----------------------------------------------------------------------------
def page_matches():
    header()

    # One-time Data Doctor (normalize player_name aliases)
    if st.session_state.get("is_admin"):
        with st.expander("🩺 Data Doctor: Normalize player names in lineups"):
            st.write("This will apply aliases (Ani→Anirudh, Abdullah Y13→Mohammad Abdullah) to `player_name`.")
            if st.button("Repair player_name"):
                lfix = load_table("lineups")
                if lfix.empty:
                    st.info("No lineups to fix.")
                else:
                    lfix["player_name"] = lfix["player_name"].fillna("").astype(str).str.strip().map(normalize_name)
                    svc = _service()
                    updates = [{"id": rid, "player_name": nm} for rid, nm in zip(lfix["id"], lfix["player_name"])]
                    for i in range(0, len(updates), 500):
                        svc.table("lineups").upsert(updates[i:i+500], on_conflict="id").execute()
                    st.success("player_name normalized. Refreshing…")
                    refresh_all()

    matches = load_table("matches")
    lineups = load_table("lineups")
    players = load_table("players")
    players["name"] = players["name"].fillna("").astype(str).map(normalize_name)

    # Add New Match (admin)
    if st.session_state.get("is_admin"):
        with st.expander("➕ Add new match"):
            with st.form("add_match_form"):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    season = st.number_input("Season", 2000, 2100, date.today().year, 1)
                with c2:
                    gw = st.number_input("Gameweek", 1, 1000, 1, 1)
                with c3:
                    side_count = st.selectbox("Side count", [5,7], index=0)
                with c4:
                    dt = st.date_input("Date (optional)", value=date.today())

                f1, f2 = st.columns(2)
                with f1:
                    fa = st.selectbox("Formation A", ["1-2-1","2-2","1-3","2-1-2-1","1-2-3","3-2-1"],
                                      index=(3 if int(side_count)==7 else 0))
                with f2:
                    fb = st.selectbox("Formation B", ["1-2-1","2-2","1-3","2-1-2-1","1-2-3","3-2-1"],
                                      index=(3 if int(side_count)==7 else 0))

                preA, preB = [], []
                if not lineups.empty and not matches.empty:
                    last = matches.sort_values(["season","gw"]).tail(1)
                    if not last.empty:
                        mid_last = last.iloc[0]["id"]
                        preA = lineups[(lineups["match_id"]==mid_last) & (lineups["team"]=="Non-bibs")]["player_name"].dropna().astype(str).map(normalize_name).tolist()
                        preB = lineups[(lineups["match_id"]==mid_last) & (lineups["team"]=="Bibs")]["player_name"].dropna().astype(str).map(normalize_name).tolist()
                squadA = _squad_picker("Non-bibs squad", players, "new_sqA", preA)
                squadB = _squad_picker("Bibs squad", players, "new_sqB", preB)

                all_candidates = sorted(set(squadA + squadB))
                motm_name = st.selectbox("MOTM (optional)", ["—"] + all_candidates, index=0)
                notes = st.text_input("Notes (optional)")

                submitted = st.form_submit_button("Create match & seed slots")
                if submitted:
                    mid = _uuid_match(int(season), int(gw))
                    row = {
                        "id": mid, "season": int(season), "gw": int(gw),
                        "side_count": int(side_count),
                        "team_a": "Non-bibs", "team_b": "Bibs",
                        "score_a": 0, "score_b": 0,
                        "date": str(dt),
                        "motm_name": (None if motm_name=="—" else motm_name),
                        "is_draw": False,
                        "formation_a": fa, "formation_b": fb,
                        "notes": notes or ""
                    }
                    upsert_match(row)

                    # Seed lineups GK first then by formation order
                    def seed(team_label: str, squad: List[str], formation: str):
                        rows = []
                        parts = parts_of(formation)
                        if not squad: return
                        rows.append({
                            "id": str(uuid.uuid4()), "season": int(season), "gw": int(gw),
                            "match_id": mid, "team": team_label, "player_id": None,
                            "player_name": squad[0], "is_gk": True, "goals": 0, "assists": 0,
                            "line": 0, "slot": 0, "position": ""
                        })
                        idx = 1
                        for li, cnt in enumerate(parts, start=1):
                            for sj in range(cnt):
                                if idx >= len(squad): break
                                rows.append({
                                    "id": str(uuid.uuid4()), "season": int(season), "gw": int(gw),
                                    "match_id": mid, "team": team_label, "player_id": None,
                                    "player_name": squad[idx], "is_gk": False, "goals": 0, "assists": 0,
                                    "line": li, "slot": sj, "position": ""
                                })
                                idx += 1
                        insert_lineups(rows)

                    seed("Non-bibs", squadA, fa)
                    seed("Bibs", squadB, fb)
                    st.success("Match created and slots seeded. Scroll down to edit.")
                    load_table.clear()
                    st.experimental_rerun()

    # Select match
    matches = load_table("matches")
    if matches.empty:
        st.info("No matches yet.")
        return
    matches["_label"] = matches.apply(lambda r: f"S{int(r.get('season') or 0)} – GW{int(r.get('gw') or 0)}", axis=1)
    matches = matches.sort_values(["season","gw"])
    sel = st.selectbox("Select match", matches["_label"].tolist(), key="sel_match")
    m = matches[matches["_label"]==sel].iloc[0]
    mid = m["id"]

    # Summary header
    cL, cR = st.columns([1,1])
    with cL:
        st.subheader(f"{m.get('team_a','Non-bibs')} {int(m.get('score_a') or 0)} – {int(m.get('score_b') or 0)} {m.get('team_b','Bibs')}")
        if str(m.get("motm_name") or "").strip():
            st.caption(f"⭐ MOTM: {m['motm_name']}")
        st.caption(f"{m.get('formation_a') or default_formation(m.get('side_count'))} vs {m.get('formation_b') or default_formation(m.get('side_count'))}")
        if str(m.get("notes") or "").strip():
            st.write(m["notes"])
    with cR:
        if st.session_state.get("is_admin"):
            sa = st.number_input("Score A", 0, 99, int(m.get("score_a") or 0), 1)
            sb = st.number_input("Score B", 0, 99, int(m.get("score_b") or 0), 1)
            motm = st.text_input("MOTM", value=str(m.get("motm_name") or ""))
            if st.button("Save summary"):
                upsert_match({"id": mid, "score_a": int(sa), "score_b": int(sb), "motm_name": motm})
                st.success("Saved.")
                refresh_all()

    # Load lineups for pitch
    all_lineups = load_table("lineups")
    a_rows = all_lineups[(all_lineups["match_id"]==mid) & (all_lineups["team"]=="Non-bibs")].copy()
    b_rows = all_lineups[(all_lineups["match_id"]==mid) & (all_lineups["team"]=="Bibs")].copy()
    for df in (a_rows, b_rows):
        for c in ["player_name","goals","assists","is_gk","line","slot"]:
            if c not in df.columns: df[c] = None
        df["player_name"] = df["player_name"].fillna("").astype(str)
        for c in ["goals","assists","line","slot"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        df["is_gk"] = df["is_gk"].fillna(False).astype(bool)

    fa = m.get("formation_a") or default_formation(m.get("side_count"))
    fb = m.get("formation_b") or default_formation(m.get("side_count"))
    render_match_pitch(a_rows, b_rows, fa, fb, m.get("motm_name"), height=520)

    st.subheader("Edit lineups & stats (Admin)")
    if not st.session_state.get("is_admin"):
        st.info("Login as admin to edit.")
        return

    players = load_table("players")
    players["name"] = players["name"].fillna("").astype(str).map(normalize_name)

    squadA_prefill = a_rows["player_name"].dropna().astype(str).map(normalize_name).tolist()
    squadB_prefill = b_rows["player_name"].dropna().astype(str).map(normalize_name).tolist()
    squadA = _squad_picker("Non-bibs squad", players, f"squadA_{mid}", squadA_prefill)
    squadB = _squad_picker("Bibs squad", players, f"squadB_{mid}", squadB_prefill)

    c1, c2 = st.columns(2)
    with c1:
        formation_a = st.selectbox("Formation A", ["1-2-1","2-2","1-3","2-1-2-1","1-2-3","3-2-1"],
                                   index=(3 if int(m.get("side_count") or 5)==7 else 0),
                                   key=f"fa_{mid}")
    with c2:
        formation_b = st.selectbox("Formation B", ["1-2-1","2-2","1-3","2-1-2-1","1-2-3","3-2-1"],
                                   index=(3 if int(m.get("side_count") or 5)==7 else 0),
                                   key=f"fb_{mid}")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Non-bibs slots**")
        chosenA, gA, aA = _slot_table("Non-bibs", formation_a, squadA, a_rows, f"A_{mid}")
        if st.button("Save Non-bibs", key=f"saveA_{mid}"):
            _save_team_lineup(mid, "Non-bibs", int(m.get("season") or 0), int(m.get("gw") or 0), chosenA, gA, aA)
            upsert_match({"id": mid, "formation_a": formation_a})
            st.success("Saved Non-bibs")
            refresh_all()
    with colB:
        st.markdown("**Bibs slots**")
        chosenB, gB, aB = _slot_table("Bibs", formation_b, squadB, b_rows, f"B_{mid}")
        if st.button("Save Bibs", key=f"saveB_{mid}"):
            _save_team_lineup(mid, "Bibs", int(m.get("season") or 0), int(m.get("gw") or 0), chosenB, gB, aB)
            upsert_match({"id": mid, "formation_b": formation_b})
            st.success("Saved Bibs")
            refresh_all()

# -----------------------------------------------------------------------------
# Stats helpers (filters & tables)
# -----------------------------------------------------------------------------
def _apply_filters(f: pd.DataFrame, season_sel: int, last_n: int) -> pd.DataFrame:
    df = f.copy()
    if season_sel != -1:
        df = df[df["season"]==season_sel]
    if last_n and "gw" in df.columns and not df.empty:
        maxgw = int(df["gw"].max())
        df = df[df["gw"] >= max(1, maxgw - int(last_n) + 1)]
    return df

def _player_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    agg = df.groupby("name", as_index=False).agg(
        GP=("match_id","nunique"),
        W=("result", lambda s: (s=="W").sum()),
        D=("result", lambda s: (s=="D").sum()),
        L=("result", lambda s: (s=="L").sum()),
        Goals=("goals","sum"),
        Assists=("assists","sum"),
        GA=("ga","sum"),
        ContribPct=("contrib_pct","mean"),
    )
    agg["Win%"]     = (agg["W"]/agg["GP"]).fillna(0)*100
    agg["G+A / G"]  = (agg["GA"]/agg["GP"]).replace([pd.NA,float("inf")], 0).fillna(0).round(2)
    agg["Goals/G"]  = (agg["Goals"]/agg["GP"]).replace([pd.NA,float("inf")], 0).fillna(0).round(2)
    agg["Assists/G"]= (agg["Assists"]/agg["GP"]).replace([pd.NA,float("inf")], 0).fillna(0).round(2)
    agg["Contrib%"] = agg["ContribPct"].round(1)
    return agg[["name","GP","W","D","L","Win%","Goals","Assists","GA","G+A / G","Goals/G","Assists/G","Contrib%"]]

def leaderboard(df: pd.DataFrame, metric: str, min_games: int, top_n: int):
    if df.empty:
        st.info("No data.")
        return
    t = _player_table(df)
    t = t[t["GP"] >= int(min_games)]
    col = {"G+A":"GA","Goals":"Goals","Assists":"Assists","Win %":"Win%","Contribution %":"Contrib%"}[metric]
    t = t.sort_values([col,"GA"], ascending=[False,False]).head(int(top_n))
    st.dataframe(hide_index(t.reset_index(drop=True)), use_container_width=True)

def duos_table(df: pd.DataFrame, min_games: int, top_n: int, player_filter: Optional[str]=None):
    if df.empty: st.info("No data."); return
    rows = []
    for (mid, team), grp in df.groupby(["match_id","team"]):
        names = grp["name"].dropna().unique().tolist()
        res = grp["result"].iloc[0] if not grp.empty else "D"
        if player_filter and player_filter not in names: 
            continue
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                n1, n2 = names[i], names[j]
                if player_filter and player_filter not in (n1, n2):
                    continue
                sub = grp.set_index("name").loc[[n1,n2]]
                ga = int(sub["goals"].sum() + sub["assists"].sum())
                rows.append({"pair":" & ".join(sorted([n1,n2])), "mid":mid, "team":team, "ga":ga, "win":int(res=="W")})
    if not rows: st.info("No pairs."); return
    d = pd.DataFrame(rows).groupby("pair", as_index=False).agg(
        GP=("mid","nunique"),
        GA=("ga","sum"),
        Wins=("win","sum"),
        WinPct=("win", lambda s: s.mean()*100.0)
    )
    d = d[d["GP"] >= int(min_games)].sort_values(["WinPct","GA","GP"], ascending=[False,False,False]).head(int(top_n))
    st.dataframe(hide_index(d.reset_index(drop=True)), use_container_width=True)

def nemesis_table(df: pd.DataFrame, min_games: int, top_n: int, player_filter: Optional[str]=None):
    if df.empty: st.info("No data."); return
    rows = []
    for mid, grp in df.groupby("match_id"):
        A = grp[grp["team"]=="Non-bibs"]["name"].dropna().unique().tolist()
        B = grp[grp["team"]=="Bibs"]["name"].dropna().unique().tolist()
        resA = grp[grp["team"]=="Non-bibs"]["result"].iloc[0] if not grp[grp["team"]=="Non-bibs"].empty else "D"
        resB = grp[grp["team"]=="Bibs"]["result"].iloc[0] if not grp[grp["team"]=="Bibs"].empty else "D"
        if player_filter:
            if player_filter in A:
                for y in B:
                    rows.append({"p":player_filter,"opp":y,"win":int(resA=="W"),"mid":mid})
            if player_filter in B:
                for y in A:
                    rows.append({"p":player_filter,"opp":y,"win":int(resB=="W"),"mid":mid})
        else:
            for x in A:
                for y in B:
                    rows.append({"p":x,"opp":y,"win":int(resA=="W"),"mid":mid})
            for x in B:
                for y in A:
                    rows.append({"p":x,"opp":y,"win":int(resB=="W"),"mid":mid})
    if not rows: st.info("No rows."); return
    d = pd.DataFrame(rows).groupby(["p","opp"], as_index=False).agg(GP=("mid","nunique"), Wins=("win","sum"), WinPct=("win", lambda s: s.mean()*100.0))
    if player_filter:
        d = d[d["p"]==player_filter]
    d = d[d["GP"] >= int(min_games)].sort_values(["WinPct","GP"], ascending=[True,False]).head(int(top_n))
    if d.empty: st.info("No nemesis after filter."); return
    st.dataframe(hide_index(d.reset_index(drop=True).rename(columns={"p":"Player","opp":"Nemesis"})), use_container_width=True)

# -----------------------------------------------------------------------------
# Ratings helpers (Finishing / Playmaking / Impact / Overall)
# -----------------------------------------------------------------------------
def _percentile(series: pd.Series) -> pd.Series:
    if series.empty: return series
    # rank pct in 0..100
    return series.rank(pct=True) * 100.0

def _scale_50_95(percentile_series: pd.Series) -> pd.Series:
    # Map 0..100 → 50..95 for nicer, less “99 spam”
    return (50.0 + 0.45 * percentile_series).clip(50,95).round(0)

def compute_player_ratings(fact: pd.DataFrame, player: str, min_gp_for_ratings: int = 3) -> Dict[str, float]:
    """
    Finishing: percentile(Goals/G)
    Playmaking: percentile(Assists/G)
    Impact: 0.6*percentile(Win%) + 0.2*percentile(GP) + 0.2*percentile(GA/G)
    Overall: 0.4*Impact + 0.35*Finishing + 0.25*Playmaking
    All scaled to 50..95.
    """
    if fact.empty:
        return {"Finishing": 50.0, "Playmaking": 50.0, "Impact": 50.0, "Overall": 50.0}

    table = _player_table(fact)
    if table.empty:
        return {"Finishing": 50.0, "Playmaking": 50.0, "Impact": 50.0, "Overall": 50.0}

    # base measures
    table["GA/G"] = (table["GA"] / table["GP"]).replace([pd.NA, float("inf")], 0).fillna(0)
    table["WinPct"] = table["Win%"].fillna(0)

    # percentiles
    p_goals_g   = _percentile(table["Goals/G"]).fillna(0)
    p_assists_g = _percentile(table["Assists/G"]).fillna(0)
    p_ga_g      = _percentile(table["GA/G"]).fillna(0)
    p_win       = _percentile(table["WinPct"]).fillna(0)
    p_gp        = _percentile(table["GP"]).fillna(0)

    # scaled ratings
    finishing_all   = _scale_50_95(p_goals_g)
    playmaking_all  = _scale_50_95(p_assists_g)
    impact_pct      = 0.6*p_win + 0.2*p_gp + 0.2*p_ga_g
    impact_all      = _scale_50_95(impact_pct)

    table["Finishing"]  = finishing_all
    table["Playmaking"] = playmaking_all
    table["Impact"]     = impact_all
    table["Overall"]    = (0.4*table["Impact"] + 0.35*table["Finishing"] + 0.25*table["Playmaking"]).round(0).clip(50,95)

    # Enforce min GP for validity: if below threshold, nudge down slightly
    table.loc[table["GP"] < int(min_gp_for_ratings), ["Finishing","Playmaking","Impact","Overall"]] = \
        (table.loc[table["GP"] < int(min_gp_for_ratings), ["Finishing","Playmaking","Impact","Overall"]] * 0.95).round(0)

    row = table[table["name"]==player]
    if row.empty:
        return {"Finishing": 50.0, "Playmaking": 50.0, "Impact": 50.0, "Overall": 50.0}
    r = row.iloc[0]
    return {
        "Finishing":  float(r["Finishing"]),
        "Playmaking": float(r["Playmaking"]),
        "Impact":     float(r["Impact"]),
        "Overall":    float(r["Overall"]),
    }

# -----------------------------------------------------------------------------
# Page: Stats
# -----------------------------------------------------------------------------
def page_stats():
    header()
    players = load_table("players")
    matches = load_table("matches")
    lineups = load_table("lineups")
    fact = build_fact(players, matches, lineups)
    if fact.empty:
        st.info("No data.")
        return

    seasons = sorted([int(x) for x in fact["season"].dropna().unique().tolist()])
    seasons = [-1] + seasons if seasons else [-1]
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        sel_season = st.selectbox("Season (or All)", seasons, format_func=lambda x:"All" if x==-1 else x, index=len(seasons)-1)
    with c2:
        min_games = st.number_input("Min games", 0, 200, 2, 1)
    with c3:
        last_n = st.number_input("Last N GWs (0=all)", 0, 500, 5, 1)
    with c4:
        top_n = st.number_input("Rows", 5, 200, 20, 5)

    f = _apply_filters(fact, sel_season, last_n)

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Leaderboards","📊 Player Table","🤝 Duos","🧨 Nemesis"])
    with tab1:
        metric = st.selectbox("Metric", ["G+A","Goals","Assists","Win %","Contribution %"], index=0)
        leaderboard(f, metric, min_games, top_n)
    with tab2:
        t = _player_table(f)
        t = t[t["GP"] >= int(min_games)].sort_values(["GA","Goals"], ascending=[False,False]).reset_index(drop=True)
        st.dataframe(hide_index(t), use_container_width=True)
    with tab3:
        duos_table(f, max(1, int(min_games)), int(top_n))
    with tab4:
        nemesis_table(f, max(1, int(min_games)), top_n=10)

# -----------------------------------------------------------------------------
# Page: Players (premium profile)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Page: Players (premium profile) — REPLACE WHOLE FUNCTION
# -----------------------------------------------------------------------------
def page_players():
    header()
    players = load_table("players")
    matches = load_table("matches")
    lineups = load_table("lineups")
    fact = build_fact(players, matches, lineups)
    if players.empty:
        st.info("No players.")
        return

    names = players["name"].dropna().astype(str).map(normalize_name).sort_values().tolist()
    sel = st.selectbox("Player", names, key="p_sel")
    p = players[players["name"].map(normalize_name)==sel]
    p = p.iloc[0] if not p.empty else {"name": sel, "photo_url":"", "notes":""}

    # Top section: avatar + stat cards
    c1, c2 = st.columns([1,2])
    with c1:
        if p.get("photo_url"):
            st.image(p["photo_url"], width=220)
        else:
            st.markdown(
                f"<div class='card' style='width:220px;height:220px;border-radius:16px;"
                f"display:flex;align-items:center;justify-content:center;'>"
                f"<div style='font-size:48px;font-weight:800'>{name_initials(sel)}</div></div>",
                unsafe_allow_html=True
            )
        if str(p.get("notes") or "").strip():
            st.caption(p["notes"])

    with c2:
        mine = fact[fact["name"]==sel]
        if mine.empty:
            st.info("No appearances yet.")
            return

        gp = mine["match_id"].nunique()
        w = int((mine["result"]=="W").sum())
        d = int((mine["result"]=="D").sum())
        l = int((mine["result"]=="L").sum())
        g = int(mine["goals"].sum()); a = int(mine["assists"].sum()); ga = g+a
        winpct = (w/gp*100.0) if gp else 0.0
        contrib = float(mine["contrib_pct"].mean() or 0)

        # Stat cards row 1
        cc = st.columns(4)
        for label, val in [("GP",gp), ("W-D-L", f"{w}-{d}-{l}"), ("Win %", f"{winpct:.1f}%"), ("G + A", ga)]:
            cc[0].markdown(
                f"<div class='statCard'><div class='statLabel'>{label}</div>"
                f"<div class='statValue'>{val}</div></div>",
                unsafe_allow_html=True
            )
            cc = cc[1:] + cc[:1]

        # Stat cards row 2 (FIXED f-string here)
        cc = st.columns(4)
        for label, val in [
            ("Goals", g),
            ("Assists", a),
            ("Avg Contrib %", f"{contrib:.1f}%"),
            ("G+A / G", f"{(ga/gp if gp else 0):.2f}"),
        ]:
            cc[0].markdown(
                f"<div class='statCard'><div class='statLabel'>{label}</div>"
                f"<div class='statValue'>{val}</div></div>",
                unsafe_allow_html=True
            )
            cc = cc[1:] + cc[:1]

    # Ratings (Finishing / Playmaking / Impact / Overall)
    ratings = compute_player_ratings(fact, sel, min_gp_for_ratings=3)
    r1, r2, r3, r4 = st.columns(4)
    for c, (lab, key) in zip(
        [r1,r2,r3,r4],
        [("Finishing","Finishing"),("Playmaking","Playmaking"),("Impact","Impact"),("Overall","Overall")]
    ):
        c.markdown(
            f"<div class='statCard'><div class='statLabel'>{lab}</div>"
            f"<div class='statValue'>{int(ratings[key])}</div></div>",
            unsafe_allow_html=True
        )

    # Form & recent games
    N = st.number_input("Last N games (form)", 3, 30, 5, 1, key="pf_lastN")
    last = mine.sort_values(["season","gw"]).tail(int(N))
    form = "".join(last["result"].map({"W":"W","D":"D","L":"L"}).tolist())
    st.markdown(f"<div class='badge'>Recent form: <strong>{form}</strong></div>", unsafe_allow_html=True)

    st.markdown("#### Recent games")
    recent = mine.sort_values(["season","gw"], ascending=[False,False])[["season","gw","team","goals","assists","result"]].head(10)
    recent = recent.rename(columns={"season":"Season","gw":"GW","team":"Team","goals":"G","assists":"A","result":"Res"})
    st.dataframe(hide_index(recent.reset_index(drop=True)), use_container_width=True)

    # Player-specific Duos
    st.markdown("#### Teammate Duos")
    colD1, colD2 = st.columns(2)
    with colD1:
        min_joint = st.number_input("Min games together", 1, 50, 3, 1, key="duo_min")
    with colD2:
        rows = st.number_input("Rows", 3, 50, 5, 1, key="duo_rows")
    duos_table(fact, min_games=int(min_joint), top_n=int(rows), player_filter=sel)

    # Player-specific Nemesis
    st.markdown("#### Nemesis (most difficult opponents)")
    colN1, colN2 = st.columns(2)
    with colN1:
        min_meet = st.number_input("Min head-to-head", 1, 50, 3, 1, key="nem_min")
    with colN2:
        nrows = st.number_input("Rows ", 3, 50, 5, 1, key="nem_rows")
    nemesis_table(fact, min_games=int(min_meet), top_n=int(nrows), player_filter=sel)


# -----------------------------------------------------------------------------
# Page: Awards
# -----------------------------------------------------------------------------
def page_awards():
    header()
    matches = load_table("matches")
    awards = load_table("awards")

    st.subheader("MOTM (from matches)")
    if matches.empty:
        st.info("No matches.")
    else:
        motm = matches[["season","gw","motm_name"]].copy()
        motm = motm[motm["motm_name"].notna() & (motm["motm_name"].astype(str)!="")]
        if motm.empty: st.info("No MOTM.")
        else: st.dataframe(hide_index(motm.sort_values(["season","gw"]).reset_index(drop=True)), use_container_width=True)

    st.subheader("POTM (manual)")
    if not st.session_state.get("is_admin"):
        if not awards.empty:
            potm = awards[awards["type"]=="POTM"][["season","month","player_name","notes"]].sort_values(["season","month"])
            st.dataframe(hide_index(potm.reset_index(drop=True)), use_container_width=True)
        else:
            st.info("Login as admin to add POTM.")
        return

    c1,c2,c3 = st.columns(3)
    season = c1.number_input("Season", 2000, 2100, date.today().year, 1)
    month  = c2.selectbox("Month", list(range(1,13)), index=(date.today().month-1))
    plist = load_table("players")
    name   = c3.selectbox("Player", plist["name"].dropna().astype(str).map(normalize_name).sort_values().tolist())
    notes  = st.text_input("Notes (optional)")
    if st.button("Add POTM"):
        _service().table("awards").insert({
            "id": str(uuid.uuid4()), "season": int(season), "month": int(month),
            "type": "POTM", "gw": None,
            "player_id": None, "player_name": name, "notes": notes or ""
        }).execute()
        st.success("Added.")
        refresh_all()

    awards = load_table("awards")
    if not awards.empty:
        potm = awards[awards["type"]=="POTM"][["season","month","player_name","notes"]].sort_values(["season","month"])
        st.dataframe(hide_index(potm.reset_index(drop=True)), use_container_width=True)

# -----------------------------------------------------------------------------
# Page: Player Manager
# -----------------------------------------------------------------------------
def page_player_manager():
    header()
    if not st.session_state.get("is_admin"):
        st.info("Admin only.")
        return

    players = load_table("players")
    players["name"] = players["name"].fillna("").astype(str).map(normalize_name)

    c1, c2 = st.columns([2,1])
    with c1:
        sel = st.selectbox("Select player", players["name"].dropna().astype(str).sort_values().tolist() if not players.empty else [])
        current = players[players["name"]==sel].iloc[0] if (not players.empty and sel) else None
        new_name = st.text_input("Name", value=(current["name"] if current is not None else ""))
        notes = st.text_area("Notes", value=(current.get("notes","") if current is not None else ""), height=120)

        up = st.file_uploader("Avatar (JPG/PNG/HEIC)", type=["jpg","jpeg","png","heic","HEIC"])
        photo_url = current.get("photo_url","") if current is not None else ""
        if up is not None:
            img = None
            ext = (up.name.split(".")[-1] or "").lower()
            try:
                if ext == "heic" and _HEIC:
                    heif = pillow_heif.read_heif(up.read())  # type: ignore
                    img = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
                elif ext == "heic" and not _HEIC:
                    st.error("HEIC not supported on this host. Please upload JPG/PNG.")
                else:
                    img = Image.open(up).convert("RGB")
            except Exception as e:
                st.error(f"Image read failed: {e}")
            if img is not None:
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True); buf.seek(0)
                key = f"{uuid.uuid4().hex}.png"
                try:
                    _service().storage.from_(AVATAR_BUCKET).upload(key, buf, {"content-type":"image/png","x-upsert":"true"})
                    public = _service().storage.from_(AVATAR_BUCKET).get_public_url(key)
                    if isinstance(public, dict) and "publicUrl" in public: photo_url = public["publicUrl"]
                    elif isinstance(public, str): photo_url = public
                    st.image(photo_url, width=160, caption="Uploaded")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

        if st.button("Save player"):
            if current is not None:
                _service().table("players").update({"name": new_name.strip(), "notes": notes.strip(), "photo_url": photo_url.strip()}).eq("id", current["id"]).execute()
            else:
                _service().table("players").upsert({"name": new_name.strip(), "notes": notes.strip(), "photo_url": photo_url.strip()}, on_conflict="name").execute()
            st.success("Saved.")
            refresh_all()

    with c2:
        if current is not None:
            st.caption("Current")
            if current.get("photo_url"): st.image(current["photo_url"], width=140)
            else:
                st.markdown(f"<div class='card' style='width:140px;height:140px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:32px;font-weight:800'>{name_initials(current['name'])}</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
def run_app():
    st.session_state.setdefault("is_admin", False)
    page = st.sidebar.radio("Go to", ["Matches","Stats","Players","Awards","Player Manager"], index=0, key="nav")
    if page == "Matches": page_matches()
    elif page == "Stats": page_stats()
    elif page == "Players": page_players()
    elif page == "Awards": page_awards()
    elif page == "Player Manager": page_player_manager()

if __name__ == "__main__":
    run_app()
