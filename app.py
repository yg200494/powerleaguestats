import os, uuid, io
from datetime import date
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image
from supabase import create_client

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Powerleague Stats", layout="wide", initial_sidebar_state="collapsed")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
AVATAR_BUCKET = st.secrets.get("AVATAR_BUCKET", "avatars")

# =============================================================================
# SUPABASE
# =============================================================================
@st.cache_resource
def _client():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

@st.cache_resource
def _service():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# =============================================================================
# THEME / CSS (black+gold, mobile-first)
# =============================================================================
CSS = """
<style>
:root{
  --bg:#0b0f14; --panel:#0f1620; --muted:#18202b; --text:#e9edf3; --sub:#9aa6b2;
  --gold:#f6d35f; --teal:#57d2c8; --chip:#11161d; --border:#1a2230;
}
html,body,.stApp{background:var(--bg); color:var(--text); -webkit-text-size-adjust:100%}
.block-container{padding-top:.6rem; padding-bottom:2rem; max-width:1100px}
.topbar{position:sticky; top:0; z-index:50; background:var(--panel); border-bottom:1px solid var(--border);
  padding:10px 12px;}
.brand{font-weight:800; letter-spacing:.2px}
.brand small{font-weight:400; opacity:.75; margin-left:6px}
.pitchWrap{width:100%; max-width:980px; margin:0 auto}
.card{background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:14px}
.statCard{background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:14px; text-align:center}
.statLabel{font-size:12px; color:var(--sub)}
.statValue{font-size:20px; font-weight:800; color:var(--gold)}
.badge{display:inline-flex; gap:6px; align-items:center; font-size:12px; background:var(--chip);
  border:1px solid var(--border); border-radius:999px; padding:2px 8px}
.small{color:var(--sub); font-size:12px}
.stDataFrame, .stDataFrame div{color:var(--text) !important}
table td, table th{font-size:14px}
hr.sep{border:0;border-top:1px solid var(--border);margin:10px 0 16px 0}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# OPTIONAL HEIC SUPPORT
# =============================================================================
try:
    import pillow_heif  # type: ignore
    _HEIC = True
except Exception:
    pillow_heif = None
    _HEIC = False

# =============================================================================
# HELPERS
# =============================================================================
def normalize_name(n: str) -> str:
    return (n or "").strip()

def name_initials(name: str) -> str:
    parts = [p for p in (name or "").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()

@st.cache_data(ttl=60)
def load_table(name: str) -> pd.DataFrame:
    data = _client().table(name).select("*").execute().data
    return pd.DataFrame(data or [])

def refresh_all():
    st.cache_data.clear()

# ---------- Image / Storage ----------
def _png_from_uploaded_file(upfile) -> Optional[Image.Image]:
    try:
        filename = (getattr(upfile, "name", "") or "").lower()
        ext = filename.split(".")[-1] if "." in filename else ""
        if ext == "heic":
            if _HEIC and pillow_heif is not None:
                heif = pillow_heif.read_heif(upfile.read())
                return Image.frombytes(heif.mode, heif.size, heif.data, "raw").convert("RGB")
            else:
                st.error("HEIC not supported on this host. Please upload JPG/PNG.")
                return None
        return Image.open(upfile).convert("RGB")
    except Exception as e:
        st.error(f"Image read failed: {e}")
        return None

def _square_thumbnail(img: Image.Image, size: int = 420) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size))

def _storage_upload_png(bytes_data: bytes, key: str) -> str:
    storage = _service().storage.from_(AVATAR_BUCKET)
    try:
        storage.upload(key, bytes_data, {"content-type": "image/png", "x-upsert": "true"})
    except TypeError:
        storage.upload(path=key, file=bytes_data, file_options={"contentType": "image/png", "upsert": "true"})
    pub = storage.get_public_url(key)
    if isinstance(pub, dict) and "publicUrl" in pub:
        return pub["publicUrl"]
    return str(pub)

# ---------- Formation ----------
def default_formation_for(side_count: int) -> str:
    return "1-2-1" if side_count == 5 else "2-1-2-1"

PRESETS_5S = ["1-2-1", "2-1-1", "2-2", "1-3"]
PRESETS_7S = ["2-1-2-1", "3-2-1", "2-2-2", "1-3-2"]

def parts_of(formation: str) -> List[int]:
    try:
        return [int(x) for x in str(formation or "").split("-") if x.strip().isdigit()]
    except Exception:
        return [1,2,1]

def lineup_slots(formation: str) -> List[Tuple[int,int]]:
    slots = []
    for li, cnt in enumerate(parts_of(formation), start=1):
        for s in range(1, cnt+1):
            slots.append((li, s))
    return slots

# =============================================================================
# PITCH SVG (separate team pitch, large avatars, no overlap)
# =============================================================================
def _display_name(n: str, max_len=14) -> str:
    n = (n or "").strip()
    return n if len(n) <= max_len else (n[:max_len-1] + "…")

def svg_chip(text: str, accent="#f6d35f") -> str:
    w = max(28, 10 + 6*len(text))
    return (
        f"<g>"
        f"<rect rx='10' ry='10' x='{-(w//2)}' y='-11' width='{w}' height='22' "
        f"fill='#0e1319' stroke='#1b2430'/>"
        f"<text x='0' y='5' text-anchor='middle' font-size='12' fill='{accent}'>{text}</text>"
        f"</g>"
    )

def chip_goal(n: int) -> str:
    return svg_chip(f"{n}g", "#f6d35f")

def chip_assist(n: int) -> str:
    return svg_chip(f"{n}a", "#57d2c8")

def _player_node(x: float, y: float, name: str, goals: int, assists: int,
                 motm: bool, photo_url: str) -> List[str]:
    initials = name_initials(name)
    name_disp = _display_name(name, 14)
    r = 30  # bigger avatars
    chip_gap = 8
    clip_id = uuid.uuid4().hex

    def chips():
        g = chip_goal(goals) if goals>0 else ""
        a = chip_assist(assists) if assists>0 else ""
        if goals>0 and assists>0:
            return (
                f"<g transform='translate({x-26},{y+r+chip_gap})'>{g}</g>"
                f"<g transform='translate({x+26},{y+r+chip_gap})'>{a}</g>"
            )
        elif goals>0:
            return f"<g transform='translate({x},{y+r+chip_gap})'>{g}</g>"
        elif assists>0:
            return f"<g transform='translate({x},{y+r+chip_gap})'>{a}</g>"
        return ""

    star = (f"<circle cx='{x+20}' cy='{y-20}' r='10' fill='#f6d35f'/>"
            f"<text x='{x+20}' y='{y-16}' text-anchor='middle' font-size='12' font-weight='800' fill='#000'>★</text>"
            ) if motm else ""

    if photo_url:
        avatar = (
            f"<clipPath id='clip_{clip_id}'><circle cx='{x}' cy='{y}' r='{r}'/></clipPath>"
            f"<image href='{photo_url}' x='{x-r}' y='{y-r}' width='{2*r}' height='{2*r}' preserveAspectRatio='xMidYMid slice' clip-path='url(#clip_{clip_id})' />"
        )
    else:
        avatar = (
            f"<circle cx='{x}' cy='{y}' r='{r}' fill='#1a2230' stroke='#2a3647' stroke-width='1'/>"
            f"<text x='{x}' y='{y+7}' text-anchor='middle' font-size='16' font-weight='700' fill='#e7eaf0'>{initials}</text>"
        )

    name_text = f"<text x='{x}' y='{y+r+30}' text-anchor='middle' font-size='12' fill='#ffffff'>{name_disp}</text>"
    return [star, avatar, chips(), name_text]

def team_pitch_svg(rows: pd.DataFrame, formation: str, motm_name: Optional[str], left_side: bool) -> str:
    """Draw a full pitch but only populate one half; stacks well on mobile."""
    W, H = 940, 520
    margin = 28
    box_top = H*0.20
    box_bot = H*0.80
    six_top = H*0.32
    six_bot = H*0.68
    left_box_w = 120
    six_w = 55
    goal_depth = 14

    parts = parts_of(formation)

    pitch = []
    pitch.append(f"<svg viewBox='0 0 {W} {H}' width='100%' height='100%' preserveAspectRatio='xMidYMid meet'>")
    pitch.append("<defs>"
                 "<linearGradient id='g' x1='0' y1='0' x2='0' y2='1'>"
                 "<stop offset='0%' stop-color='#2f7a43'/>"
                 "<stop offset='60%' stop-color='#2a6f3c'/>"
                 "<stop offset='100%' stop-color='#235f34'/>"
                 "</linearGradient>"
                 "</defs>")
    pitch.append(f"<rect x='0' y='0' width='{W}' height='{H}' fill='url(#g)'/>")
    pitch.append(f"<rect x='{margin}' y='{margin}' width='{W-2*margin}' height='{H-2*margin}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<line x1='{W/2}' y1='{margin}' x2='{W/2}' y2='{H-margin}' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<circle cx='{W/2}' cy='{H/2}' r='65' fill='none' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<circle cx='{W/2}' cy='{H/2}' r='4' fill='#ffffff'/>")

    # Boxes
    # Left
    pitch.append(f"<rect x='{margin}' y='{box_top}' width='{left_box_w}' height='{box_bot - box_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<rect x='{margin}' y='{six_top}' width='{six_w}' height='{six_bot - six_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<line x1='{margin - goal_depth}' y1='{H/2 - 8}' x2='{margin}' y2='{H/2 - 8}' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<line x1='{margin - goal_depth}' y1='{H/2 + 8}' x2='{margin}' y2='{H/2 + 8}' stroke='#ffffff' stroke-width='3'/>")
    # Right
    pitch.append(f"<rect x='{W - margin - left_box_w}' y='{box_top}' width='{left_box_w}' height='{box_bot - box_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<rect x='{W - margin - six_w}' y='{six_top}' width='{six_w}' height='{six_bot - six_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<line x1='{W - margin}' y1='{H/2 - 8}' x2='{W - margin + goal_depth}' y2='{H/2 - 8}' stroke='#ffffff' stroke-width='3'/>")
    pitch.append(f"<line x1='{W - margin}' y1='{H/2 + 8}' x2='{W - margin + goal_depth}' y2='{H/2 + 8}' stroke='#ffffff' stroke-width='3'/>")

    # Layout only one half
    if rows is not None and not rows.empty:
        half_w = (W - 2*margin) / 2
        x_left = margin
        x_right = W - margin
        side_x0 = x_left if left_side else W/2
        side_x1 = W/2 if left_side else x_right
        usable_w = (side_x1 - side_x0)
        usable_h = (H - 2*margin)
        total_lines = len(parts) + 1  # GK + others
        y_step = usable_h / (total_lines + 1)

        # GK
        gk = rows[rows.get("is_gk", False) == True]
        if not gk.empty:
            gkr = gk.iloc[0]
            gx = side_x0 + (usable_w*0.12 if left_side else usable_w*0.88)
            gy = margin + y_step
            pitch += _player_node(
                gx, gy,
                str(gkr.get("name") or gkr.get("player_name") or ""),
                int(gkr.get("goals") or 0),
                int(gkr.get("assists") or 0),
                (motm_name or "") == (gkr.get("name") or gkr.get("player_name")),
                str(gkr.get("photo_url") or "")
            )

        # Other lines
        cur_y = margin + (2 * y_step)
        for li, cnt in enumerate(parts, start=1):
            for s in range(1, cnt+1):
                r = rows[(rows.get("line")==li) & (rows.get("slot")==s)]
                if r.empty:
                    continue
                rr = r.iloc[0]
                xgap = usable_w / (cnt + 1)
                x = side_x0 + xgap * s
                y = cur_y
                pitch += _player_node(
                    x, y,
                    str(rr.get("name") or rr.get("player_name") or ""),
                    int(rr.get("goals") or 0),
                    int(rr.get("assists") or 0),
                    (motm_name or "") == (rr.get("name") or rr.get("player_name")),
                    str(rr.get("photo_url") or "")
                )
            cur_y += y_step

    pitch.append("</svg>")
    return "<div class='pitchWrap'>" + "".join(pitch) + "</div>"

def render_team_pitch(rows: pd.DataFrame, formation: str, motm_name: Optional[str], left_side: bool, height: int = 520):
    inner = team_pitch_svg(rows, formation, motm_name, left_side)
    wrapper = (
        "<html><head><meta name='viewport' content='width=device-width, initial-scale=1, maximum-scale=1'/>"
        "<style>html,body{margin:0;padding:0;background:transparent}</style></head>"
        "<body>" + inner + "</body></html>"
    )
    st.components.v1.html(wrapper, height=height, scrolling=False)

# =============================================================================
# FACT / STATS
# =============================================================================
@st.cache_data(ttl=120)
def build_fact(players: pd.DataFrame, matches: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    if lineups.empty or matches.empty:
        return pd.DataFrame(columns=[
            "match_id","season","gw","team","name","goals","assists","result","contrib_pct","photo_url"
        ])

    l = lineups.copy()
    m = matches.copy()

    l["name"] = l.get("name", pd.Series(index=l.index)).fillna(l.get("player_name")).fillna("").astype(str).map(normalize_name)
    l["goals"] = pd.to_numeric(l.get("goals"), errors="coerce").fillna(0).astype(int)
    l["assists"] = pd.to_numeric(l.get("assists"), errors="coerce").fillna(0).astype(int)
    l["is_gk"] = l.get("is_gk", False).astype(bool)
    l["team"] = l.get("team").fillna("").astype(str)

    m["is_draw"] = m.get("is_draw", False).astype(bool)
    for c in ["season","gw","score_a","score_b"]:
        m[c] = pd.to_numeric(m.get(c), errors="coerce").astype("Int64")

    j = l.merge(
        m[["id","season","gw","team_a","team_b","score_a","score_b","is_draw","motm_name"]],
        left_on="match_id", right_on="id", how="left", suffixes=("","_m")
    )
    j.rename(columns={"id":"match_id"}, inplace=True)

    def row_result(row):
        if bool(row.get("is_draw")) or (
           pd.notna(row.get("score_a")) and pd.notna(row.get("score_b")) and int(row["score_a"]) == int(row["score_b"])
        ):
            return "D"
        if row["team"] == "Non-bibs":
            return "W" if int(row["score_a"]) > int(row["score_b"]) else "L"
        return "W" if int(row["score_b"]) > int(row["score_a"]) else "L"

    j["result"] = j.apply(row_result, axis=1)

    tg = j.groupby(["match_id","team"], as_index=False)["goals"].sum().rename(columns={"goals":"team_goals"})
    j = j.merge(tg, on=["match_id","team"], how="left")
    j["contrib_pct"] = ((j["goals"] + j["assists"]) / j["team_goals"].replace(0, pd.NA) * 100).round(1).fillna(0)

    if not players.empty and {"name","photo_url"}.issubset(set(players.columns)):
        pp = players[["name","photo_url"]].copy()
        pp["name"] = pp["name"].astype(str).map(normalize_name)
        j = j.merge(pp, on="name", how="left")
    else:
        j["photo_url"] = ""

    return j

def _scale_0_99(series, floor=35, ceil=95):
    if len(series) == 0 or series.max() == series.min():
        return pd.Series([50]*len(series), index=series.index if hasattr(series, "index") else None)
    z = (series - series.min()) / (series.max() - series.min())
    return (floor + z * (ceil - floor)).clip(floor, ceil).round()

def compute_player_ratings(fact: pd.DataFrame, name: str, min_gp_for_ratings=3):
    mine = fact[fact["name"]==name]
    if mine["match_id"].nunique() < min_gp_for_ratings:
        return {"Finishing": 50, "Playmaking": 50, "Impact": 50, "Overall": 50}

    peers = fact.copy()
    gpg = mine.groupby("match_id")["goals"].sum().mean()
    apg = mine.groupby("match_id")["assists"].sum().mean()
    contrib = mine["contrib_pct"].mean()

    peer_gpg = peers.groupby(["name","match_id"])["goals"].sum().groupby("name").mean()
    peer_apg = peers.groupby(["name","match_id"])["assists"].sum().groupby("name").mean()

    fin = int(_scale_0_99(pd.Series([gpg, *peer_gpg.values]))[0])
    ply = int(_scale_0_99(pd.Series([apg, *peer_apg.values]))[0])

    team_avg_win = peers.groupby("name")["result"].apply(lambda s: (s=="W").mean()).mean()
    my_win = (mine["result"]=="W").mean()
    impact_raw = 0.6*max(0, my_win - float(team_avg_win)) + 0.4*(contrib/100.0)
    impact = int((35 + impact_raw*120))
    impact = max(35, min(95, impact))

    overall = int(round(0.4*fin + 0.4*ply + 0.2*impact))
    return {"Finishing": fin, "Playmaking": ply, "Impact": impact, "Overall": overall}

def teammates_duos_for(fact: pd.DataFrame, player: str, min_games=3, top_n=5) -> pd.DataFrame:
    mine = fact[fact["name"]==player][["match_id","team"]].drop_duplicates()
    if mine.empty: return pd.DataFrame()
    j = fact.merge(mine, on=["match_id","team"], how="inner")
    j = j[j["name"] != player]
    grp = j.groupby("name").agg(GP=("match_id","nunique"), Win=("result", lambda s: (s=="W").sum()))
    grp["Win%"] = (grp["Win"] / grp["GP"] * 100).round(1)
    grp = grp[grp["GP"] >= int(min_games)].sort_values(["Win%","GP"], ascending=[False, False]).head(int(top_n))
    grp = grp.reset_index().rename(columns={"name":"Teammate"})
    return grp[["Teammate","GP","Win%"]]

def nemesis_for(fact: pd.DataFrame, player: str, min_games=3, top_n=5) -> pd.DataFrame:
    mine = fact[fact["name"]==player][["match_id","team"]].drop_duplicates()
    if mine.empty: return pd.DataFrame()
    opp = fact.merge(mine, on="match_id", suffixes=("","_mine"))
    opp = opp[opp["team"] != opp["team_mine"]]
    grp = opp.groupby("name").agg(GP=("match_id","nunique"), Win=("result", lambda s: (s=="W").sum()))
    grp["Win%"] = (grp["Win"] / grp["GP"] * 100).round(1)
    grp = grp[grp["GP"] >= int(min_games)].sort_values(["Win%","GP"], ascending=[True, False]).head(int(top_n))
    grp = grp.reset_index().rename(columns={"name":"Opponent"})
    return grp[["Opponent","GP","Win%"]]

# =============================================================================
# HEADER (render once; unique keys)
# =============================================================================
def header(key_prefix: str = "hdr"):
    top = st.container()
    with top:
        c1, c2, c3 = st.columns([3,2,2])
        c1.markdown("<div class='brand'>Powerleague Stats <small>beta</small></div>", unsafe_allow_html=True)
        with c2:
            if "is_admin" not in st.session_state:
                st.session_state["is_admin"] = False
            if not st.session_state["is_admin"]:
                pw = st.text_input("Admin password",
                                   type="password",
                                   key=f"{key_prefix}_adm_pw",
                                   label_visibility="collapsed",
                                   placeholder="Admin password")
                if st.button("Login", key=f"{key_prefix}_adm_login"):
                    if (pw or "") == ADMIN_PASSWORD:
                        st.session_state["is_admin"] = True
                        st.success("Admin mode on.")
                        st.rerun()
                    else:
                        st.error("Wrong password.")
            else:
                st.success("Admin mode")
        with c3:
            if st.session_state.get("is_admin"):
                if st.button("Logout", key=f"{key_prefix}_adm_logout"):
                    st.session_state["is_admin"] = False
                    st.rerun()

# =============================================================================
# LINEUP EDITOR (tap-assign; mobile-first)
# =============================================================================
def _team_names_from(df_team: pd.DataFrame) -> List[str]:
    if df_team is None or df_team.empty:
        return []
    return (
        df_team.get("name").fillna(df_team.get("player_name")).dropna().astype(str)
        .map(normalize_name).unique().tolist()
    )

def team_quick_assign(team_label: str,
                      formation: str,
                      df_team: pd.DataFrame,
                      team_player_names: List[str],
                      keypref: str):
    cur = {}
    if df_team is not None and not df_team.empty:
        for _, r in df_team.iterrows():
            li = int(r.get("line") or 0); s = int(r.get("slot") or 0)
            nm = str(r.get("name") or r.get("player_name") or "").strip()
            if li > 0 and s > 0 and nm:
                cur[(li, s)] = nm

    st.markdown(f"**{team_label}**  ·  Formation **{formation}**")
    grid = []
    for li, cnt in enumerate(parts_of(formation), start=1):
        cols = st.columns(cnt, vertical_alignment="center")
        for s in range(1, cnt+1):
            slot_key = f"{keypref}_slot_{li}_{s}"
            current = cur.get((li, s), "— empty —")
            used = set(cur.values())
            choices = ["— empty —"] + [n for n in team_player_names if (n == current or n not in used)]
            with cols[s-1]:
                st.caption(f"Line {li} • Slot {s}")
                choice = st.selectbox("", choices,
                                      index=(choices.index(current) if current in choices else 0),
                                      key=slot_key,
                                      label_visibility="collapsed")
                grid.append(((li, s), None if choice == "— empty —" else choice))

    c1, c2, c3 = st.columns(3)
    clicked = {"auto": False, "reset": False, "save": False}
    with c1:
        if st.button("Auto-fill", key=f"{keypref}_auto"):
            clicked["auto"] = True
    with c2:
        if st.button("Reset team", key=f"{keypref}_reset"):
            clicked["reset"] = True
    with c3:
        if st.button("Save team", type="primary", key=f"{keypref}_save"):
            clicked["save"] = True

    return grid, clicked

def apply_autofill(grid, team_names, formation):
    target = lineup_slots(formation)
    assigned = {nm for _, nm in grid if nm}
    pool = [n for n in team_names if n not in assigned]
    out = []
    p_i = 0
    for (li, s) in target:
        cur = next((nm for (L,S), nm in grid if L==li and S==s and nm), None)
        if cur:
            out.append(((li,s), cur))
        else:
            nm = pool[p_i] if p_i < len(pool) else None
            out.append(((li,s), nm))
            if nm: p_i += 1
    return out

def save_team_grid(match_id, season, gw, team_label, formation, grid, df_team):
    svc = _service()
    sample = {}
    if df_team is not None and not df_team.empty:
        for _, r in df_team.iterrows():
            nm = normalize_name(str(r.get("name") or r.get("player_name") or ""))
            if nm and nm not in sample:
                sample[nm] = r
    rows = []
    for (li, s), nm in grid:
        if not nm: continue
        base = sample.get(normalize_name(nm), {})
        rows.append({
            "id": str(uuid.uuid4()),
            "season": int(season) if pd.notna(season) else None,
            "gw": int(gw) if pd.notna(gw) else None,
            "match_id": str(match_id),
            "team": team_label,
            "player_id": base.get("player_id"),
            "player_name": nm,
            "name": nm,
            "is_gk": bool(base.get("is_gk") or False),
            "goals": int(base.get("goals") or 0),
            "assists": int(base.get("assists") or 0),
            "line": int(li),
            "slot": int(s),
            "position": ""
        })
    # Persist
    svc.table("lineups").delete().eq("match_id", str(match_id)).eq("team", team_label).execute()
    for i in range(0, len(rows), 500):
        svc.table("lineups").insert(rows[i:i+500]).execute()
    # Save formation field
    col = "formation_a" if team_label == "Non-bibs" else "formation_b"
    svc.table("matches").update({col: formation}).eq("id", str(match_id)).execute()
    return True

# =============================================================================
# PAGES
# =============================================================================
def page_matches():
    matches = load_table("matches")
    players = load_table("players")
    lineups = load_table("lineups")

    if matches.empty:
        st.info("No matches yet.")
        return

    opts = matches.sort_values(["season","gw"]).apply(
        lambda r: (r["id"], f"S{int(r['season'])} · GW{int(r['gw'])} · {r.get('team_a','Non-bibs')} {int(r.get('score_a') or 0)}–{int(r.get('score_b') or 0)} {r.get('team_b','Bibs')}"),
        axis=1
    ).tolist()
    sel = st.selectbox("Select match", opts, format_func=lambda t: t[1], index=len(opts)-1, key="match_sel")
    mid = sel[0]
    m = matches[matches["id"]==mid].iloc[0]

    # Summary header
    st.markdown(
        f"<div class='card'><b>Season {int(m['season'])} · GW{int(m['gw'])}</b>"
        f"<div class='small'>Score: {m.get('team_a','Non-bibs')} {int(m.get('score_a') or 0)} – {int(m.get('score_b') or 0)} {m.get('team_b','Bibs')}</div>"
        f"<div class='small'>MOTM: <span class='badge'>{m.get('motm_name') or '-'}</span></div>"
        f"</div>", unsafe_allow_html=True
    )

    # Formations (5s vs 7s presets)
    side_count = int(m.get("side_count") or 5)
    presets = PRESETS_5S if side_count == 5 else PRESETS_7S
    cfa, cfb = st.columns(2)
    with cfa:
        fa = st.selectbox("Formation (Non-bibs)", presets,
                          index=(presets.index(m.get("formation_a")) if m.get("formation_a") in presets else 0),
                          key=f"fa_{mid}")
    with cfb:
        fb = st.selectbox("Formation (Bibs)", presets,
                          index=(presets.index(m.get("formation_b")) if m.get("formation_b") in presets else 0),
                          key=f"fb_{mid}")

    # Lineups for this match
    a_rows = lineups[(lineups["match_id"]==mid) & (lineups["team"]=="Non-bibs")].copy()
    b_rows = lineups[(lineups["match_id"]==mid) & (lineups["team"]=="Bibs")].copy()

    # Two separate pitches (stack on phone, side-by-side on desktop)
    pa, pb = st.columns(2)
    with pa:
        st.subheader("Non-bibs")
        render_team_pitch(a_rows, fa, m.get("motm_name"), left_side=True, height=520)
    with pb:
        st.subheader("Bibs")
        render_team_pitch(b_rows, fb, m.get("motm_name"), left_side=False, height=520)

    # Admin: arrange lineups
    if st.session_state.get("is_admin"):
        st.markdown("### Arrange lineups")
        a_names = _team_names_from(a_rows)
        b_names = _team_names_from(b_rows)
        colA, colB = st.columns(2)
        with colA:
            gridA, clickA = team_quick_assign("Non-bibs", fa, a_rows, a_names, keypref=f"A_{mid}")
            if clickA["auto"]:
                gridA = apply_autofill(gridA, a_names, fa); st.rerun()
            if clickA["reset"]:
                _service().table("lineups").delete().eq("match_id", str(mid)).eq("team", "Non-bibs").execute(); st.rerun()
            if clickA["save"]:
                if save_team_grid(mid, m["season"], m["gw"], "Non-bibs", fa, gridA, a_rows):
                    st.success("Non-bibs saved."); refresh_all(); st.rerun()
        with colB:
            gridB, clickB = team_quick_assign("Bibs", fb, b_rows, b_names, keypref=f"B_{mid}")
            if clickB["auto"]:
                gridB = apply_autofill(gridB, b_names, fb); st.rerun()
            if clickB["reset"]:
                _service().table("lineups").delete().eq("match_id", str(mid)).eq("team", "Bibs").execute(); st.rerun()
            if clickB["save"]:
                if save_team_grid(mid, m["season"], m["gw"], "Bibs", fb, gridB, b_rows):
                    st.success("Bibs saved."); refresh_all(); st.rerun()

        # Quick goals/assists editor
        st.markdown("### Edit goals / assists")
        def _edit_team(df_team: pd.DataFrame, label: str):
            if df_team.empty:
                st.info(f"No {label} lineup yet.")
                return
            temp = df_team[["id","name","goals","assists"]].fillna({"goals":0,"assists":0}).copy()
            temp["goals"] = temp["goals"].astype(int)
            temp["assists"] = temp["assists"].astype(int)
            temp = temp.rename(columns={"name":"Player","goals":"G","assists":"A"})
            st.dataframe(temp[["Player","G","A"]], use_container_width=True, hide_index=True)
            with st.form(f"ga_{label}_{mid}"):
                p = st.selectbox("Player", temp["Player"].tolist(), key=f"ga_sel_{label}_{mid}")
                g = st.number_input("Goals", 0, 50, int(temp[temp["Player"]==p]["G"].iloc[0]) if not temp.empty else 0, key=f"ga_g_{label}_{mid}")
                a = st.number_input("Assists", 0, 50, int(temp[temp["Player"]==p]["A"].iloc[0]) if not temp.empty else 0, key=f"ga_a_{label}_{mid}")
                ok = st.form_submit_button("Save")
                if ok:
                    rid = df_team[df_team["name"]==p]["id"].iloc[0]
                    _service().table("lineups").update({"goals": int(g), "assists": int(a)}).eq("id", rid).execute()
                    st.success("Updated."); refresh_all(); st.rerun()
        c1, c2 = st.columns(2)
        with c1: _edit_team(a_rows, "A")
        with c2: _edit_team(b_rows, "B")

def page_stats():
    players = load_table("players")
    matches = load_table("matches")
    lineups = load_table("lineups")
    fact = build_fact(players, matches, lineups)
    if fact.empty:
        st.info("No stats yet."); return

    seasons = sorted([int(s) for s in fact["season"].dropna().unique().tolist()])
    seasons = ["All"] + seasons
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_season = st.selectbox("Season", seasons, index=0)
    with c2:
        min_gp = st.number_input("Min games", 1, 50, 3, 1)
    with c3:
        last_n = st.number_input("Last N (0=all)", 0, 50, 0, 1)
    with c4:
        metric = st.selectbox("Metric", ["G+A", "Goals", "Assists", "Contribution%", "Win%", "MOTM"])

    df = fact.copy()
    if sel_season != "All":
        df = df[df["season"] == int(sel_season)]
    if last_n and last_n > 0:
        df = df.sort_values(["season","gw"]).groupby("name").tail(int(last_n))

    agg = df.groupby("name").agg(
        GP=("match_id","nunique"),
        W=("result", lambda s: (s=="W").sum()),
        D=("result", lambda s: (s=="D").sum()),
        L=("result", lambda s: (s=="L").sum()),
        goals=("goals","sum"),
        assists=("assists","sum"),
        contrib=("contrib_pct","mean")
    ).reset_index()
    agg["Win%"] = (agg["W"] / agg["GP"] * 100).round(1)
    agg["G+A"] = agg["goals"] + agg["assists"]
    agg["Contribution%"] = agg["contrib"].round(1)
    agg = agg.drop(columns=["contrib"])
    sort_by = {"G+A":"G+A","Goals":"goals","Assists":"assists","Contribution%":"Contribution%","Win%":"Win%","MOTM":"G+A"}[metric]
    agg = agg[agg["GP"] >= int(min_gp)].sort_values([sort_by,"GP"], ascending=[False, False])

    nice = agg.rename(columns={"name":"Player","goals":"G","assists":"A"})
    st.dataframe(nice[["Player","GP","W","D","L","Win%","G","A","G+A","Contribution%"]], use_container_width=True, hide_index=True)

def page_players():
    players = load_table("players")
    matches = load_table("matches")
    lineups = load_table("lineups")
    fact = build_fact(players, matches, lineups)
    if players.empty:
        st.info("No players."); return

    names = players["name"].dropna().astype(str).map(normalize_name).sort_values().tolist()
    sel = st.selectbox("Player", names, key="p_sel")
    p = players[players["name"].map(normalize_name)==sel]
    p = p.iloc[0] if not p.empty else {"name": sel, "photo_url":"", "notes":""}

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
            st.info("No appearances yet."); return

        gp = mine["match_id"].nunique()
        w = int((mine["result"]=="W").sum())
        d = int((mine["result"]=="D").sum())
        l = int((mine["result"]=="L").sum())
        g = int(mine["goals"].sum()); a = int(mine["assists"].sum()); ga = g+a
        winpct = (w/gp*100.0) if gp else 0.0
        contrib = float(mine["contrib_pct"].mean() or 0)

        cc = st.columns(4)
        cc[0].markdown(f"<div class='statCard'><div class='statLabel'>GP</div><div class='statValue'>{gp}</div></div>", unsafe_allow_html=True)
        cc[1].markdown(f"<div class='statCard'><div class='statLabel'>W-D-L</div><div class='statValue'>{w}-{d}-{l}</div></div>", unsafe_allow_html=True)
        cc[2].markdown(f"<div class='statCard'><div class='statLabel'>Win %</div><div class='statValue'>{winpct:.1f}%</div></div>", unsafe_allow_html=True)
        cc[3].markdown(f"<div class='statCard'><div class='statLabel'>G + A</div><div class='statValue'>{ga}</div></div>", unsafe_allow_html=True)

        cc2 = st.columns(4)
        cc2[0].markdown(f"<div class='statCard'><div class='statLabel'>Goals</div><div class='statValue'>{g}</div></div>", unsafe_allow_html=True)
        cc2[1].markdown(f"<div class='statCard'><div class='statLabel'>Assists</div><div class='statValue'>{a}</div></div>", unsafe_allow_html=True)
        cc2[2].markdown(f"<div class='statCard'><div class='statLabel'>Avg Contrib %</div><div class='statValue'>{contrib:.1f}%</div></div>", unsafe_allow_html=True)
        cc2[3].markdown(f"<div class='statCard'><div class='statLabel'>G+A / G</div><div class='statValue'>{(ga/gp if gp else 0):.2f}</div></div>", unsafe_allow_html=True)

    ratings = compute_player_ratings(fact, sel, min_gp_for_ratings=3)
    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(f"<div class='statCard'><div class='statLabel'>Finishing</div><div class='statValue'>{int(ratings['Finishing'])}</div></div>", unsafe_allow_html=True)
    r2.markdown(f"<div class='statCard'><div class='statLabel'>Playmaking</div><div class='statValue'>{int(ratings['Playmaking'])}</div></div>", unsafe_allow_html=True)
    r3.markdown(f"<div class='statCard'><div class='statLabel'>Impact</div><div class='statValue'>{int(ratings['Impact'])}</div></div>", unsafe_allow_html=True)
    r4.markdown(f"<div class='statCard'><div class='statLabel'>Overall</div><div class='statValue'>{int(ratings['Overall'])}</div></div>", unsafe_allow_html=True)

    st.markdown("#### Recent games / form")
    N = st.number_input("Last N games", 3, 30, 5, 1, key="pf_lastN")
    last = mine.sort_values(["season","gw"]).tail(int(N))
    form = "".join(last["result"].map({"W":"W","D":"D","L":"L"}).tolist())
    st.markdown(f"<div class='badge'>Recent form: <strong>{form}</strong></div>", unsafe_allow_html=True)
    recent = mine.sort_values(["season","gw"], ascending=[False,False])[["season","gw","team","goals","assists","result"]].head(10)
    recent = recent.rename(columns={"season":"Season","gw":"GW","team":"Team","goals":"G","assists":"A","result":"Res"})
    st.dataframe(recent.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("#### Teammate Duos")
    colD1, colD2 = st.columns(2)
    with colD1: min_joint = st.number_input("Min games together", 1, 50, 3, 1, key="duo_min")
    with colD2: rows = st.number_input("Rows", 3, 50, 5, 1, key="duo_rows")
    duos = teammates_duos_for(fact, sel, int(min_joint), int(rows))
    if not duos.empty: st.dataframe(duos, use_container_width=True, hide_index=True)
    else: st.info("No duo data yet.")

    st.markdown("#### Nemesis (toughest opponents)")
    colN1, colN2 = st.columns(2)
    with colN1: min_meet = st.number_input("Min head-to-head", 1, 50, 3, 1, key="nem_min")
    with colN2: nrows = st.number_input("Rows ", 3, 50, 5, 1, key="nem_rows")
    nem = nemesis_for(fact, sel, int(min_meet), int(nrows))
    if not nem.empty: st.dataframe(nem, use_container_width=True, hide_index=True)
    else: st.info("No nemesis data yet.")

def page_awards():
    matches = load_table("matches")
    awards = load_table("awards")

    st.markdown("### Man of the Match (by GW)")
    if matches.empty:
        st.info("No matches.")
    else:
        mm = matches[['season','gw','motm_name']].dropna().sort_values(["season","gw"])
        mm = mm.rename(columns={"motm_name":"MOTM"})
        st.dataframe(mm, use_container_width=True, hide_index=True)

    st.markdown("### Player of the Month")
    if awards.empty:
        st.info("No awards recorded.")
    else:
        potm = awards[awards["type"]=="POTM"][["season","month","player_name","notes"]].rename(
            columns={"player_name":"POTM"}
        ).sort_values(["season","month"])
        st.dataframe(potm, use_container_width=True, hide_index=True)

def page_player_manager():
    if not st.session_state.get("is_admin"):
        st.info("Admin only."); return

    players = load_table("players")
    if not players.empty:
        players["name"] = players["name"].fillna("").astype(str).map(normalize_name)

    left, right = st.columns([2, 1])
    with left:
        options = ["➕ Add new player"] + (players["name"].dropna().astype(str).sort_values().tolist() if not players.empty else [])
        sel = st.selectbox("Select player", options, key="pm_select")

        current = None if sel == "➕ Add new player" or players.empty else players.loc[players["name"] == sel].iloc[0]

        new_name = st.text_input("Name", value=(current["name"] if current is not None else ""), key="pm_name")
        notes = st.text_area("Notes", value=(current.get("notes", "") if current is not None else ""), height=120, key="pm_notes")

        photo_url = (current.get("photo_url") or "") if current is not None else ""

        up = st.file_uploader("Avatar (JPG/PNG/HEIC)", type=["jpg","jpeg","png","heic","HEIC"], key="pm_avatar")
        if up is not None:
            img = _png_from_uploaded_file(up)
            if img is not None:
                img = _square_thumbnail(img, size=420)
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                key = f"{uuid.uuid4().hex}.png"
                try:
                    photo_url = _storage_upload_png(buf.getvalue(), key)
                    st.image(photo_url, width=160, caption="Uploaded")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

        colA, colB, _ = st.columns([1,1,2])
        with colA:
            if st.button("Remove photo", key="pm_remove_photo"):
                photo_url = ""
        with colB:
            if st.button("Save player", type="primary", key="pm_save"):
                payload = {"name": (new_name or "").strip(),
                           "notes": (notes or "").strip(),
                           "photo_url": (photo_url or "").strip()}
                try:
                    if current is not None:
                        _service().table("players").update(payload).eq("id", current["id"]).execute()
                    else:
                        _service().table("players").upsert(payload, on_conflict="name").execute()
                    st.success("Saved."); refresh_all(); st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")

    with right:
        st.caption("Preview")
        if (photo_url or ""):
            st.image(photo_url, width=140)
        else:
            initials = name_initials((new_name or (sel if sel != "➕ Add new player" else "")))
            st.markdown(
                f"<div class='card' style='width:140px;height:140px;border-radius:14px;"
                f"display:flex;align-items:center;justify-content:center;font-size:32px;font-weight:800'>{initials}</div>",
                unsafe_allow_html=True,
            )
        st.write("")
        st.markdown(f"<div class='small'>Name:</div><div class='badge'>{(new_name or sel).strip()}</div>", unsafe_allow_html=True)
        if (notes or "").strip():
            st.write("")
            st.markdown(f"<div class='small'>Notes:</div><div class='card'>{notes.strip()}</div>", unsafe_allow_html=True)

# =============================================================================
# ROUTER (header called ONCE)
# =============================================================================
def run_app():
    header("hdr")  # render once to avoid duplicate widget IDs
    pages = ["Matches", "Stats", "Players", "Awards", "Player Manager"]
    page = st.sidebar.radio("Go to", pages, index=0)
    if page == "Matches": page_matches()
    elif page == "Stats": page_stats()
    elif page == "Players": page_players()
    elif page == "Awards": page_awards()
    elif page == "Player Manager": page_player_manager()

if __name__ == "__main__":
    run_app()
