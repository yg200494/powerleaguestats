# Powerleague Stats – streamlined, mobile-first
# Streamlit + Supabase, premium lineup SVG, robust stats

import os, io, uuid, base64
from typing import List, Optional, Tuple
import streamlit as st
import pandas as pd
from supabase import create_client
from datetime import date

# ==============================
# Config
# ==============================
st.set_page_config(page_title="Powerleague Stats", layout="wide", initial_sidebar_state="collapsed")

SUPABASE_URL         = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY    = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]
ADMIN_PASSWORD       = st.secrets.get("ADMIN_PASSWORD", "")
AVATAR_BUCKET        = st.secrets.get("AVATAR_BUCKET", "avatars")

sb      = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)      # public (read)
sb_admin= create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)   # service (write)

# ==============================
# CSS (premium, phone-friendly)
# ==============================
st.markdown("""
<style>
:root{
  --bg:#0b0f14; --panel:#0f1620; --muted:#18202b;
  --text:#e9edf3; --sub:#9aa6b2; --gold:#f6d35f; --aqua:#57d2c8;
  --chip:#11161d; --border:#1a2230; --accent:#1e2735;
}
html,body,.stApp{
  background:var(--bg); color:var(--text);
  -webkit-text-size-adjust:100%;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
  touch-action:manipulation;
}
.block-container{
  padding-top:.6rem; padding-bottom:2rem; max-width:1200px;
  padding-left:clamp(.6rem, 2.5vw, 1rem);
  padding-right:clamp(.6rem, 2.5vw, 1rem);
}
/* Notch-friendly sticky header */
.topbar{
  position:sticky; top:0; z-index:50;
  background:var(--panel); border-bottom:1px solid var(--border);
  padding:calc(8px + env(safe-area-inset-top)) 12px 10px 12px;
}
.brand{font-weight:800; letter-spacing:.2px}
.brand small{font-weight:400; opacity:.75; margin-left:6px}
.card{
  background:var(--panel); border:1px solid var(--border);
  border-radius:14px; padding:14px;
}
.badge{
  display:inline-flex; gap:6px; align-items:center;
  font-size:12px; background:var(--chip); color:var(--text);
  border:1px solid var(--border); border-radius:999px; padding:2px 8px;
}
.small{color:var(--sub); font-size:12px}
hr.sep{border:0;border-top:1px solid var(--border);margin:10px 0 16px}
.pitchWrap{ width:100%; max-width:980px; margin:0 auto }
.pitchWrap svg{ max-width:100%; max-height:620px; height:auto; display:block }
.dualPitch{ display:grid; gap:16px; grid-template-columns:1fr }
@media (min-width: 900px){ .dualPitch{ grid-template-columns:1fr 1fr } }
.stButton > button{
  background:linear-gradient(180deg, #1a2533 0%, #0f1620 100%);
  color:var(--text); border:1px solid var(--border);
  border-radius:10px; padding:.5rem .9rem; font-weight:600;
}
.stButton > button:hover{ border-color:#2a3647 }
[data-testid="stTabs"] div[role="tablist"] { gap: 6px; }
/* Mobile tighten */
@media (max-width: 600px){
  .pitchWrap svg{ max-height:520px }
}
@media (max-width: 420px){
  .pitchWrap svg{ max-height:460px }
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Utilities
# ==============================
def normalize_name(n: Optional[str]) -> str:
    return " ".join((n or "").strip().split())

def name_initials(n: str) -> str:
    parts = [p for p in normalize_name(n).split(" ") if p]
    if not parts: return "?"
    return (parts[0][0] + (parts[1][0] if len(parts)>1 else "")).upper()

@st.cache_data(ttl=60)
def load_table(name: str) -> pd.DataFrame:
    try:
        data = sb.table(name).select("*").execute().data
        return pd.DataFrame(data or [])
    except Exception:
        return pd.DataFrame([])

def normalize_matches(m: pd.DataFrame) -> pd.DataFrame:
    if m is None or m.empty:
        return pd.DataFrame(columns=["id","season","gw","team_a","team_b","score_a","score_b","date","motm_name","is_draw","formation_a","formation_b","side_count"])
    out = m.copy()
    out["id"]        = out.get("id").astype(str)
    out["season"]    = pd.to_numeric(out.get("season"), errors="coerce").astype("Int64")
    out["gw"]        = pd.to_numeric(out.get("gw"), errors="coerce").astype("Int64")
    out["score_a"]   = pd.to_numeric(out.get("score_a"), errors="coerce").fillna(0).astype(int)
    out["score_b"]   = pd.to_numeric(out.get("score_b"), errors="coerce").fillna(0).astype(int)
    out["is_draw"]   = out.get("is_draw", False).astype(bool)
    out["team_a"]    = out.get("team_a","Non-bibs")
    out["team_b"]    = out.get("team_b","Bibs")
    out["motm_name"] = out.get("motm_name","")
    out["formation_a"]= out.get("formation_a").fillna("")
    out["formation_b"]= out.get("formation_b").fillna("")
    out["side_count"]= pd.to_numeric(out.get("side_count"), errors="coerce").fillna(5).astype(int)
    return out

def normalize_lineups(l: pd.DataFrame) -> pd.DataFrame:
    if l is None or l.empty:
        return pd.DataFrame(columns=["match_id","team","name","player_name","is_gk","goals","assists","line","slot","photo_url"])
    out = l.copy()
    # accept either "name" or "player_name"
    out["name"]   = out.get("name", pd.Series(index=out.index)).fillna(out.get("player_name")).fillna("").astype(str).map(normalize_name)
    out["team"]   = out.get("team").fillna("").astype(str)
    out["match_id"] = out.get("match_id").astype(str)
    out["goals"]  = pd.to_numeric(out.get("goals"), errors="coerce").fillna(0).astype(int)
    out["assists"]= pd.to_numeric(out.get("assists"), errors="coerce").fillna(0).astype(int)
    out["line"]   = pd.to_numeric(out.get("line"), errors="coerce").fillna(0).astype(int)
    out["slot"]   = pd.to_numeric(out.get("slot"), errors="coerce").fillna(0).astype(int)
    out["is_gk"]  = out.get("is_gk", False)
    if out["is_gk"].dtype != bool:
        out["is_gk"] = out["is_gk"].astype(str).str.lower().isin(["1","true","t","yes","y"])
    if "photo_url" not in out.columns: out["photo_url"] = ""
    return out

# HEIC support
try:
    import pillow_heif
    from PIL import Image
    _HEIC=True
except Exception:
    from PIL import Image
    _HEIC=False

def _png_from_uploaded_file(up) -> Optional[Image.Image]:
    if up is None: return None
    name = up.name or "image"
    ext = (name.split(".")[-1] or "").lower()
    try:
        if ext in ("heic","HEIC"):
            if not _HEIC: return None
            heif = pillow_heif.read_heif(up.read())  # type: ignore
            return Image.frombytes(heif.mode, heif.size, heif.data, "raw").convert("RGB")
        else:
            return Image.open(up).convert("RGB")
    except Exception:
        return None

def _service(): return sb_admin

# ==============================
# Formation helpers + premium SVG
# ==============================
def parts_of(formation: str) -> List[int]:
    f = (formation or "").strip()
    if not f: return [1,2,1]  # default 5s
    try:
        nums = [int(x) for x in f.split("-") if str(x).strip().isdigit()]
        return nums if nums else [1,2,1]
    except Exception:
        return [1,2,1]

def lineup_slots(formation: str) -> List[Tuple[int,int]]:
    parts = parts_of(formation)
    slots = []
    for li, cnt in enumerate(parts, start=1):
        for s in range(1, cnt+1):
            slots.append((li,s))
    return slots

def ensure_positions(df_team: pd.DataFrame, formation: str) -> pd.DataFrame:
    """Ensure GK at (line=0,slot=1) and others fill formation order without collisions."""
    if df_team is None or df_team.empty:
        return pd.DataFrame(columns=["name","is_gk","goals","assists","line","slot","photo_url"])
    out = df_team.copy()
    out = normalize_lineups(out)
    # Build available slots
    slots = [(0,1)] + lineup_slots(formation)
    taken = {(int(r["line"]), int(r["slot"])) for _, r in out.iterrows() if r["line"]>0 and r["slot"]>0}
    gk_idx = out.index[out["is_gk"]==True].tolist()
    if gk_idx:
        gi = gk_idx[0]
        out.at[gi,"line"]=0; out.at[gi,"slot"]=1
    # Fill non-gk
    ptr=0
    ordered = [s for s in slots if s != (0,1)]
    for idx, r in out.iterrows():
        if r["is_gk"]: continue
        li, s = int(r["line"]), int(r["slot"])
        if li>0 and s>0 and (li,s) not in taken:
            taken.add((li,s))
            continue
        while ptr < len(ordered) and ordered[ptr] in taken:
            ptr += 1
        if ptr < len(ordered):
            li2, s2 = ordered[ptr]; ptr += 1
            out.at[idx,"line"]=li2; out.at[idx,"slot"]=s2
            taken.add((li2,s2))
        else:
            out.at[idx,"line"]=len(parts_of(formation)); out.at[idx,"slot"]=99
    return out

def _display_name(n: str, max_len=16) -> str:
    n = normalize_name(n)
    return n if len(n)<=max_len else (n[:max_len-1]+"…")

def _chip(text: str, accent="#f6d35f") -> str:
    w = max(26, 10 + 7*len(text))
    return (
        f"<g><rect rx='9' ry='9' x='{-(w//2)}' y='-11' width='{w}' height='22' fill='#0e1319' stroke='#1b2430'/>"
        f"<text x='0' y='5' text-anchor='middle' font-size='11' fill='{accent}'>{text}</text></g>"
    )
def chip_goal(n:int)->str:   return _chip(f"{n}g", "#f6d35f")
def chip_assist(n:int)->str: return _chip(f"{n}a", "#57d2c8")

def _player_node(x: float, y: float, name: str, goals: int, assists: int,
                 motm: bool, photo_url: str, r: int=34) -> List[str]:
    initials = name_initials(name)
    name_disp = _display_name(name, 16)
    clip_id = uuid.uuid4().hex
    chips = ""
    if goals>0 and assists>0:
        chips = (
            f"<g transform='translate({x-28},{y+r+12})'>{chip_goal(goals)}</g>"
            f"<g transform='translate({x+28},{y+r+12})'>{chip_assist(assists)}</g>"
        )
    elif goals>0:
        chips = f"<g transform='translate({x},{y+r+12})'>{chip_goal(goals)}</g>"
    elif assists>0:
        chips = f"<g transform='translate({x},{y+r+12})'>{chip_assist(assists)}</g>"
    star = (f"<circle cx='{x+24}' cy='{y-24}' r='12' fill='#f6d35f'/>"
            f"<text x='{x+24}' y='{y-19}' text-anchor='middle' font-size='12' font-weight='800' fill='#000'>★</text>") if motm else ""
    if photo_url:
        avatar = (
            f"<clipPath id='clip_{clip_id}'><circle cx='{x}' cy='{y}' r='{r}'/></clipPath>"
            f"<image href='{photo_url}' x='{x-r}' y='{y-r}' width='{2*r}' height='{2*r}' "
            f"preserveAspectRatio='xMidYMid slice' clip-path='url(#clip_{clip_id})' />"
            f"<circle cx='{x}' cy='{y}' r='{r}' fill='none' stroke='#2a3647' stroke-width='1'/>"
        )
    else:
        avatar = (
            f"<circle cx='{x}' cy='{y}' r='{r}' fill='#1a2230' stroke='#2a3647' stroke-width='1'/>"
            f"<text x='{x}' y='{y+8}' text-anchor='middle' font-size='18' font-weight='700' fill='#e7eaf0'>{initials}</text>"
        )
    label = f"<text x='{x}' y='{y+r+38}' text-anchor='middle' font-size='12' fill='#ffffff'>{name_disp}</text>"
    return [star, avatar, chips, label]

def _half_pitch_svg(rows: pd.DataFrame, formation: str, motm_name: str, left_side: bool) -> str:
    # Dimensions & lines
    rows = ensure_positions(rows, formation)
    W, H = 980, 620
    margin = 32
    box_top = H*0.20; box_bot = H*0.80
    six_top = H*0.32; six_bot = H*0.68
    left_box_w = 120; six_w = 55; goal_depth = 14

    parts = parts_of(formation)
    motm = (motm_name or "").strip().lower()

    svg = []
    svg.append(f"<svg viewBox='0 0 {W} {H}' width='100%' height='100%' preserveAspectRatio='xMidYMid meet'>")
    svg.append("<defs><linearGradient id='g' x1='0' y1='0' x2='0' y2='1'>"
               "<stop offset='0%' stop-color='#2f7a43'/>"
               "<stop offset='60%' stop-color='#2a6f3c'/>"
               "<stop offset='100%' stop-color='#235f34'/>"
               "</linearGradient></defs>")
    # Pitch
    svg.append(f"<rect x='0' y='0' width='{W}' height='{H}' fill='url(#g)'/>")
    svg.append(f"<rect x='{margin}' y='{margin}' width='{W-2*margin}' height='{H-2*margin}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<line x1='{W/2}' y1='{margin}' x2='{W/2}' y2='{H-margin}' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<circle cx='{W/2}' cy='{H/2}' r='65' fill='none' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<circle cx='{W/2}' cy='{H/2}' r='4' fill='#ffffff'/>")
    # Boxes
    svg.append(f"<rect x='{margin}' y='{box_top}' width='{left_box_w}' height='{box_bot-box_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<rect x='{margin}' y='{six_top}' width='{six_w}' height='{six_bot-six_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<line x1='{margin-goal_depth}' y1='{H/2-8}' x2='{margin}' y2='{H/2-8}' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<line x1='{margin-goal_depth}' y1='{H/2+8}' x2='{margin}' y2='{H/2+8}' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<rect x='{W-margin-left_box_w}' y='{box_top}' width='{left_box_w}' height='{box_bot-box_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<rect x='{W-margin-six_w}' y='{six_top}' width='{six_w}' height='{six_bot-six_top}' fill='none' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<line x1='{W-margin}' y1='{H/2-8}' x2='{W-margin+goal_depth}' y2='{H/2-8}' stroke='#ffffff' stroke-width='3'/>")
    svg.append(f"<line x1='{W-margin}' y1='{H/2+8}' x2='{W-margin+goal_depth}' y2='{H/2+8}' stroke='#ffffff' stroke-width='3'/>")

    # Coordinates
    usable_w = (W - 2*margin) / 2
    side_x0  = margin if left_side else (W/2)
    side_x1  = (W/2) if left_side else (W - margin)
    total_lines = len(parts) + 1
    y_step = (H - 2*margin) / (total_lines + 1)

    # GK (line 0)
    gk = rows[rows["is_gk"]==True]
    if not gk.empty:
        r = gk.iloc[0]
        gx = side_x0 + (usable_w*0.12 if left_side else usable_w*0.88)
        gy = margin + y_step
        svg += _player_node(gx, gy, r["name"], int(r["goals"]), int(r["assists"]), normalize_name(r["name"]).lower()==motm, str(r.get("photo_url") or ""), r=38)

    # Outfield
    cur_y = margin + (2*y_step)
    for li, cnt in enumerate(parts, start=1):
        xgap = (side_x1 - side_x0) / (cnt + 1)
        for s in range(1, cnt+1):
            rr = rows[(rows["line"]==li) & (rows["slot"]==s)]
            if rr.empty: continue
            row = rr.iloc[0]
            x = side_x0 + xgap * s
            y = cur_y
            svg += _player_node(x, y, row["name"], int(row["goals"]), int(row["assists"]),
                                normalize_name(row["name"]).lower()==motm, str(row.get("photo_url") or ""))
        cur_y += y_step

    svg.append("</svg>")
    return "<div class='pitchWrap'>" + "".join(svg) + "</div>"

def render_team_pitch(rows: pd.DataFrame, formation: str, motm_name: str, left_side: bool, height: int = 620):
    inner = _half_pitch_svg(rows, formation, motm_name, left_side)
    wrapper = ("<html><head><meta name='viewport' content='width=device-width, initial-scale=1, maximum-scale=1'/>"
               "<style>html,body{margin:0;padding:0;background:transparent}</style></head>"
               "<body>" + inner + "</body></html>")
    st.components.v1.html(wrapper, height=height, scrolling=False)

# ==============================
# Stats builder (defensive)
# ==============================
@st.cache_data(ttl=120)
def build_fact(players: pd.DataFrame, matches: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    l = normalize_lineups(lineups)
    m = normalize_matches(matches)

    base_cols = ["match_id","season","gw","team","name","goals","assists","result","contrib_pct","photo_url"]
    if l.empty or m.empty:
        return pd.DataFrame(columns=base_cols)

    j = l.merge(
        m[["id","season","gw","team_a","team_b","score_a","score_b","is_draw","motm_name"]],
        left_on="match_id", right_on="id", how="left"
    ).rename(columns={"id":"match_id"})

    # safe types
    for c in ["match_id","team","name"]:
        j[c] = j.get(c, pd.Series(index=j.index)).astype(str)
    j["season"]   = pd.to_numeric(j.get("season"), errors="coerce").astype("Int64")
    j["gw"]       = pd.to_numeric(j.get("gw"), errors="coerce").astype("Int64")
    j["goals"]    = pd.to_numeric(j.get("goals"), errors="coerce").fillna(0).astype(int)
    j["assists"]  = pd.to_numeric(j.get("assists"), errors="coerce").fillna(0).astype(int)
    j["score_a"]  = pd.to_numeric(j.get("score_a"), errors="coerce").fillna(0).astype(int)
    j["score_b"]  = pd.to_numeric(j.get("score_b"), errors="coerce").fillna(0).astype(int)
    j["is_draw"]  = j.get("is_draw", False).astype(bool)

    def row_result(row):
        if row["is_draw"] or row["score_a"]==row["score_b"]:
            return "D"
        if row["team"] == "Non-bibs":
            return "W" if row["score_a"] > row["score_b"] else "L"
        return "W" if row["score_b"] > row["score_a"] else "L"

    j["result"] = j.apply(row_result, axis=1)

    tg = j.groupby(["match_id","team"], as_index=False)["goals"].sum().rename(columns={"goals":"team_goals"})
    j = j.merge(tg, on=["match_id","team"], how="left")
    j["contrib_pct"] = ((j["goals"] + j["assists"]) / j["team_goals"].replace(0, pd.NA) * 100).round(1).fillna(0)

    # photos
    if not players.empty and {"name","photo_url"}.issubset(players.columns):
        pp = players[["name","photo_url"]].copy()
        pp["name"] = pp["name"].astype(str).map(normalize_name)
        j = j.merge(pp, on="name", how="left")
    else:
        j["photo_url"] = ""

    for c in base_cols:
        if c not in j.columns: j[c] = pd.NA
    return j[base_cols]

# ==============================
# Header / auth
# ==============================
def header():
    st.markdown(
        "<div class='topbar'><span class='brand'>Powerleague Stats</span> "
        "<span class='small'>beta</span></div>",
        unsafe_allow_html=True
    )
    col1, col2 = st.columns([1,1])
    with col1:
        pass
    with col2:
        if not st.session_state.get("is_admin", False):
            with st.popover("Admin login", use_container_width=True):
                pw = st.text_input("Admin password", type="password")
                if st.button("Login"):
                    if pw == ADMIN_PASSWORD and pw.strip():
                        st.session_state["is_admin"] = True
                        st.success("Admin mode enabled")
                        st.experimental_rerun()
                    else:
                        st.error("Wrong password")
        else:
            st.success("Admin mode")

# ==============================
# Pages
# ==============================
def page_matches():
    header()
    matches = normalize_matches(load_table("matches"))
    lineups = normalize_lineups(load_table("lineups"))
    players = load_table("players")

    if matches.empty:
        st.info("No matches yet."); return

    matches["label"] = matches.apply(lambda r: f"S{int(r['season'])} – GW{int(r['gw'])}", axis=1)
    matches = matches.sort_values(["season","gw"])

    mid_map = {row["label"]: row["id"] for _, row in matches.iterrows()}
    sel = st.selectbox("Select match", list(mid_map.keys()), index=len(mid_map.keys())-1)
    mid = mid_map[sel]
    m = matches[matches["id"]==mid].iloc[0]

    # Banner
    left, right = st.columns([1,1])
    with left:
        st.markdown(f"### {m['team_a']} {m['score_a']} – {m['score_b']} {m['team_b']}")
        st.markdown(f"<span class='badge'>MOTM: <b>{m.get('motm_name','')}</b></span>", unsafe_allow_html=True)
        st.caption(f"Season {int(m['season'])} · GW {int(m['gw'])} · {m.get('date') or ''}")

    # Team rows
    a_rows = lineups[(lineups["match_id"]==str(mid)) & (lineups["team"]=="Non-bibs")].copy()
    b_rows = lineups[(lineups["match_id"]==str(mid)) & (lineups["team"]=="Bibs")].copy()

    # Fallback formation (respect side_count)
    def default_form(side_count:int)->str:
        return "1-2-1" if int(side_count)==5 else "2-1-2-1"
    fa = m.get("formation_a") or default_form(int(m.get("side_count",5)))
    fb = m.get("formation_b") or default_form(int(m.get("side_count",5)))

    st.markdown("<div class='dualPitch'>", unsafe_allow_html=True)
    cA, cB = st.columns(2) if st.columns else (None, None)
    with st.container():
        # we cannot rely on the columns object outside Streamlit layout; use dualPitch CSS grid
        pass
    st.markdown("</div>", unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Non-bibs")
        render_team_pitch(a_rows, fa, m.get("motm_name",""), left_side=True, height=600)
    with colB:
        st.markdown("#### Bibs")
        render_team_pitch(b_rows, fb, m.get("motm_name",""), left_side=False, height=600)

    # Simple list under pitches (mobile friendly)
    st.markdown("#### Lineups")
    def tidy(df):
        if df is None or df.empty: return pd.DataFrame(columns=["Player","G","A","GK"])
        dd = df.copy()
        dd["Player"] = dd["name"].astype(str)
        dd["G"] = dd["goals"].astype(int)
        dd["A"] = dd["assists"].astype(int)
        dd["GK"]= dd["is_gk"].map({True:"Yes", False:""})
        return dd[["Player","G","A","GK"]].sort_values(["GK","Player"], ascending=[False, True])
    a_tbl = tidy(a_rows); b_tbl = tidy(b_rows)
    t1, t2 = st.columns(2)
    with t1:
        st.caption("Non-bibs")
        st.dataframe(a_tbl, use_container_width=True, hide_index=True)
    with t2:
        st.caption("Bibs")
        st.dataframe(b_tbl, use_container_width=True, hide_index=True)

def page_stats():
    header()
    players = load_table("players")
    matches = normalize_matches(load_table("matches"))
    lineups  = normalize_lineups(load_table("lineups"))
    fact = build_fact(players, matches, lineups)
    if matches.empty or fact.empty:
        st.info("No stats yet."); return

    seasons = ["All"] + sorted([int(s) for s in matches["season"].dropna().unique().tolist()])

    c1, c2, c3, c4 = st.columns(4)
    with c1: sel_season = st.selectbox("Season", seasons, index=0)
    with c2: min_gp     = st.number_input("Min games", 1, 50, 3, 1)
    with c3: last_n     = st.number_input("Last N (0=all)", 0, 50, 0, 1)
    with c4: metric     = st.selectbox("Metric", ["G+A", "Goals", "Assists", "Contribution%", "Win%"])

    df = fact.copy()
    if sel_season != "All":
        df = df[df["season"] == int(sel_season)]
    if last_n and last_n > 0:
        df = df.sort_values(["season","gw"]).groupby("name", as_index=False).tail(int(last_n))

    agg = df.groupby("name", as_index=False).agg(
        GP=("match_id","nunique"),
        W=("result", lambda s: int((s=="W").sum())),
        D=("result", lambda s: int((s=="D").sum())),
        L=("result", lambda s: int((s=="L").sum())),
        goals=("goals","sum"),
        assists=("assists","sum"),
        contrib=("contrib_pct","mean"),
    )
    agg["Win%"] = (agg["W"] / agg["GP"] * 100).round(1).fillna(0)
    agg["G+A"]  = agg["goals"] + agg["assists"]
    agg["Contribution%"] = agg["contrib"].round(1).fillna(0)
    agg = agg.drop(columns=["contrib"])
    sort_by = {"G+A":"G+A","Goals":"goals","Assists":"assists","Contribution%":"Contribution%","Win%":"Win%"}[metric]
    agg = agg[agg["GP"] >= int(min_gp)].sort_values([sort_by,"GP"], ascending=[False, False])

    nice = agg.rename(columns={"name":"Player","goals":"G","assists":"A"})
    st.dataframe(nice[["Player","GP","W","D","L","Win%","G","A","G+A","Contribution%"]],
                 use_container_width=True, hide_index=True)

def page_player_manager():
    header()
    if not st.session_state.get("is_admin"):
        st.info("Admin only."); return

    players = load_table("players")
    players["name"] = players["name"].fillna("").astype(str).map(normalize_name)
    names = players["name"].dropna().astype(str).sort_values().tolist()

    c1, c2 = st.columns([2,1])
    with c1:
        sel = st.selectbox("Select player", names)
        current = players[players["name"]==sel].iloc[0] if (not players.empty and sel) else None
        new_name = st.text_input("Name", value=(current["name"] if current is not None else ""))
        notes = st.text_area("Notes", value=(current.get("notes","") if current is not None else ""), height=120)

        up = st.file_uploader("Avatar (JPG/PNG/HEIC)", type=["jpg","jpeg","png","heic","HEIC"])
        photo_url = current.get("photo_url","") if current is not None else ""
        if up is not None:
            img = _png_from_uploaded_file(up)
            if img is None:
                st.error("Image read failed (HEIC requires host support). Try JPG/PNG.")
            else:
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True); buf.seek(0)
                key = f"{uuid.uuid4().hex}.png"
                try:
                    sb_admin.storage.from_(AVATAR_BUCKET).upload(key, buf.getvalue(), {"content-type":"image/png","x-upsert":"true"})
                    public = sb_admin.storage.from_(AVATAR_BUCKET).get_public_url(key)
                    photo_url = public["publicUrl"] if isinstance(public, dict) and "publicUrl" in public else (public if isinstance(public, str) else "")
                    st.image(photo_url, width=160, caption="Uploaded")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

        if st.button("Save player", use_container_width=True):
            try:
                if current is not None:
                    sb_admin.table("players").update({"name": new_name.strip(), "notes": notes.strip(), "photo_url": photo_url.strip()}).eq("id", current["id"]).execute()
                else:
                    sb_admin.table("players").upsert({"name": new_name.strip(), "notes": notes.strip(), "photo_url": photo_url.strip()}, on_conflict="name").execute()
                st.success("Saved."); st.cache_data.clear()
            except Exception as e:
                st.error(f"Save failed: {e}")

    with c2:
        if current is not None:
            st.caption("Current")
            if current.get("photo_url"): st.image(current["photo_url"], width=140)
            else:
                st.markdown(f"<div class='card' style='width:140px;height:140px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:32px;font-weight:800'>{name_initials(current['name'])}</div>", unsafe_allow_html=True)

# ==============================
# Router
# ==============================
def run_app():
    page = st.sidebar.radio("Navigate", ["Matches", "Stats", "Player Manager"], index=0)
    if page == "Matches":        page_matches()
    elif page == "Stats":        page_stats()
    elif page == "Player Manager": page_player_manager()

if __name__ == "__main__":
    run_app()
