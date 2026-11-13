"""Ki-67 in the T1D pancreas: AI Passport impact project app.

One page. The project brief (problem, cohort, data) sits at the top, the approach in the
sidebar, and the tools follow one step after the next:
1. Load the fluorescence (DAPI, INS, GCG, Ki-67) and H&E sections
2. Register fluorescence to H&E
3. View a region side by side or overlaid, layers on/off
4. Quantify Ki-67 in endocrine and exocrine areas; compare stages across donors
5. Digital staining: export training pairs, evaluate predicted stains
A timeline fixed at the bottom shows which step you are on.

Run with:  uv run streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from girder_client import AuthenticationError
from plotly.subplots import make_subplots
from PIL import Image
from scipy import stats
from skimage.segmentation import find_boundaries

from impactproject.athena import DEFAULT_API, Archive, HttpError
from impactproject.cohort import (compare_groups, correlate, discover, discover_archive, load_donor_info,
                                  measure_section, measured, paired_compartments, planes_of, save_result,
                                  section_of, stage_of)
from impactproject.digital_stain import FL_ORDER, donor_summary, evaluate, export_pairs, paint_stains
from impactproject.quantify import ROLES, Thresholds, measure, suggest_thresholds
from impactproject.registration import read_pair, register_sections, warp_into
from impactproject.slides import (ArchiveSection, LocalSection, Section, StainSeparated, brightfield_roles,
                                  guess_channel, markers_in_name)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
RESULTS = DATA / "results" / "ki67_sections.csv"
PAIRS = DATA / "digital_staining"
STAGES = ["ND", "sAAb+", "mAAb+", "Recent-onset T1D"]
STAGE_HELP = "ND: non-diabetic · sAAb+: single autoantibody · mAAb+: multiple autoantibodies"
ROLE_LABEL = {"dapi": "Nuclei (DAPI)", "ins": "Insulin (INS)", "gcg": "Glucagon (GCG)", "ki67": "Ki-67"}
ROLE_GUESS = {"dapi": ("dapi", "hoechst", "nuc", "hematoxylin"), "ins": ("ins",), "gcg": ("gcg", "glucagon"),
              "ki67": ("ki-67", "ki67", "mki67")}
SLIDE_EXT = (".svs", ".ndpi", ".tif", ".tiff", ".btf", ".scn")
NONE = -1  # marker not present on this slide
TISSUE_TH = 0.3  # default tissue threshold on brightfield total absorbance (glass ~0.1, tissue ~0.6)
GCG_TH = 0.5  # default glucagon (blue chromogen) threshold, above the faint purple background in nuclei
# display colours (RGB 0..1): blue, green, magenta, yellow
ROLE_COLOR = {"dapi": (0.25, 0.45, 1.0), "ins": (0.1, 0.9, 0.3), "gcg": (1.0, 0.2, 0.85), "ki67": (1.0, 0.85, 0.1)}
STEPS = [("load", "Load sections"), ("register", "Register"), ("view", "View a region"),
         ("quantify", "Quantify Ki-67"), ("stain", "Digital staining")]

st.set_page_config(page_title="Ki-67 in the T1D Pancreas", page_icon="🔬", layout="wide")

ss = st.session_state
for k, v in {"archive": None, "k_reg": None, "k_reg_key": None, "k_roi": None, "k_roi_picked": False,
             "k_pair": None, "k_pair_key": None, "k_result": None, "k_th": None, "k_eval": None,
             "k_exported": set(), "k_saved": set()}.items():
    ss.setdefault(k, v)

# ---------------------------------------------------------------- page styles
st.html("""
<style>
:root { --hema:#4b2a7b; --eosin:#d0577f; --ins:#2f8f5b; --gcg:#c2452d; --ki67:#8a5a1f;
        --tint:#f3eef8; --line:#e6e0ec; --muted:#6a5f74; --fg:#22192b; }
[data-testid="stMainBlockContainer"] { padding-bottom: 120px; max-width: 1280px; }
.brief { display:grid; gap:28px; color:var(--fg); }
.brief .label { font-family:'IBM Plex Mono', ui-monospace, Consolas, monospace; font-size:.75rem;
  letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
.brief h1 { font-family:'Newsreader', Georgia, serif; font-weight:600; font-size:clamp(1.9rem,3.6vw,2.7rem);
  line-height:1.12; margin:6px 0 10px; padding:0; text-wrap:balance; }
.brief .lede { font-size:1.08rem; color:var(--muted); max-width:62ch; margin:0; }
.brief .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.brief .chip { display:inline-flex; align-items:center; gap:8px; font-family:'IBM Plex Mono', monospace;
  font-size:.8rem; padding:3px 10px; border:1px solid var(--line); border-radius:999px; background:#fff; }
.brief .dot { width:10px; height:10px; border-radius:50%; flex:none; }
.dot.ins{background:var(--ins)} .dot.gcg{background:var(--gcg)} .dot.ki{background:var(--ki67)}
.dot.he{background:linear-gradient(90deg,var(--hema) 50%,var(--eosin) 50%)}
.brief .row { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:20px; }
.brief .card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:18px 20px;
  display:grid; gap:10px; align-content:start; min-width:0; }
.brief .card p { margin:0; line-height:1.55; }
.brief .question { background:var(--tint); border-color:transparent; }
.brief .question .q { font-family:'Newsreader', Georgia, serif; font-size:1.2rem; line-height:1.35; }
.brief .stages { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; align-items:end; }
.brief .stage { display:grid; gap:4px; }
.brief .track { height:64px; display:flex; align-items:flex-end; border-bottom:1px solid var(--line); }
.brief .bar { width:100%; background:var(--hema); border-radius:5px 5px 0 0; opacity:.85; }
.brief .n { font-family:'IBM Plex Mono', monospace; font-size:1.25rem; font-weight:500; }
.brief .abbr { font-family:'IBM Plex Mono', monospace; font-size:.75rem; color:var(--muted); }
.brief .axis { display:flex; justify-content:space-between; font-family:'IBM Plex Mono', monospace;
  font-size:.72rem; color:var(--muted); }
.brief .sec { display:grid; gap:4px; padding-top:8px; border-top:1px solid var(--line); }
.brief .sec:first-of-type { border-top:none; padding-top:0; }
@media (max-width: 900px) { .brief .row { grid-template-columns:minmax(0,1fr); } }

.approach ol { list-style:none; margin:0; padding:0; counter-reset:s; display:grid; gap:12px; }
.approach li { counter-increment:s; display:grid; grid-template-columns:26px minmax(0,1fr); gap:10px; }
.approach li::before { content:counter(s); width:24px; height:24px; border-radius:50%; display:grid;
  place-items:center; font-family:'IBM Plex Mono', monospace; font-size:.8rem; color:var(--hema);
  border:1.5px solid var(--hema); }
.approach b { display:block; font-size:.92rem; }
.approach span { color:var(--muted); font-size:.85rem; line-height:1.45; }

.timeline { position:fixed; left:50%; transform:translateX(-50%); bottom:14px; z-index:999991;
  width:min(760px, calc(100vw - 32px)); background:#fff; border:1px solid var(--line); border-radius:16px;
  box-shadow:0 8px 28px rgba(34,25,43,.14); padding:10px 18px; }
.timeline ol { list-style:none; margin:0; padding:0; display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); }
.timeline li { position:relative; display:grid; justify-items:center; gap:4px; text-align:center; }
.timeline li::before { content:""; position:absolute; top:13px; left:-50%; width:100%; height:2px;
  background:var(--line); z-index:0; }
.timeline li:first-child::before { display:none; }
.timeline li.done::before, .timeline li.current::before { background:var(--hema); }
.timeline a { text-decoration:none; color:inherit; display:grid; justify-items:center; gap:4px; z-index:1; }
.timeline .node { width:28px; height:28px; border-radius:50%; display:grid; place-items:center;
  font-family:'IBM Plex Mono', monospace; font-size:.85rem; background:#fff; border:2px solid var(--line);
  color:var(--muted); position:relative; z-index:1; }
.timeline .done .node { background:var(--hema); border-color:var(--hema); color:#fff; }
.timeline .current .node { border-color:var(--hema); color:var(--hema); box-shadow:0 0 0 4px var(--tint); }
.timeline .name { font-size:.78rem; color:var(--muted); line-height:1.2; }
.timeline .current .name { color:var(--fg); font-weight:600; }
.timeline .here { font-family:'IBM Plex Mono', monospace; font-size:.65rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--hema); }
@media (max-width: 640px) { .timeline .name { display:none; } }
</style>
""")

# ---------------------------------------------------------------- project brief
st.html("""
<div class="brief">
  <div>
    <span class="label">AI Passport · Impact project</span>
    <h1>Ki-67 proliferation across the endocrine and exocrine pancreas in type 1 diabetes</h1>
    <p class="lede">Assessment of Ki-67 protein density, and digital staining of islet and Ki-67 markers
      on H&amp;E-stained tissue sections.</p>
    <div class="chips">
      <span class="chip"><span class="dot ins"></span>INS</span>
      <span class="chip"><span class="dot gcg"></span>GCG</span>
      <span class="chip"><span class="dot ki"></span>Ki-67</span>
      <span class="chip"><span class="dot he"></span>H&amp;E</span>
    </div>
  </div>
  <div class="row">
    <div class="card">
      <span class="label">Problem</span>
      <p>Pancreatic tissue morphology and islet composition change as type 1 diabetes (T1D) progresses.
        Islet dysfunction is known to precede insulin loss in islets.</p>
    </div>
    <div class="card question">
      <span class="label">Open question</span>
      <p class="q">At the preclinical stage of disease, does cell proliferation in the pancreas decline,
        or does it increase to compensate?</p>
    </div>
    <div class="card">
      <span class="label">Donor cohort · 42 donors</span>
      <div class="stages">
        <div class="stage"><div class="track"><div class="bar" style="height:83%"></div></div>
          <span class="n">10</span><span class="abbr">ND</span></div>
        <div class="stage"><div class="track"><div class="bar" style="height:83%"></div></div>
          <span class="n">10</span><span class="abbr">sAAb+</span></div>
        <div class="stage"><div class="track"><div class="bar" style="height:100%"></div></div>
          <span class="n">12</span><span class="abbr">mAAb+</span></div>
        <div class="stage"><div class="track"><div class="bar" style="height:83%"></div></div>
          <span class="n">10</span><span class="abbr">T1D</span></div>
      </div>
      <div class="axis"><span>Healthy</span><span>Preclinical</span><span>Onset</span></div>
    </div>
  </div>
  <div class="row" style="grid-template-columns:repeat(2,minmax(0,1fr))">
    <div class="card">
      <span class="label">Data · section 1 · immunostain</span>
      <div class="chips" style="margin-top:0">
        <span class="chip"><span class="dot ins"></span>INS</span>
        <span class="chip"><span class="dot gcg"></span>GCG</span>
        <span class="chip"><span class="dot ki"></span>Ki-67</span>
      </div>
      <p>Endocrine markers define the islets. Ki-67 marks proliferating cells.</p>
    </div>
    <div class="card">
      <span class="label">Data · section 2 · consecutive H&amp;E</span>
      <div class="chips" style="margin-top:0"><span class="chip"><span class="dot he"></span>H&amp;E</span></div>
      <p>The adjacent section, used as the target for digital staining.</p>
    </div>
  </div>
</div>
""")

# ---------------------------------------------------------------- sidebar: approach and plan
with st.sidebar:
    st.html("""
<div class="approach">
  <p style="font-family:'IBM Plex Mono',monospace;font-size:.75rem;letter-spacing:.08em;
     text-transform:uppercase;color:#6a5f74;margin:0 0 10px">Approach</p>
  <ol>
    <li><div><b>Segment the endocrine area</b><span>Islets, identified by insulin (INS) and glucagon (GCG) staining.</span></div></li>
    <li><div><b>Segment the exocrine area</b><span>Tissue negative for the endocrine markers.</span></div></li>
    <li><div><b>Segment Ki-67</b><span>Ki-67 density measured separately in endocrine and exocrine areas.</span></div></li>
    <li><div><b>Digitally stain H&amp;E</b><span>Generate Ki-67 and islet marker staining on H&amp;E sections to increase the data available for analysis.</span></div></li>
  </ol>
</div>
""")
    st.divider()
    st.markdown("**Model**  \nMachine learning, convolutional neural network (CNN)")
    st.markdown("**Success criteria**\n"
                "- Ki-67 density parameters extracted from endocrine and exocrine areas\n"
                "- Digital staining of Ki-67 and islet markers on H&E that matches real staining "
                "in unseen images")
    st.markdown("**Risks and limits**  \nTime is limited.")

# ---------------------------------------------------------------- timeline (fixed at the bottom)
done: set[str] = set()
timeline_slot = st.empty()


def show_timeline():
    current = next((k for k, _ in STEPS if k not in done), None)
    items = []
    for i, (key, name) in enumerate(STEPS, 1):
        state = "done" if key in done else "current" if key == current else "todo"
        mark = "✓" if state == "done" else str(i)
        here = '<span class="here">You are here</span>' if state == "current" else ""
        items.append(f'<li class="{state}"><a href="#{key}"><span class="node">{mark}</span>'
                     f'<span class="name">{name}</span>{here}</a></li>')
    timeline_slot.html(f'<nav class="timeline" aria-label="Progress"><ol>{"".join(items)}</ol></nav>')


@st.fragment
def cohort(title: str = "Cohort"):
    st.subheader(title)
    if not RESULTS.exists():
        st.caption("Measure whole sections to build the cohort table. Results are saved in "
                   "data/results/ki67_sections.csv on this computer.")
        return
    df = pd.read_csv(RESULTS, dtype={"donor": str})
    df = df[df["Compartment"].isin(["Endocrine", "Exocrine"])]
    metric = st.radio("Measure", ["Ki-67+ per mm²", "Ki-67 index (%)"], horizontal=True, key="k_coh_metric")
    order = [s for s in STAGES if s in set(df["stage"])]
    fig = go.Figure()
    colors = {"Endocrine": "#2a78d6", "Exocrine": "#eb6834"}
    for comp in ("Endocrine", "Exocrine"):
        d = df[df["Compartment"] == comp]
        fig.add_trace(go.Violin(x=d["stage"], y=d[metric], name=comp, legendgroup=comp, scalegroup=comp,
                                line_color=colors[comp], box_visible=True, meanline_visible=False,
                                points="all", pointpos=0, jitter=0.3, customdata=d["donor"],
                                hovertemplate="donor %{customdata}<br>%{y:.2f}<extra>" + comp + "</extra>"))
    fig.update_layout(violinmode="group", height=420, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(title="Disease stage", categoryorder="array", categoryarray=order),
                      yaxis=dict(title=metric, rangemode="tozero"), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    tests = []
    for comp in ("Endocrine", "Exocrine"):
        groups = [g[metric].dropna().to_numpy() for _, g in df[df["Compartment"] == comp].groupby("stage")]
        groups = [g for g in groups if len(g)]
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            hs, p = stats.kruskal(*groups)
            tests.append(f"{comp}: Kruskal–Wallis H = {hs:.2f}, p = {p:.3g} ({len(groups)} stages)")
    st.caption(" · ".join(tests) if tests else "Stage comparison needs at least two donors in each of two stages.")
    st.dataframe(df, hide_index=True, width="stretch", height=200)
    st.download_button("Download cohort table (CSV)", df.to_csv(index=False).encode(), "ki67_cohort.csv", "text/csv")
    donor_stats(df, metric)


@st.cache_data(show_spinner=False)
def read_info(path: str, mtime: float) -> pd.DataFrame:
    return load_donor_info(path)


def donor_stats(df: pd.DataFrame, metric: str):
    """Combine the cohort table with the donor information spreadsheet and test for differences."""
    st.markdown("#### Statistics with donor information")
    files = sorted((f for f in DATA.glob("*") if f.suffix.lower() in (".xlsx", ".xls", ".csv")
                    and not f.name.startswith("~$")), key=lambda f: f.stat().st_mtime, reverse=True)
    i1, i2 = st.columns(2)
    up = i2.file_uploader("…or upload a donor spreadsheet", type=["xlsx", "xls", "csv"], key="k_info_up")
    pick = i1.selectbox("Donor information (spreadsheet in the data folder)", files, format_func=lambda f: f.name,
                        key="k_info_file", disabled=up is not None,
                        help="One row per donor. The donor ID column (case_id) is matched to the number in "
                             "the slide names.") if files else None
    try:
        info = load_donor_info(up) if up is not None else read_info(str(pick), pick.stat().st_mtime) if pick else None
    except Exception as e:  # unreadable spreadsheet
        st.error(f"Could not read the spreadsheet: {e}")
        return
    if info is None:
        st.caption("Put the donor spreadsheet (.xlsx) in the data folder or upload it to compare Ki-67 with "
                   "age, sex, BMI, HbA1c, C-peptide and autoantibodies.")
        return

    data = df.merge(info.drop(columns=["stage"], errors="ignore"), on="donor", how="left",
                    suffixes=("", " (info)"), indicator=True)
    found = set(data.loc[data["_merge"] == "both", "donor"])
    missing = sorted(set(df["donor"]) - found)
    data = data.drop(columns="_merge")
    msg = f"{len(found)} of {df['donor'].nunique()} measured donors found in the spreadsheet."
    if missing:
        msg += " Not found: " + ", ".join(missing)
    st.caption(msg)
    if "info_stage" in data:
        clash = data[data["info_stage"].notna() & (data["info_stage"] != data["stage"])]["donor"].unique()
        if len(clash):
            st.warning("Stage from the slide folder differs from the spreadsheet's donor_type for donors: "
                       + ", ".join(clash) + ". Check these before trusting the stage comparison.")

    skip = {"donor", "Compartment", "fl_section", "he_section", "analysis_mpp", "dapi_th", "endocrine_th",
            "ki67_th", "min_islet_um2", "min_nucleus_um2", "case_id", "RR_id"}
    info_cols = [c for c in info.columns if c not in skip]
    cat_cols = ["stage"] + [c for c in info_cols if c != "stage"
                            and not pd.api.types.is_numeric_dtype(info[c]) and 2 <= info[c].nunique() <= 8]
    num_cols = [c for c in info_cols if pd.api.types.is_numeric_dtype(info[c]) and info[c].notna().sum() >= 4]

    comp = st.radio("Compartment", ["Endocrine", "Exocrine"], horizontal=True, key="k_st_comp")
    d = data[data["Compartment"] == comp]
    colors = ["#2a78d6", "#eb6834", "#2f8f5b", "#8a5a1f", "#c2452d", "#6a5f74", "#4b2a7b", "#9a9893"]
    t_grp, t_cor, t_pair = st.tabs(["Compare groups", "Correlate with donor data", "Endocrine vs exocrine"])

    with t_grp:
        grp = st.selectbox("Group donors by", cat_cols, key="k_st_grp",
                           format_func=lambda c: "stage (from slide folders)" if c == "stage" else c)
        dd = d.dropna(subset=[grp, metric])
        levels = [s for s in STAGES if s in set(dd[grp])] if grp in ("stage", "info_stage") else \
            sorted(dd[grp].astype(str).unique())
        fig = go.Figure()
        for k, lv in enumerate(levels):
            g = dd[dd[grp].astype(str) == str(lv)]
            fig.add_trace(go.Box(x=[str(lv)] * len(g), y=g[metric], name=str(lv), boxpoints="all", jitter=0.3,
                                 pointpos=0, marker_color=colors[k % len(colors)], customdata=g["donor"],
                                 hovertemplate="donor %{customdata}<br>%{y:.2f}<extra></extra>"))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                          xaxis=dict(title=grp, categoryorder="array", categoryarray=[str(v) for v in levels]),
                          yaxis=dict(title=f"{comp} {metric}", rangemode="tozero"))
        st.plotly_chart(fig, width="stretch", key="k_st_grp_fig")
        summary, pw = compare_groups(dd.assign(**{grp: dd[grp].astype(str)}), grp, metric)
        st.markdown(f"**{summary}**")
        if len(pw):
            st.dataframe(pw.round(4), hide_index=True, width="stretch")
            st.caption("Pairwise Mann–Whitney U tests; 'p (Holm)' is corrected for the number of comparisons. "
                       "Non-parametric tests suit small groups (about 10 donors each) and skewed counts.")

    with t_cor:
        if not num_cols:
            st.caption("No numeric donor columns with enough values.")
        else:
            default = next((i for i, c in enumerate(num_cols) if c.lower().startswith("age")), 0)
            x = st.selectbox("Donor variable", num_cols, index=default, key="k_st_x")
            fig = go.Figure()
            for k, s in enumerate([s for s in STAGES if s in set(d["stage"])]):
                g = d[d["stage"] == s]
                fig.add_trace(go.Scatter(x=g[x], y=g[metric], mode="markers", name=s, customdata=g["donor"],
                                         marker=dict(size=10, color=colors[k], line=dict(width=1, color="white")),
                                         hovertemplate="donor %{customdata}<br>%{x} · %{y:.2f}<extra></extra>"))
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1),
                              xaxis=dict(title=x), yaxis=dict(title=f"{comp} {metric}", rangemode="tozero"))
            st.plotly_chart(fig, width="stretch", key="k_st_cor_fig")
            lines = [f"All donors: {correlate(d, x, metric)}"]
            lines += [f"{s}: {correlate(d[d['stage'] == s], x, metric)}" for s in STAGES if s in set(d["stage"])]
            st.markdown("  \n".join(lines))
            rows = [{"donor variable": c, "result": correlate(d, c, metric)} for c in num_cols]
            with st.expander("All donor variables at once"):
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("Many tests at once: treat single small p-values as leads, not findings.")

    with t_pair:
        st.caption("Is Ki-67 higher in islets than in the surrounding exocrine tissue of the same donor?")
        lines = [f"All stages: {paired_compartments(data, metric) or 'no data'}"]
        lines += [f"{s}: {paired_compartments(data[data['stage'] == s], metric) or 'no data'}"
                  for s in STAGES if s in set(data["stage"])]
        st.markdown("  \n".join(lines))

    st.download_button("Download Ki-67 + donor information (CSV)", data.to_csv(index=False).encode(),
                       "ki67_with_donor_info.csv", "text/csv")



def halt():
    """Stop here until the user completes this step, keeping the timeline up to date.

    Saved cohort results stay visible below, so they can be shown without loading slides.
    """
    show_timeline()
    if RESULTS.exists():
        st.divider()
        cohort("Saved cohort results")
    st.stop()


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Opening slide...")
def local_section(path: str, mtime: float) -> LocalSection:
    sec = LocalSection(path)
    sec.calibrate()
    return sec


def to_src(img: np.ndarray) -> str:
    """Image array -> JPEG data URI (small payload for plotly)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def composite(fl: np.ndarray, roles: list[str], show: dict[str, bool]) -> np.ndarray:
    """Additive colour composite of the shown fluorescence channels (float 0..1 RGB)."""
    out = np.zeros(fl.shape[:2] + (3,), np.float32)
    for k, r in enumerate(roles):
        if show.get(r):
            out += fl[..., k, None] * np.array(ROLE_COLOR[r], np.float32)
    return np.clip(out, 0, 1)


def blend(he: np.ndarray, fl_rgb: np.ndarray, alpha: float) -> np.ndarray:
    """Fluorescence over H&E: opaque where the signal is strong, H&E shows through elsewhere."""
    a = np.clip(fl_rgb.max(-1, keepdims=True) * 1.5, 0, 1) * alpha
    return he.astype(np.float32) / 255 * (1 - a) + fl_rgb * a


def draw_marks(img: np.ndarray, masks: dict | None, points: np.ndarray | None,
               outline: bool, dots: bool) -> np.ndarray:
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.dtype != np.uint8 else img.copy()
    if masks is not None and outline:
        edge = find_boundaries(masks["endocrine"], mode="outer")
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        out[edge] = (0, 230, 255)
    if points is not None and dots:
        r = max(4, out.shape[1] // 300)
        for y, x in points:
            cv2.circle(out, (int(x), int(y)), r, (255, 140, 0), 2)
    return out


def section_info(sec: Section) -> str:
    size = f"{sec.width:,} × {sec.height:,} px"
    mm = f" · {sec.width * sec.mpp / 1000:.1f} × {sec.height * sec.mpp / 1000:.1f} mm" if sec.mpp else ""
    kind = "brightfield RGB" if sec.rgb else f"{len(sec.channels)} channels: " + ", ".join(sec.channels)
    return f"{size}{mm} · {kind}"


def project_tree() -> list[dict] | None:
    """The project folder (pasted link) and every folder below it, with their slides; cached."""
    link = ss.get("k_root_link", "")
    m = re.search(r"folder/([0-9a-fA-F]{24})", link) or re.fullmatch(r"\s*([0-9a-fA-F]{24})\s*", link)
    if not m:
        if link.strip():
            st.error("No folder ID found in this link. Open the folder on FUSION and copy the whole address.")
        return None
    key = (ss.archive.api_url, ss.archive.who, m.group(1))
    if ss.get("k_tree_key") != key:
        try:
            with st.spinner("Listing the folders and slides..."):
                ss.k_tree_name = ss.archive.folder(m.group(1))["name"]
                ss.k_tree = ss.archive.walk(m.group(1))
            ss.k_tree_key = key
        except HttpError as e:
            ss.k_tree_key = None
            st.error(f"The archive refused this folder ({e.status}). If it is in someone else's private "
                     "space, ask them to share it with you (folder → Access control).")
            return None
    return ss.k_tree


def archive_picker(prefix: str, want: str) -> dict | None:
    """Folder dropdown, then slide dropdown, within the project folder. `want`: 'IHC' or 'H&E'."""
    if ss.archive is None:
        st.info("Connect to FUSION first (the box above these two columns).")
        return None
    tree = ss.get("k_tree") if ss.get("k_tree_key") else None
    if not tree:
        st.info("Paste the project folder link in the box above these two columns.")
        return None
    folders = [f for f in tree if f["slides"]]
    if not folders:
        others = sum(f["others"] for f in tree)
        st.warning("No slides in this folder or below it." + (
            f" {others} files are there that FUSION has not turned into viewable slides yet "
            "(still processing, or not a slide format)." if others else ""))
        return None
    idx = next((i for i, f in enumerate(folders)
                if section_of(f["path"], f["slides"][0]["name"]) == want), 0)
    folder = st.selectbox("Folder", folders, index=idx, key=f"{prefix}_folder",
                          format_func=lambda f: (" / ".join(f["path"]) or ss.k_tree_name)
                          + f"  ({len(f['slides'])} slides)")
    slides = folder["slides"]
    sidx = 0
    if want == "H&E" and (m := re.search(r"(?<!\d)(\d{4,})(?!\d)", ss.get("kfl_picked_name", ""))):
        # default to the H&E slide of the same donor as the immunostain
        sidx = next((i for i, s in enumerate(slides) if m.group(1) in s["name"]), 0)
    return st.selectbox("Slide", slides, index=sidx, format_func=lambda i: i["name"],
                        key=f"{prefix}_item_{folder['id']}_{m.group(1) if want == 'H&E' and m else ''}")


def load_section(prefix: str, label: str, want: str) -> Section | None:
    src = st.radio("Source", ["FUSION / Athena", "Local file", "Upload"], horizontal=True, key=f"{prefix}_src",
                   help="FUSION: pick the folder and slide from the project folder (connect above first). "
                        "Local file: a slide on this computer; the app reads only the parts it needs. "
                        "Uploads are copied to data/uploads/.")
    try:
        if src == "Local file":
            p = st.text_input(f"Path to the {label} slide", key=f"{prefix}_path",
                              placeholder=r"C:\slides\donor01_IF.ome.tif",
                              help="In File Explorer: Shift + right-click the file → Copy as path")
            p = p.strip().strip('"')
            if not p:
                return None
            path = Path(p)
            if not path.exists():
                st.error("File not found. Check the path.")
                return None
            if path.is_dir():
                files = sorted(f for f in path.iterdir() if f.name.lower().endswith(SLIDE_EXT))
                if not files:
                    st.error("This is a folder with no slide files in it. Paste the path of the slide file "
                             "(.svs, .ndpi, .tif, .ome.tif).")
                    return None
                path = st.selectbox("Slide in this folder", files, format_func=lambda f: f.name,
                                    key=f"{prefix}_pick")
            return local_section(str(path), path.stat().st_mtime)
        if src == "Upload":
            up = st.file_uploader(f"{label} slide", type=["tif", "tiff", "svs", "ndpi", "btf"],
                                  key=f"{prefix}_up")
            if up is None:
                return None
            UPLOADS.mkdir(parents=True, exist_ok=True)
            dest = UPLOADS / up.name
            if not dest.exists() or dest.stat().st_size != up.size:
                with st.spinner("Saving upload..."):
                    dest.write_bytes(up.getbuffer())
            return local_section(str(dest), dest.stat().st_mtime)
        item = archive_picker(prefix, want)
        if item is None:
            return None
        ss[f"{prefix}_picked_name"] = item["name"]
        key = f"{prefix}_arch_{item['_id']}"
        if key not in ss:
            with st.spinner("Reading slide information..."):
                sec = ArchiveSection(ss.archive, item)
                sec.calibrate()
                ss[key] = sec
        return ss[key]
    except HttpError as e:
        st.error(f"The archive returned {e.status} for this slide.")
    except Exception as e:  # unreadable or unsupported file
        st.error(f"Could not open this slide: {e}")
    return None


# ================================================================ 1. load
st.header("1 · Load sections", anchor="load")
st.caption("Open the immunostained section (multiplex fluorescence with DAPI, INS, GCG, Ki-67, or brightfield "
           "IHC) and its consecutive H&E section. A slide file or a folder containing it both work.")
with st.expander(f"FUSION / Athena: connected as {ss.archive.who}" if ss.archive else
                 "Connect to FUSION (or Athena)", expanded=not ss.archive):
    with st.form("login"):
        api = st.text_input("Server API", DEFAULT_API,
                            help="FUSION: https://fusionpub.rc.ufl.edu/api/v1 · "
                                 "Athena: https://athena.rc.ufl.edu/api/v1")
        st.caption("Fill in username and password **or** an API key. Leave all empty for public data only.")
        a1, a2, a3 = st.columns(3)
        user = a1.text_input("Username").strip()
        pw = a2.text_input("Password", type="password")
        key = a3.text_input("API key", type="password",
                            help="On the archive: your user menu → My account → API keys").strip()
        if st.form_submit_button("Connect", type="primary"):
            ss.archive = None
            if not key and bool(user) != bool(pw):
                ss.connect_msg = ("error", "Enter both username and password.")
            else:
                try:
                    with st.spinner("Connecting..."):
                        ss.archive = Archive(api,
                                             username=user if not key else None,
                                             password=pw if not key else None,
                                             api_key=key or None)
                    ss.connect_msg = None
                except HttpError as e:
                    ss.connect_msg = ("error", f"Could not connect ({e.status}): check the server address. "
                                               f"{e.responseText[:200]}")
                except AuthenticationError:
                    ss.connect_msg = ("error", "Sign-in failed: the username/password or API key was not accepted.")
                except Exception as e:  # network errors etc.
                    ss.connect_msg = ("error", f"Could not connect: {str(e) or type(e).__name__}. "
                                               "Check the server address and your network/VPN.")
            st.rerun()
        del pw, key  # never kept beyond the login call
    if ss.archive and st.button("Disconnect"):
        ss.archive = None
        st.rerun()

if ss.archive:
    host = ss.archive.api_url.split("//")[-1].split("/")[0]
    if ss.archive.user:
        st.success(f"Connected to **{host}** as **{ss.archive.who}**.", icon="✅")
    else:
        st.warning(f"Connected to **{host}** without signing in: only public data is visible. "
                   "Sign in to open private or shared folders.", icon="⚠️")
elif ss.get("connect_msg"):
    st.error(ss.connect_msg[1], icon="❌")

if ss.archive:
    p1, p2 = st.columns([5, 1], vertical_alignment="bottom")
    p1.text_input("Project folder on FUSION (paste the folder's browser address)", key="k_root_link",
                  placeholder="https://fusionpub.rc.ufl.edu/#user/…/folder/…",
                  help="The folder that holds all slides, e.g. one with ND, sAAB and mAAB inside, each with "
                       "H&E and IHC folders. Both sections below and 'Measure all donors' use it.")
    if p2.button("Refresh list", help="Read the folder again, e.g. after new uploads"):
        ss.k_tree_key = None
    tree = project_tree()
    if tree:
        n_sl = sum(len(f["slides"]) for f in tree)
        n_other = sum(f["others"] for f in tree)
        st.caption(f"**{ss.k_tree_name}**: {len(tree)} folders, {n_sl} slides"
                   + (f" · {n_other} files not yet viewable as slides" if n_other else ""))

c_fl, c_he = st.columns(2, gap="large")
with c_fl:
    st.subheader("Immunostain")
    fl = load_section("kfl", "immunostain", "IHC")
    if fl is not None and fl.rgb:
        # brightfield IHC: split into hematoxylin / DAB / red chromogen so it works like fluorescence
        key = ("k_sep", fl.name, fl.width, fl.height)
        if ss.get("k_sep_key") != key:
            with st.spinner("Separating the stains (hematoxylin, DAB)..."):
                sep = StainSeparated(fl)
                sep.calibrate()
            ss.k_sep, ss.k_sep_key = sep, key
        fl = ss.k_sep
        st.caption(section_info(fl.base) + " · brightfield IHC, separated into blue, DAB (brown) and red "
                   "chromogen. Markers in the file name: "
                   + (", ".join(ROLE_LABEL[m] for m in ("ins", "gcg", "ki67") if m in markers_in_name(fl.name))
                      or "none listed"))
    elif fl is not None:
        st.caption(section_info(fl))
with c_he:
    st.subheader("H&E")
    he = load_section("khe", "H&E", "H&E")
    if he is not None:
        st.caption(section_info(he))

if he is not None and not he.rgb:
    st.error("The H&E slide opened as a multi-channel image. Choose the brightfield H&E file.")
    halt()
if fl is None or he is None:
    st.info("Load both sections to continue.")
    halt()

if ss.get("k_auto_from") != fl.name:
    # a new immunostain slide: fill donor and stage from its name (and folder) when they can be read
    ss.k_auto_from = fl.name
    for r in ROLES:  # channel defaults follow the new slide's name
        ss.pop(f"k_role_{r}", None)
    if m := re.search(r"(?<!\d)(\d{4,})(?!\d)", fl.name):
        ss.k_donor = m.group(1)
    folder = ss.get("kfl_folder") if ss.get("kfl_src", "").startswith("FUSION") else None
    guess = stage_of(fl.name) or (stage_of(folder["path"][0]) if folder and folder["path"] else None)
    if guess in STAGES:
        ss.k_stage = guess

m1, m2, m3, m4 = st.columns(4)
donor = m1.text_input("Donor ID", key="k_donor", placeholder="e.g. 6123")
stage = m2.selectbox("Disease stage", STAGES, key="k_stage", help=STAGE_HELP)
for sec, col, nm in ((fl, m3, "immunostain"), (he, m4, "H&E")):
    if sec.mpp is None:
        sec.mpp = col.number_input(f"µm per pixel ({nm})", 0.05, 20.0, 0.5, 0.01, key=f"k_mpp_{nm}",
                                   help="Not recorded in the file. Check the scan settings.")
    else:
        col.metric(f"{nm} pixel size", f"{sec.mpp:.3f} µm")

brightfield = isinstance(fl, StainSeparated)
named = markers_in_name(fl.name)  # markers listed in the file name, e.g. INS_GCG_Ki67
marker_opts = [i for i in range(len(fl.channels))
               if not (brightfield and (i in StainSeparated.RGB_CHANNELS or i == StainSeparated.TISSUE))]
with st.expander("Channels", expanded=brightfield or
                 any(guess_channel(fl.channels, *ROLE_GUESS[r]) is None for r in ROLES)):
    bf_default = brightfield_roles(fl.name) if brightfield else {}
    if brightfield:
        st.caption(
            "nPOD IHC colours: **insulin red**, **Ki-67 brown (DAB)**, **glucagon blue**. Slides with glucagon "
            "(INS_GCG_Ki67) have **no nuclear stain**: the faint purple in nuclei is glucagon background, so "
            "nuclei are not counted and there is no Ki-67 index, only Ki-67+ per mm². INS_Ki67 slides have a "
            "blue hematoxylin counterstain, used for nuclei. Markers not in the file name are switched off. "
            "Tissue is outlined from all stains together, so it does not depend on nuclei.")
    elif named:
        st.caption("Markers not listed in the file name are switched off.")
    rc = st.columns(4)
    roles: dict[str, int] = {}
    for col, r in zip(rc, ROLES):
        g = guess_channel(fl.channels, *ROLE_GUESS[r])
        absent = (bf_default.get(r) == NONE) if brightfield else (r != "dapi" and bool(named) and r not in named)
        if absent:
            # not on this slide (from the file name): shown, locked off
            col.selectbox(ROLE_LABEL[r], [NONE], key=f"k_role_{r}_off", disabled=True,
                          format_func=lambda i, r=r: "— not stained on this slide —" if r == "dapi"
                          else "— not on this slide —",
                          help="From the file name: " + ("glucagon slides have no nuclear counterstain."
                                                         if r == "dapi" else "this marker is not listed."))
            roles[r] = NONE
            continue
        if brightfield:
            opts = marker_opts + [NONE]
            idx = opts.index(bf_default[r])
        elif r == "dapi":
            opts = marker_opts
            idx = opts.index(g) if g in opts else 0
        else:
            opts = marker_opts + [NONE]
            idx = opts.index(g) if g in opts else len(opts) - 1
        roles[r] = col.selectbox(ROLE_LABEL[r], opts, index=idx, key=f"k_role_{r}",
                                 format_func=lambda i: "— not on this slide —" if i == NONE else fl.channels[i])
    used = [v for v in roles.values() if v != NONE]
    if len(set(used)) < len(used):
        st.warning("Two roles use the same channel. Check the channel names.")
    if st.toggle("Show channel previews", brightfield, key="k_ch_prev",
                 help="A small piece of tissue split into each channel, to check which stain is which."):
        pkey = ("k_prev", fl.name)
        if ss.get("k_prev_key") != pkey:
            with st.spinner("Reading a preview..."):
                # a ~1.5 mm square at the centre of the slide, at ~2 µm/px
                side = (1500 / fl.mpp) if fl.mpp else fl.width / 8
                cx, cy = fl.width / 2, fl.height / 2
                box = (max(0, cx - side / 2), max(0, cy - side / 2),
                       min(fl.width, cx + side / 2), min(fl.height, cy + side / 2))
                n = len(fl.channels)
                ss.k_prev = fl.read(box, 600, list(range(n)))
                ss.k_prev_key = pkey
        arr = ss.k_prev
        shown = list(StainSeparated.RGB_CHANNELS) if brightfield else []
        pics, caps = [], []
        if brightfield:
            pics.append(np.clip(arr[..., shown] * 255, 0, 255).astype(np.uint8))
            caps.append("As scanned")
        for i in marker_opts:
            pics.append((np.clip(arr[..., i], 0, 1) * 255).astype(np.uint8))
            caps.append(fl.channels[i] + " (bright = stain)")
        st.image(pics, caption=caps, width=220)
        st.caption("Centre of the slide; it may miss islets. Step 3 shows any region you pick in full detail.")
# Tissue outline and registration: brightfield uses total absorbance (works without a nuclear stain),
# fluorescence uses DAPI.
TCH = StainSeparated.TISSUE if brightfield else roles["dapi"]
SIGNAL = "density" if brightfield else "nuclei"
MISSING = [k for k, r in enumerate(ROLES) if roles[r] == NONE]
CHANS = [roles[r] if roles[r] != NONE else TCH for r in ROLES] + [TCH]  # ROLES order, then tissue
NP = len(CHANS)  # marker planes + tissue plane


def read_roles(arr: np.ndarray) -> np.ndarray:
    """Blank the planes of markers that are not on this slide."""
    arr[..., MISSING] = 0
    return arr


if brightfield and roles["ins"] == NONE and roles["gcg"] == NONE and roles["ki67"] == NONE:
    st.error("No markers found for this slide. Check the Channels box.")
    halt()
if not donor:
    st.info("Enter the donor ID to continue.")
    halt()
done.add("load")

# ================================================================ 2. register
st.header("2 · Register the immunostain to H&E", anchor="register")
st.caption(("Matches the tissue itself (total stain of both slides: tissue, ducts, vessels, septa), since the "
            "IHC may have no nuclear stain." if brightfield else
            "Matches nuclei: DAPI in fluorescence against hematoxylin in H&E.")
           + " The coarse fit covers the whole section; each region you view is aligned again, searching "
             "up to ~300 µm around the coarse position.")
reg_key = (he.name, fl.name, TCH, SIGNAL, he.mpp, fl.mpp)
if ss.k_reg_key != reg_key:
    ss.k_reg, ss.k_pair, ss.k_result, ss.k_roi, ss.k_roi_picked = None, None, None, None, False
if st.button("Register sections", type="primary"):
    bar = st.progress(0.0, "Reading overviews")
    try:
        ss.k_reg = register_sections(he, fl, TCH, progress=lambda f, m: bar.progress(f, m), signal=SIGNAL)
        ss.k_reg_key = reg_key
        ss.k_pair = None
        st.rerun()
    except ValueError as e:
        st.error(str(e))
    bar.empty()

reg = ss.k_reg
if reg is None:
    st.info("Press **Register sections**. It takes a few seconds to a minute.")
    halt()
done.add("register")

with st.expander("Adjust registration by hand", expanded=reg.score_after < 0.3):
    a1, a2, a3, a4 = st.columns(4)
    rot = a1.number_input("Rotate (°)", -20.0, 20.0, 0.0, 0.1, key="k_adj_rot")
    dx = a2.number_input("Shift right (µm)", -2000.0, 2000.0, 0.0, 5.0, key="k_adj_dx")
    dy = a3.number_input("Shift down (µm)", -2000.0, 2000.0, 0.0, 5.0, key="k_adj_dy")
    scl = a4.number_input("Scale", 0.8, 1.25, 1.0, 0.005, key="k_adj_scale", format="%.3f")
T = reg.with_adjustment(rot, dx, dy, scl, he.mpp, (he.width, he.height))

q1, q2 = st.columns([1, 3])
quality = reg.score_after
q1.metric("Match (DAPI vs hematoxylin)", f"{quality:.2f}", f"{quality - reg.score_before:+.2f} from rigid start",
          help="Normalised cross-correlation inside the tissue, 1 = identical. Consecutive sections "
               "usually land between 0.3 and 0.8 at this coarse scale. Check the overlay by eye.")
if quality < 0.2:
    q1.warning("Weak match. Check the overlay and adjust by hand.")
ov_mode = q1.radio("Overview", ["Overlay", "Checkerboard", "Side by side"], key="k_ov_mode")
ov_alpha = q1.slider("Fluorescence opacity", 0.0, 1.0, 0.7, 0.05, key="k_ov_alpha")

he_ov = reg.he_overview
s_he = reg.he_scale
ov_key = ("fl_ov", fl.name, tuple(roles.values()), brightfield)
if ss.get("k_flov_key") != ov_key:
    ss.k_flov = read_roles(fl.read(None, min(2048, int(he_ov.shape[1] * fl.width / he.width * 1.2) + 1), CHANS))
    ss.k_flov_key = ov_key
fl_ov = ss.k_flov
s_fl = fl_ov.shape[1] / fl.width
M_ov = np.diag([s_fl, s_fl, 1]) @ T @ np.diag([1 / s_he, 1 / s_he, 1])
fl_ov_w = warp_into(fl_ov, M_ov, (he_ov.shape[1], he_ov.shape[0]))
fl_rgb_ov = composite(fl_ov_w, list(ROLES), {"dapi": True, "ins": True, "gcg": True, "ki67": False})
if ov_mode == "Overlay":
    ov_img = (blend(he_ov, fl_rgb_ov, ov_alpha) * 255).astype(np.uint8)
elif ov_mode == "Checkerboard":
    n = max(8, he_ov.shape[1] // 12)
    yy, xx = np.indices(he_ov.shape[:2])
    cb = ((yy // n + xx // n) % 2 == 0)[..., None]
    ov_img = np.where(cb, he_ov, (fl_rgb_ov * 255).astype(np.uint8))
else:
    ov_img = np.concatenate([he_ov, (fl_rgb_ov * 255).astype(np.uint8)], axis=1)

with q2:
    fig = go.Figure(go.Image(source=to_src(ov_img), hoverinfo="skip"))
    if ss.k_roi is not None:
        l, t, r, b = (v * s_he for v in ss.k_roi)
        fig.add_shape(type="rect", x0=l, y0=t, x1=r, y1=b, line=dict(color="#ffd400", width=2))
    # invisible grid so box selection works on the image
    gy, gx = np.mgrid[0:he_ov.shape[0]:8, 0:he_ov.shape[1]:8]
    fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", marker=dict(size=1, opacity=0),
                             hoverinfo="skip", showlegend=False))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), dragmode="select",
                      xaxis=dict(visible=False, range=[0, ov_img.shape[1]]),
                      yaxis=dict(visible=False, scaleanchor="x", range=[he_ov.shape[0], 0]))
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="box", key="k_ovfig")
    st.caption("**Drag a box** on the image to pick the region to view at full resolution (step 3). "
               + ("Blue = nuclei, " if roles["dapi"] != NONE else "")
               + "green = insulin, magenta = glucagon (as detected on the immunostain).")
    if roles["ins"] != NONE or roles["gcg"] != NONE:
        if st.button("Go to the region with the most islets",
                     help="Picks the 1.5 mm square with the most insulin/glucagon stain, a good place to check "
                          "the registration and the digital staining."):
            endo_ov = np.maximum(fl_ov_w[..., ROLES.index("ins")], fl_ov_w[..., ROLES.index("gcg")])
            side_ov = 1500 / he.mpp * s_he if he.mpp else he_ov.shape[1] / 10
            dens = cv2.blur((endo_ov > 0.3).astype(np.float32), (max(1, int(side_ov)),) * 2)
            yy, xx = np.unravel_index(int(np.argmax(dens)), dens.shape)
            half = side_ov / 2 / s_he
            cx, cy = xx / s_he, yy / s_he
            ss.k_roi = (max(0, cx - half), max(0, cy - half), min(he.width, cx + half), min(he.height, cy + half))
            ss.k_roi_picked = True
            st.rerun()
    box = (ev or {}).get("selection", {}).get("box", []) if isinstance(ev, dict) else \
        getattr(getattr(ev, "selection", None), "box", [])
    if box:
        xs, ys = box[0]["x"], box[0]["y"]
        new = (min(xs) / s_he, min(ys) / s_he, max(xs) / s_he, max(ys) / s_he)
        new = (max(0, new[0]), max(0, new[1]), min(he.width, new[2]), min(he.height, new[3]))
        if new[0] < he.width and new != ss.k_roi and new[2] - new[0] > 10 and new[3] - new[1] > 10:
            ss.k_roi, ss.k_roi_picked = new, True
            st.rerun()

# ================================================================ 3. view region
st.header("3 · View a region", anchor="view")
if ss.k_roi is None:
    side = 1000 / he.mpp
    ss.k_roi = (he.width / 2 - side / 2, he.height / 2 - side / 2, he.width / 2 + side / 2, he.height / 2 + side / 2)
roi = ss.k_roi
roi_um = ((roi[2] - roi[0]) * he.mpp, (roi[3] - roi[1]) * he.mpp)
if not ss.k_roi_picked:
    st.caption("Showing a 1 mm region at the centre. Drag a box on the overview in step 2 to choose your own.")

v1, v2, v3 = st.columns([2, 2, 3])
view_mpp = v1.select_slider("Resolution (µm per pixel)", [0.25, 0.5, 1.0, 2.0, 4.0],
                            value=1.0 if max(roi_um) < 3000 else 2.0, key="k_view_mpp")
refine = v2.toggle("Refine registration in this region", True, key="k_refine")
out_w = int(min(3000, max(64, roi_um[0] / max(view_mpp, he.mpp))))
v3.caption(f"Region {roi_um[0]:,.0f} × {roi_um[1]:,.0f} µm, read at {out_w:,} px wide. "
           "Large regions are capped at 3000 px.")

pair_key = (reg_key, tuple(np.round(T.ravel(), 6)), tuple(round(v) for v in roi), out_w, refine)
if ss.k_pair_key != pair_key:
    with st.spinner("Reading and aligning the region..."):
        extra = list(StainSeparated.RGB_CHANNELS) if brightfield else []
        ss.k_pair = read_pair(he, fl, T, roi, out_w, CHANS + extra, TCH, refine, SIGNAL)
        ss.k_pair.fl[..., MISSING] = 0
        ss.k_pair_key = pair_key
        ss.k_result = None
pair = ss.k_pair
pair_fl = pair.fl[..., :len(ROLES)]  # marker planes in ROLES order
pair_planes = planes_of(pair.fl[..., :NP])  # markers + tissue, for measuring
ihc_rgb = (pair.fl[..., NP:] * 255).astype(np.uint8) if pair.fl.shape[-1] > NP else None
st.caption(f"Local match {pair.score:.2f}" + (" (aligned again in this region)" if pair.refined else "")
           + (". Below ~0.3, check by eye: the sections may be far apart or the region has little structure."
              if pair.score < 0.3 else ""))

with st.container(border=True):
    l1, l2, l3, l4 = st.columns([2, 4, 2, 2])
    mode = l1.radio("Display", ["Side by side", "Overlay"], key="k_mode")
    lc = l2.columns(4)
    show = {r: lc[i].checkbox(ROLE_LABEL[r].split(" ")[0], value=r != "dapi", key=f"k_show_{r}",
                              disabled=roles[r] == NONE) and roles[r] != NONE
            for i, r in enumerate(ROLES)}
    show_he = l3.checkbox("H&E", True, key="k_show_he")
    real_ihc = brightfield and l3.checkbox("Real IHC colours", True, key="k_ihc_real",
                                           help="Side by side: show the IHC slide as scanned "
                                                "instead of the separated markers.")
    alpha = l3.slider("Opacity", 0.0, 1.0, 0.75, 0.05, key="k_alpha", disabled=mode != "Overlay")
    res = ss.k_result
    outline = l4.checkbox("Islet outlines", True, key="k_show_out", disabled=res is None,
                          help="Cyan outline of each detected islet (endocrine area); everything else inside "
                               "the tissue is exocrine. Available after measuring the region (step 4), where "
                               "the ⓘ icons explain the detection.")
    dots = l4.checkbox("Ki-67+ nuclei", True, key="k_show_dots", disabled=res is None,
                       help="An orange circle on each counted Ki-67+ nucleus. Available after measuring the region (step 4).")

masks = res.masks if res is not None else None
pts = res.ki67_points if res is not None else None
fl_rgb = composite(pair_fl, list(ROLES), show)
he_img = pair.he if show_he else np.full_like(pair.he, 255)
if mode == "Overlay":
    base = blend(he_img, fl_rgb, alpha) if show_he else fl_rgb
    panels = [("Overlay", draw_marks(base, masks, pts, outline, dots))]
else:
    right = ("Immunostain (as scanned)", ihc_rgb) if real_ihc and ihc_rgb is not None else ("Immunostain", fl_rgb)
    panels = [("H&E", draw_marks(he_img, masks, pts, outline, dots)),
              (right[0], draw_marks(right[1], masks, pts, outline, dots))]
fig = make_subplots(1, len(panels), subplot_titles=[p[0] for p in panels], horizontal_spacing=0.01)
h, w = pair.he.shape[:2]
for i, (_, img) in enumerate(panels, 1):
    fig.add_trace(go.Image(source=to_src(img), hoverinfo="skip"), 1, i)
    sfx = "" if i == 1 else str(i)
    fig.update_xaxes(visible=False, range=[0, w], matches=None if i == 1 else "x", row=1, col=i)
    fig.update_yaxes(visible=False, range=[h, 0], matches=None if i == 1 else "y", scaleanchor=f"x{sfx}",
                     row=1, col=i)
fig.update_layout(height=640 if mode == "Overlay" else 520, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, width="stretch", key="k_viewfig", config={"scrollZoom": True})
st.caption("Scroll to zoom and drag to pan; in side-by-side view both panels move together.")
# full-resolution PNG of each panel, with the islet outlines and Ki-67 circles as shown
dl = st.columns(len(panels) + 2)
for col, (name, img) in zip(dl, panels):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    tag = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    col.download_button(f"Download {name} (PNG)", buf.getvalue(),
                        f"{tag}_{donor}_{int(roi[0])}_{int(roi[1])}.png", "image/png", key=f"k_dl_{tag}",
                        help="Full resolution of this region, with the outlines and circles that are switched on.")
if ss.k_roi_picked or ss.k_result is not None:
    done.add("view")

# ================================================================ 4. quantify
st.header("4 · Quantify Ki-67", anchor="quantify")
st.caption("Endocrine = INS or GCG positive islets · exocrine = remaining tissue. Ki-67+ nuclei are "
           "counted per mm², and as a % of all nuclei where nuclei are stained. Use the same thresholds for every "
           "donor. Hover over each ⓘ below for how it is detected.")
HAS_NUC = roles["dapi"] != NONE
NUC = "the blue hematoxylin counterstain" if brightfield else "DAPI"
ISL = (" or ".join(n for r, n in (("ins", "red chromogen (insulin)"), ("gcg", "blue chromogen (glucagon)"))
                   if roles[r] != NONE) or "no islet marker") if brightfield else "INS or GCG"
KI = "DAB (brown)" if brightfield else "the Ki-67 channel"
HELP = {
    "stains": (
        "The IHC slide is split by colour deconvolution into three stain amounts per pixel: **red chromogen** "
        "(insulin), **DAB** (brown, Ki-67) and **blue** (glucagon on INS/GCG/Ki-67 slides, or the hematoxylin "
        "counterstain on INS/Ki-67 slides). A fourth signal, **tissue**, is the total of all stains. Each is "
        "scaled to 0–1 on a fixed scale that is the same for every slide, so one threshold means the same "
        "stain intensity on every donor." if brightfield else
        "Each fluorescence channel is scaled to 0–1 from the slide overview (1st to 99.8th percentile)."),
    "tissue": (
        "**Tissue** = anything that absorbs light. The total stain (all colours together) is smoothed over ~5 µm "
        "and kept where it is above the *Tissue threshold*; gaps up to ~10 µm are closed, holes filled, and "
        "pieces smaller than 2,000 µm² dropped. Empty glass, fat spaces and large lumens are excluded from all "
        "areas. This works whether or not nuclei are stained." if brightfield else
        f"**Tissue** = where there are nuclei. The {NUC} signal is smoothed over ~5 µm and kept where it is "
        "above half the *Nuclei threshold*; gaps up to ~10 µm are closed, holes filled, and pieces smaller "
        "than 2,000 µm² dropped. Empty glass, fat spaces and large lumens are excluded from all areas."),
    "islets": (
        f"**Islets / endocrine.** 1) Take {ISL}, smoothed over ~2 µm. 2) Keep pixels above the *INS threshold* "
        "(insulin) or the *GCG threshold* (glucagon) inside tissue. 3) Close gaps up to ~4 µm and fill holes, so "
        "a stained islet becomes one solid shape including unstained cells inside it. 4) Drop shapes smaller "
        "than *Smallest islet*. Each remaining shape is one islet; together they are the endocrine area. "
        "Note: islets that have lost insulin are found only on slides with glucagon; on INS/Ki-67 slides "
        "they count as exocrine."),
    "exocrine": (
        "**Exocrine** = tissue minus islets: acini, ducts, vessels, stroma and anything else in the tissue "
        "outline that is not an islet."),
    "nuclei": (
        f"**All nuclei** (for the Ki-67 index): {NUC} smoothed over ~2 µm; each local peak at least ~3 µm "
        "from the next, above the *Nuclei threshold*, counts as one nucleus." if HAS_NUC else
        "**All nuclei: not counted on this slide.** It has no nuclear stain (nPOD INS/GCG/Ki-67 IHC), so "
        "there is no Ki-67 index; Ki-67+ per mm² is the measure to compare."),
    "ki67": (
        f"**Ki-67+ nuclei**: {KI} smoothed over ~1 µm, kept above the *Ki-67 threshold* inside tissue; "
        "touching pixels form one blob and blobs smaller than *Smallest Ki-67+ nucleus* are dropped. Each "
        "blob is one Ki-67+ nucleus, assigned to endocrine or exocrine by where its centre falls. "
        "Touching Ki-67+ nuclei merge into one, so dense clusters are undercounted."),
    "metrics": (
        "Per compartment: **Area (mm²)**; **Nuclei**; **Ki-67+ nuclei**; **Ki-67+ per mm²** = Ki-67+ "
        "nuclei ÷ area; **Ki-67 index (%)** = Ki-67+ nuclei ÷ all nuclei × 100; **Ki-67+ area (%)** = "
        "share of the compartment covered by Ki-67 stain. For a whole section, counts and areas are summed "
        "over tiles before dividing."),
    "islet_table": (
        "One row per detected islet: its area, the % of its area above the INS/GCG threshold for each "
        "marker, its Ki-67+ nuclei and its centre (row, column in the region). The cyan islet outlines and "
        "orange Ki-67 circles in step 3 show exactly what was counted."),
}
h1, h2, h3, h4, h5, h6 = st.columns(6)
h1.markdown("**Stains**", help=HELP["stains"])
h2.markdown("**Tissue**", help=HELP["tissue"])
h3.markdown("**Islets (endocrine)**", help=HELP["islets"])
h4.markdown("**Exocrine**", help=HELP["exocrine"])
h5.markdown("**All nuclei**", help=HELP["nuclei"])
h6.markdown("**Ki-67+ nuclei**", help=HELP["ki67"])
if roles["ins"] == NONE and roles["gcg"] == NONE:
    st.warning("No islet marker (INS or GCG) on this slide, so all tissue counts as exocrine.")
if ss.k_th is None:
    ss.k_th = suggest_thresholds(planes_of(fl_ov), TISSUE_TH if brightfield else None)
th0: Thresholds = ss.k_th
t1, t2, t3, t4 = st.columns(4)
nuc_th = t1.slider("Nuclei threshold", 0.0, 1.0, round(th0.dapi, 2), 0.01, key="k_th_dapi", disabled=not HAS_NUC,
                   help=(f"Minimum {NUC} signal for a nucleus." + ("" if brightfield else
                         " Half of it marks tissue. Higher = fewer nuclei and a tighter tissue outline."))
                   if HAS_NUC else "Not used: this slide has no nuclear stain.")
tissue_th = t1.slider("Tissue threshold", 0.0, 1.0, round(th0.tissue if th0.tissue is not None else TISSUE_TH, 2),
                      0.01, key="k_th_tissue", help="Minimum total stain for tissue (glass ≈ 0.1, tissue ≈ 0.6). "
                      "Higher = tighter tissue outline that leaves out pale stroma.") if brightfield else None
ins_th = t2.slider("INS threshold" if brightfield or roles["gcg"] != NONE else "INS/GCG threshold", 0.0, 1.0,
                   round(th0.endocrine, 2), 0.01, key="k_th_endo", disabled=roles["ins"] == NONE,
                   help="Minimum insulin signal for an islet pixel. Lower = larger islets and more faint islets "
                        "found; too low picks up background.")
gcg_th = t2.slider("GCG threshold", 0.0, 1.0,
                   round(th0.gcg if th0.gcg is not None else GCG_TH if brightfield else th0.endocrine, 2), 0.01,
                   key="k_th_gcg", help="Minimum glucagon (blue) signal for an islet pixel. Keep it above the "
                   "faint purple background in nuclei, or every nucleus-rich area becomes 'islet'."
                   + (" This slide has no glucagon; the value is used for INS/GCG/Ki-67 slides in "
                      "'Measure all donors'." if roles["gcg"] == NONE else "")) \
    if roles["gcg"] != NONE or brightfield else None
ki_th = t3.slider("Ki-67 threshold", 0.0, 1.0, round(th0.ki67, 2), 0.01, key="k_th_ki",
                  help=f"Minimum {KI} signal for a Ki-67+ nucleus. Lower = more (fainter) nuclei counted; "
                       "too low counts background stain.")
th = Thresholds(
    dapi=nuc_th, endocrine=ins_th, ki67=ki_th, tissue=tissue_th, gcg=gcg_th,
    min_islet_um2=t4.number_input("Smallest islet (µm²)", 0.0, 20000.0, th0.min_islet_um2, 50.0, key="k_th_islet",
                                  help="Stained shapes smaller than this are not islets (single cells, debris). "
                                       "300 µm² ≈ 2–3 cells."),
    min_nucleus_um2=t4.number_input("Smallest Ki-67+ nucleus (µm²)", 1.0, 200.0, th0.min_nucleus_um2, 1.0,
                                    key="k_th_nuc",
                                    help="Ki-67 blobs smaller than this are ignored as specks. A nucleus is "
                                         "typically 20–60 µm²."))
if st.button("Reset thresholds from this section", help="Otsu thresholds from the section overview"):
    ss.k_th = suggest_thresholds(planes_of(fl_ov), TISSUE_TH if brightfield else None)
    for k in ("k_th_dapi", "k_th_endo", "k_th_ki", "k_th_gcg", "k_th_tissue"):
        ss.pop(k, None)
    st.rerun()

b1, b2, b3 = st.columns([1, 1, 2])
if b1.button("Measure this region", type="primary"):
    ss.k_result = measure(pair_planes, pair.mpp, th)
    ss.k_result_key = (pair_key, th)
    st.rerun()
an_mpp = b3.select_slider("Whole-section resolution (µm per pixel)", [0.5, 1.0, 2.0], 1.0, key="k_an_mpp",
                          help="1 µm/px keeps nuclei countable and is about 4× faster than 0.5.")
whole = b2.button("Measure whole section", help="Tiles the whole fluorescence section and saves the result "
                                                 "to the cohort table. Can take several minutes.")

res = ss.k_result
if res is not None:
    r1, r2 = st.columns([3, 2])
    r1.markdown("**This region**", help=HELP["metrics"])
    r1.dataframe(res.metrics.round(3), hide_index=True, width="stretch")
    r2.markdown(f"**Islets in this region ({len(res.islets)})**", help=HELP["islets"] + "\n\n" + HELP["islet_table"])
    r2.dataframe(res.islets.round(1), hide_index=True, width="stretch", height=220)

if whole:
    bar = st.progress(0.0, "Measuring")
    total = measure_section(fl, CHANS, MISSING, th, an_mpp, progress=lambda f, m: bar.progress(f, m.capitalize()))
    bar.empty()
    if total is None:
        st.error("No tissue found with the current nuclei threshold.")
    else:
        save_result(RESULTS, total, donor, stage, fl.name, he.name, th, an_mpp,
                    "+".join(r for r in ROLES[1:] if roles[r] != NONE))
        ss.k_saved.add((str(donor), fl.name))
        st.success(f"Saved donor {donor} ({stage}) to the cohort table.")
        st.dataframe(total.round(3), hide_index=True, width="stretch")


def batch_roles(sec: Section) -> tuple[list[int], list[int], str] | None:
    """Channels to read (ROLES order, then tissue), planes to blank, and the markers, for one batch slide.

    Each slide's markers come from its own file name (INS_Ki67 or INS_GCG_Ki67), not from the slide above.
    """
    if isinstance(sec, StainSeparated):
        r, tch = brightfield_roles(sec.name), StainSeparated.TISSUE
    else:
        r = {k: guess_channel(sec.channels, *ROLE_GUESS[k]) for k in ROLES}
        if r["dapi"] is None or r["ki67"] is None:
            return None
        r = {k: NONE if v is None else v for k, v in r.items()}
        tch = r["dapi"]
    if r["ki67"] == NONE:
        return None
    chans = [r[k] if r[k] != NONE else tch for k in ROLES] + [tch]
    return chans, [i for i, k in enumerate(ROLES) if r[k] == NONE], "+".join(k for k in ROLES[1:] if r[k] != NONE)


with st.container(border=True):
    st.markdown("**Measure all donors**")
    st.caption("One button measures every donor with the thresholds set above, one after another, and saves "
               "each to the cohort table as soon as it finishes. Each slide's markers come from its own file name "
               "(INS_Ki67 or INS_GCG_Ki67), the stage from the folder (ND, sAAB, mAAB, T1D) and the donor from "
               "the number in the file name.")
    batch_src = st.radio("Slides", ["FUSION project folder", "Local folder"], horizontal=True, key="k_batch_src")
    donors = None
    if batch_src == "Local folder":
        root = st.text_input("Folder with all donors", key="k_batch_root",
                             placeholder=r"C:\Users\you\Documents\nPOD_slides").strip().strip('"')
        if root and not Path(root).is_dir():
            st.error("Folder not found. Paste the folder that contains ND, sAAB, mAAB and T1D.")
        elif root:
            donors = discover(root)
    elif ss.archive is None or not ss.get("k_tree_key"):
        st.info("Connect to FUSION and paste the project folder link in step 1.")
    else:
        donors = discover_archive(ss.k_tree)
        st.caption(f"From **{ss.k_tree_name}** on FUSION. Slides are read over the network, so expect "
                   "several minutes per donor.")
    if donors is not None:
        if donors.empty:
            st.warning("No slides found. Expected folders such as ND/IHC_ND and ND/H&E_ND with slide files "
                       "whose names contain the donor number.")
        else:
            done_set = measured(RESULTS)
            donors["measured"] = [(d, n) in done_set for d, n in zip(donors["donor"], donors["IHC slide"])]
            ready = donors[donors["status"].str.startswith(("Ready", "2", "3"))]
            c1, c2, c3 = st.columns(3)
            c1.metric("Donors found", len(donors))
            c2.metric("Ready to measure", len(ready))
            c3.metric("Already measured", int(donors["measured"].sum()))
            st.dataframe(donors.drop(columns=["ihc_ref", "he_ref"]), hide_index=True, width="stretch",
                         height=min(400, 38 + 35 * len(donors)))
            redo = st.checkbox("Measure again donors already in the cohort table", False, key="k_batch_redo")
            todo = ready if redo else ready[~ready["measured"]]
            st.caption(f"{len(todo)} donors to measure. Roughly 2–10 minutes per whole slide at "
                       f"{an_mpp} µm/px. Keep this page open and do not change settings while it runs; "
                       "if it stops, press the button again and it continues with the donors not yet measured.")
            if st.button(f"Measure {len(todo)} donors", type="primary", disabled=todo.empty, key="k_batch_go"):
                overall = st.progress(0.0, "Starting")
                status = st.empty()
                log = []
                for n, rec in enumerate(todo.to_dict("records"), 1):
                    label = f"{rec['stage']} {rec['donor']}"
                    try:
                        if batch_src == "Local folder":
                            sec = LocalSection(rec["ihc_ref"])
                        else:
                            sec = ArchiveSection(ss.archive, {"_id": rec["ihc_ref"], "name": rec["IHC slide"]})
                        if sec.rgb:
                            sec = StainSeparated(sec)
                        sec.calibrate()
                        if sec.mpp is None:
                            raise ValueError("pixel size not recorded in the file")
                        plan = batch_roles(sec)
                        if plan is None:
                            raise ValueError("could not find the nuclei and Ki-67 channels")
                        chans, missing, markers = plan
                        th_b = th
                        if isinstance(sec, StainSeparated) and th.tissue is None:  # thresholds set on fluorescence
                            th_b = Thresholds(**{**th.__dict__, "tissue": TISSUE_TH})
                        tot = measure_section(sec, chans, missing, th_b, an_mpp,
                                              progress=lambda f, m: overall.progress(
                            min(1.0, (n - 1 + f) / len(todo)), f"Donor {n}/{len(todo)} ({label}): {m}"))
                        if tot is None:
                            raise ValueError("no tissue found with the current tissue threshold")
                        save_result(RESULTS, tot, rec["donor"], rec["stage"], rec["IHC slide"], rec["H&E slide"],
                                    th_b, an_mpp, markers)
                        endo = tot.set_index("Compartment")["Ki-67+ per mm²"].get("Endocrine")
                        log.append({"donor": label, "result": "saved", "endocrine Ki-67+/mm²":
                                    None if endo is None else round(float(endo), 2)})
                    except Exception as e:  # one bad slide must not stop the cohort
                        log.append({"donor": label, "result": f"failed: {e}", "endocrine Ki-67+/mm²": None})
                    status.dataframe(pd.DataFrame(log), hide_index=True, width="stretch")
                overall.progress(1.0, "Finished")
                n_ok = sum(r["result"] == "saved" for r in log)
                (st.success if n_ok == len(log) else st.warning)(
                    f"{n_ok} of {len(log)} donors measured and saved. The cohort chart below includes them.")

if ss.k_result is not None or (str(donor), fl.name) in ss.k_saved:
    done.add("quantify")


cohort()

# ================================================================ 5. digital staining
st.header("5 · Digital staining", anchor="stain",
          help="Digital staining here transfers the real IHC onto the H&E through the registration. Bright "
               "colours: each marker is laid over the H&E in a light colour (islets cyan, glucagon lime, Ki-67 "
               "yellow), stronger where the stain is stronger. IHC colours: each chromogen (insulin red, "
               "glucagon blue, Ki-67 brown) is added as absorbance, like a real double stain. Choose whether to "
               "transfer all stain as on the IHC or only what step 4 counts.")
st.caption("The immunostain (ground truth) registered onto the consecutive H&E: islets (insulin) in cyan, "
           + ("glucagon in lime, " if roles["gcg"] != NONE else "")
           + "Ki-67+ nuclei in yellow, on the region chosen in steps 2–3 (use **Go to the region with the most "
             "islets** in step 2 to find islets). Uses the thresholds from step 4.")
if ss.k_result is not None and ss.get("k_result_key") == (pair_key, th):
    ds_res = ss.k_result
elif ss.get("k_ds_key") == (pair_key, th):
    ds_res = ss.k_ds_res
else:
    with st.spinner("Detecting islets and Ki-67+ nuclei in this region..."):
        ds_res = measure(pair_planes, pair.mpp, th, per_islet=False)
    ss.k_ds_res, ss.k_ds_key = ds_res, (pair_key, th)
ds_planes = {r: pair_fl[..., k] for k, r in enumerate(ROLES) if r != "dapi" and roles[r] != NONE}
ds_mode = st.radio("Paint", ["All stain, as on the IHC", "Only what step 4 counts"], horizontal=True,
                   key="k_ds_mode",
                   help="All stain: every bit of insulin and glucagon stain above the slide background is "
                        "transferred, including faint islets below the islet threshold. Only what step 4 counts: "
                        "islet pixels above the INS/GCG thresholds. Ki-67 is always the nuclei counted in step 4 "
                        "(faint brown is mostly bleed-through from dark nuclei).")
DS_FLOOR = {"ins": 0.03, "gcg": 0.2}  # slide background of each islet chromogen (0..1 scale)
ds_masks = {}
for r in ds_planes:
    if r == "ki67":
        ds_masks[r] = cv2.dilate(ds_res.masks["ki67"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    elif ds_mode.startswith("All"):
        ds_masks[r] = (ds_planes[r] > DS_FLOOR[r]) & ds_res.masks["tissue"]
    else:
        sm = cv2.GaussianBlur(ds_planes[r], (0, 0), max(0.5, 1.0 / (pair.mpp or 1.0)))
        cut = th.gcg if r == "gcg" and th.gcg is not None else th.endocrine
        ds_masks[r] = (sm > cut) & ds_res.masks["tissue"]

s1, s2, s3 = st.columns([2, 2, 3])
ds_show = {r: s1.checkbox(ROLE_LABEL[r], True, key=f"k_ds_{r}") for r in ds_planes}
ds_style = s2.radio("Colours", ["Bright (cyan, lime, yellow)", "Like the IHC (red, blue, brown)"],
                    key="k_ds_style", help="Bright colours stand out on the pink and purple H&E.")
ds_strength = s2.slider("Strength", 0.2, 2.0, 1.0, 0.1, key="k_ds_strength",
                        help="Higher = more opaque (bright colours) or darker (IHC colours).")
ds_view = s3.radio("Compare with", ["H&E and IHC", "IHC only", "Digital stain only"], horizontal=True,
                   key="k_ds_view")
bright = ds_style.startswith("Bright")
paint_planes = {r: p for r, p in ds_planes.items() if ds_show[r]}
paint_masks = dict(ds_masks)
if bright and "ki67" in paint_planes:
    # single nuclei are a few pixels: enlarge them (~3 µm) so they stay visible when zoomed out
    k = max(3, int(2 * 3 / (pair.mpp or 1.0)) | 1)
    disk_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    paint_masks["ki67"] = cv2.dilate(ds_masks["ki67"].astype(np.uint8), disk_k) > 0
    paint_planes["ki67"] = np.maximum(cv2.dilate(paint_planes["ki67"], disk_k), 0.6)
painted = paint_stains(pair.he, paint_planes, paint_masks, ds_strength, "highlight" if bright else "chromogen")
ds_panels = [("Digitally stained H&E", painted)]
if ds_view == "H&E and IHC":
    ds_panels = [("H&E", pair.he)] + ds_panels
if ds_view != "Digital stain only":
    ds_panels.append(("IHC (as scanned, registered)", ihc_rgb) if ihc_rgb is not None else
                     ("Immunostain (registered)", (composite(pair_fl, list(ROLES), show) * 255).astype(np.uint8)))
h, w = pair.he.shape[:2]
fig = make_subplots(1, len(ds_panels), subplot_titles=[p[0] for p in ds_panels], horizontal_spacing=0.01)
for i, (_, img) in enumerate(ds_panels, 1):
    fig.add_trace(go.Image(source=to_src(img), hoverinfo="skip"), 1, i)
    sfx = "" if i == 1 else str(i)
    fig.update_xaxes(visible=False, range=[0, w], matches=None if i == 1 else "x", row=1, col=i)
    fig.update_yaxes(visible=False, range=[h, 0], matches=None if i == 1 else "y", scaleanchor=f"x{sfx}",
                     row=1, col=i)
fig.update_layout(height=520 if len(ds_panels) > 1 else 680, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, width="stretch", key="k_dsfig", config={"scrollZoom": True})
mm2 = (pair.mpp or 0) ** 2 / 1e6
areas = " · ".join(f"{ROLE_LABEL[r].split(' (')[0].lower()} painted on {int(m.sum()) * mm2:.3f} mm²"
                   for r, m in ds_masks.items() if r in ("ins", "gcg"))
st.caption(f"Local registration match {pair.score:.2f}" + (" (aligned again in this region)" if pair.refined else "")
           + (f" · {areas}" if areas else "") + f" · {len(ds_res.ki67_points)} Ki-67+ nuclei counted. "
           "Islets span both sections and line up well; single Ki-67+ nuclei are ~4 µm away in the other "
           "section, so they mark where proliferating cells are, not always the exact H&E nucleus."
           + (" Little or no islet stain in this region: pick another region in step 2." if not
              any(m.sum() * mm2 > 0.001 for r, m in ds_masks.items() if r in ("ins", "gcg")) else ""))
buf = io.BytesIO()
Image.fromarray(painted).save(buf, format="PNG")
st.download_button("Download digitally stained region (PNG)", buf.getvalue(),
                   f"digital_stain_{donor}_{int(roi[0])}_{int(roi[1])}.png", "image/png")
done.add("stain")

with st.expander("Later: train a model to stain H&E without IHC (optional)"):
    st.caption("To predict INS, GCG and Ki-67 from H&E alone, a model (e.g. trained on HiPerGator) needs "
               "registered H&E / IHC tile pairs; these tools export them and score the model's predictions.")
    tab_x, tab_e = st.tabs(["Export training pairs", "Evaluate predictions"])
with tab_x:
    e1, e2, e3, e4 = st.columns(4)
    tile_px = e1.select_slider("Tile size (px)", [256, 512, 1024], 512, key="k_x_px")
    tile_um = e2.select_slider("Tile width (µm)", [128, 256, 512], 256, key="k_x_um",
                               help="256 µm at 512 px = 0.5 µm/px")
    min_tissue = e3.slider("Minimum tissue in tile", 0.1, 1.0, 0.5, 0.05, key="k_x_tissue")
    min_score = e4.slider("Minimum local match", 0.0, 1.0, 0.2, 0.05, key="k_x_score",
                          help="Tiles that do not align well are listed in the manifest but not written.")
    e5, e6 = st.columns([1, 3])
    max_tiles = e5.number_input("Max tiles (0 = all)", 0, 100000, 50, 10, key="k_x_max")
    out_dir = Path(e6.text_input("Output folder", str(PAIRS), key="k_x_out"))
    if st.button("Export pairs for this donor"):
        bar = st.progress(0.0, "Exporting")
        man = export_pairs(he, fl, T, roles, he_ov, out_dir, donor, stage, tile_px, tile_um, min_tissue,
                           min_score, int(max_tiles) or None, progress=lambda f, m: bar.progress(f, m),
                           reg_ch=TCH, signal=SIGNAL)
        bar.empty()
        kept = man[(man["donor"] == donor) & man["kept"]]
        ss.k_exported.add(str(donor))
        st.success(f"Wrote {len(kept)} tile pairs to {out_dir / 'tiles'}. "
                   f"Manifest: {out_dir / 'manifest.csv'}")
        if len(kept):
            import tifffile
            show_n = kept.head(4)
            cols = st.columns(len(show_n))
            for col, (_, m) in zip(cols, show_n.iterrows()):
                hei = np.asarray(Image.open(out_dir / "tiles" / f"{m['tile']}_he.png"))
                fli = tifffile.imread(out_dir / "tiles" / f"{m['tile']}_fl.tif").astype(np.float32) / 255
                comp = (composite(np.moveaxis(fli, 0, -1), list(FL_ORDER),
                                  {"dapi": True, "ins": True, "gcg": True, "ki67": True}) * 255).astype(np.uint8)
                col.image([hei, comp], caption=["H&E", "Real stain"], width=140)
    st.caption("Split training and test sets **by donor**, never by tile: tiles from one donor are correlated.")

with tab_e:
    st.markdown("Put the model's outputs in one folder as `<tile>_pred.tif`, channels **INS, GCG, Ki-67** "
                "(uint8 or 0–1 float), named like the exported tiles.")
    pred_dir = Path(st.text_input("Predictions folder", str(PAIRS / "predictions"), key="k_e_pred"))
    pairs_dir = Path(st.text_input("Exported pairs folder", str(PAIRS), key="k_e_pairs"))
    if st.button("Evaluate", type="primary"):
        if not (pairs_dir / "manifest.csv").exists():
            st.error("No manifest.csv in the exported pairs folder.")
        elif not pred_dir.exists():
            st.error("Predictions folder not found.")
        else:
            with st.spinner("Comparing predicted and real stain..."):
                ss.k_eval = evaluate(pairs_dir, pred_dir, th)
    ev = ss.k_eval
    if ev is not None:
        if ev.empty:
            st.warning("No predictions matched the exported tiles. Check the file names.")
        else:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Tiles evaluated", len(ev))
            s2.metric("Islet overlap (Dice)", f"{ev['Islet Dice'].mean():.2f}")
            s3.metric("Ki-67 SSIM", f"{ev['ki67 SSIM'].mean():.2f}")
            s4.metric("Ki-67 Pearson r", f"{ev['ki67 r'].mean():.2f}")
            ds = donor_summary(ev)
            fig = go.Figure()
            lim = 0.0
            for comp, colr in (("Endocrine", "#2a78d6"), ("Exocrine", "#eb6834")):
                xr, yp = ds[f"{comp} Ki-67+/mm² real"], ds[f"{comp} Ki-67+/mm² predicted"]
                lim = max(lim, float(np.nanmax([xr.max(), yp.max(), 1])))
                fig.add_trace(go.Scatter(x=xr, y=yp, mode="markers", name=comp, customdata=ds["donor"],
                                         marker=dict(size=10, color=colr, line=dict(width=1, color="white")),
                                         hovertemplate="donor %{customdata}<br>real %{x:.1f} · predicted %{y:.1f}"
                                                       "<extra>" + comp + "</extra>"))
            fig.add_trace(go.Scatter(x=[0, lim * 1.1], y=[0, lim * 1.1], mode="lines", name="perfect agreement",
                                     line=dict(color="#9a9893", dash="dot", width=1), hoverinfo="skip"))
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1),
                              xaxis=dict(title="Real stain: Ki-67+ per mm²", rangemode="tozero"),
                              yaxis=dict(title="Digital stain: Ki-67+ per mm²", rangemode="tozero"))
            d1, d2 = st.columns([3, 2])
            d1.plotly_chart(fig, width="stretch")
            d2.markdown("**Per donor**")
            d2.dataframe(ds.round(2), hide_index=True, width="stretch")
            st.markdown("**Per tile**")
            st.dataframe(ev.round(3), hide_index=True, width="stretch", height=240)
            st.download_button("Download evaluation (CSV)", ev.to_csv(index=False).encode(),
                               "digital_staining_evaluation.csv", "text/csv")

show_timeline()
