"""Cohort helpers: find donors in a slide folder, measure whole sections, save results,
and combine them with the donor information spreadsheet for statistics.

Expected folder layout (local copy or FUSION; names are matched loosely):

    nPOD_slides/   (or Private/ on FUSION)
        ND/    H&E_ND/ ND6333_PB_H&E.svs        IHC_ND/ ND6333_PB_INS_Ki67.svs
        sAAB/  ...
        mAAB/  ...
        T1D/   ...

The stage comes from the first folder under the root, the section (H&E or IHC) from a
folder or file name, and the donor from the first number of four or more digits in the
file name. Each donor's IHC and H&E slides are paired by that number within a stage.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from impactproject.quantify import ROLES, Thresholds, combine, measure
from impactproject.slides import Section

SLIDE_EXT = (".svs", ".ndpi", ".tif", ".tiff", ".btf", ".scn")
STAGE_NAMES = {"nd": "ND", "saab": "sAAb+", "maab": "mAAb+", "t1d": "Recent-onset T1D"}
# nPOD donor_type values in the donor information spreadsheet
DONOR_TYPE_STAGE = {"no diabetes": "ND", "gada positive": "sAAb+", "single autoantibody positive": "sAAb+",
                    "multiple autoantibody positive": "mAAb+", "type 1 diabetes": "Recent-onset T1D",
                    "t1d": "Recent-onset T1D"}
_DONOR = re.compile(r"(?<!\d)(\d{4,})(?!\d)")


def stage_of(name: str) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    for k, v in STAGE_NAMES.items():
        if key.startswith(k):
            return v
    return None


def section_of(folders: tuple[str, ...], filename: str) -> str:
    """'H&E' or 'IHC' from the folder names, falling back to the file name."""
    for part in [p.lower() for p in folders]:
        if "ihc" in part or "immun" in part or part.startswith("if"):
            return "IHC"
        if "h&e" in part or re.fullmatch(r"h[_ ]?e(_.*)?", part) or "hande" in part:
            return "H&E"
    return "H&E" if re.search(r"h&e|_he([_.]|$)", filename, re.I) else "IHC"


def discover_entries(entries: list[tuple[tuple[str, ...], str]]) -> pd.DataFrame:
    """One row per (stage, donor) from slides given as (folder names + file name, reference).

    The reference is whatever opens the slide later: a local path or an archive item id.
    """
    found: dict[tuple[str, str], dict] = {}
    for parts, ref in sorted(entries):
        name = parts[-1]
        stage = (stage_of(parts[0]) if len(parts) > 1 else None) or stage_of(name)
        m = _DONOR.search(Path(name).stem)
        if not (stage and m):
            continue
        rec = found.setdefault((stage, m.group(1)), {"IHC": [], "H&E": []})
        rec[section_of(parts[:-1], name)].append((name, ref))
    rows = []
    for (stage, donor), rec in found.items():
        ihc, he = rec["IHC"], rec["H&E"]
        if not ihc:
            status = "No IHC slide"
        elif len(ihc) > 1:
            status = f"{len(ihc)} IHC slides: using the first"
        elif not he:
            status = "Ready (no H&E slide)"
        else:
            status = "Ready"
        rows.append({"stage": stage, "donor": donor,
                     "IHC slide": ihc[0][0] if ihc else "", "H&E slide": he[0][0] if he else "",
                     "status": status, "ihc_ref": ihc[0][1] if ihc else "", "he_ref": he[0][1] if he else ""})
    order = {s: i for i, s in enumerate(STAGE_NAMES.values())}
    df = pd.DataFrame(rows, columns=["stage", "donor", "IHC slide", "H&E slide", "status", "ihc_ref", "he_ref"])
    return df.sort_values(["stage", "donor"], key=lambda c: c.map(order) if c.name == "stage" else c,
                          ignore_index=True)


def discover(root: str | Path) -> pd.DataFrame:
    """Donors in a local folder (see the module docstring for the layout)."""
    root = Path(root)
    entries = [(f.relative_to(root).parts, str(f)) for f in root.rglob("*")
               if f.is_file() and f.name.lower().endswith(SLIDE_EXT)]
    return discover_entries(entries)


def discover_archive(tree: list[dict]) -> pd.DataFrame:
    """Donors in an archive folder tree, as returned by Archive.walk()."""
    return discover_entries([(f["path"] + (i["name"],), i["_id"]) for f in tree for i in f["slides"]])


# Bump when the measurement changes: rows of an older method are measured again by the batch.
# v2: brightfield markers from each file name (blue = glucagon on INS/GCG/Ki-67 slides, which have
#     no nuclear stain), tissue from total stain absorbance, separate glucagon threshold.
METHOD = "v2"


def planes_of(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Planes read in ROLES order plus the tissue plane (5th) -> dict for measure()."""
    planes = dict(zip(ROLES, np.moveaxis(arr[..., :len(ROLES)], -1, 0)))
    planes["tissue"] = arr[..., len(ROLES)] if arr.shape[-1] > len(ROLES) else None
    return planes


def measure_section(fl: Section, chans: list[int], missing: list[int], th: Thresholds, an_mpp: float,
                    progress=None) -> pd.DataFrame | None:
    """Tile the whole immunostained section and return per-compartment totals (None: no tissue).

    `chans` are the channels to read in ROLES order followed by the tissue channel; planes at
    `missing` are blanked (markers not on this slide). `progress(fraction, text)` is called
    after each tile.
    """
    tile = int(2048 * an_mpp / fl.mpp)  # level-0 px per tile, ~2048 px read
    boxes = [(x, y, min(x + tile, fl.width), min(y + tile, fl.height))
             for y in range(0, fl.height, tile) for x in range(0, fl.width, tile)]
    parts = []
    for n, bx in enumerate(boxes, 1):
        wpx = max(8, int((bx[2] - bx[0]) * fl.mpp / an_mpp))
        arr = fl.read(bx, wpx, chans)
        arr[..., missing] = 0
        planes = planes_of(arr)
        empty = (planes["tissue"].max() < th.tissue) if th.tissue is not None and planes["tissue"] is not None \
            else planes["dapi"].max() < th.dapi * 0.5
        if not empty:  # skip empty glass
            mpp_t = (bx[2] - bx[0]) * fl.mpp / wpx
            parts.append(measure(planes, mpp_t, th, per_islet=False).metrics)
        if progress:
            progress(n / len(boxes), f"tile {n}/{len(boxes)}")
    return combine(parts) if parts else None


def save_result(results: Path, total: pd.DataFrame, donor: str, stage: str, fl_name: str, he_name: str,
                th: Thresholds, an_mpp: float, markers: str = ""):
    """Write one donor's rows to the cohort table, replacing an earlier run of the same slide."""
    meta = {"donor": str(donor), "stage": stage, "fl_section": fl_name, "he_section": he_name,
            "markers": markers, "method": METHOD,
            "analysis_mpp": an_mpp, "dapi_th": th.dapi, "endocrine_th": th.endocrine, "gcg_th": th.gcg,
            "ki67_th": th.ki67, "tissue_th": th.tissue,
            "min_islet_um2": th.min_islet_um2, "min_nucleus_um2": th.min_nucleus_um2}
    rows = pd.DataFrame([{**meta, **rec} for rec in total.to_dict("records")])
    results.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(results, dtype={"donor": str}) if results.exists() else pd.DataFrame()
    if len(old):
        old = old[~((old["donor"].astype(str) == str(donor)) & (old["fl_section"] == fl_name))]
    pd.concat([old, rows], ignore_index=True).to_csv(results, index=False)


def measured(results: Path) -> set[tuple[str, str]]:
    """(donor, immunostain file name) pairs already in the cohort table with the current method."""
    if not results.exists():
        return set()
    df = pd.read_csv(results, dtype={"donor": str})
    if "method" not in df:
        return set()
    df = df[df["method"].astype(str) == METHOD]
    return set(zip(df["donor"].astype(str), df["fl_section"]))


# ---------------------------------------------------------------- donor information and statistics
def donor_key(v) -> str:
    """Normalise a donor ID so 6333, 6333.0, 'nPOD 6333' and '6333 ' all match."""
    if pd.isna(v):
        return ""
    s = str(v).strip()
    s = re.sub(r"\.0$", "", s)
    m = _DONOR.search(s)
    return m.group(1) if m else s.upper()


def load_donor_info(src, sheet=0) -> pd.DataFrame:
    """Read the donor spreadsheet (.xlsx or .csv), drop blank rows, add `donor` and `info_stage`."""
    name = getattr(src, "name", str(src)).lower()
    df = pd.read_csv(src) if name.endswith(".csv") else pd.read_excel(src, sheet_name=sheet)
    df = df.dropna(how="all")
    id_col = next((c for c in df.columns if str(c).lower() in ("case_id", "donor", "donor_id", "case id")),
                  df.columns[0])
    df = df[df[id_col].notna()].copy()
    df.insert(0, "donor", df[id_col].map(donor_key))
    if "donor_type" in df.columns:
        df.insert(1, "info_stage", df["donor_type"].map(lambda t: DONOR_TYPE_STAGE.get(str(t).strip().lower())))
    return df


def compare_groups(df: pd.DataFrame, group: str, value: str) -> tuple[str, pd.DataFrame]:
    """Kruskal–Wallis across groups and pairwise Mann–Whitney U with Holm correction."""
    d = df[[group, value]].dropna()
    groups = {g: v[value].to_numpy() for g, v in d.groupby(group, sort=False)}
    groups = {g: v for g, v in groups.items() if len(v)}
    if len(groups) < 2:
        return "Needs at least two groups with data.", pd.DataFrame()
    summary = ""
    if len(groups) > 2 and all(len(v) >= 2 for v in groups.values()):
        h, p = stats.kruskal(*groups.values())
        summary = f"Kruskal–Wallis H = {h:.2f}, p = {p:.3g} ({len(groups)} groups, n = {len(d)})"
    rows = []
    for a, b in combinations(groups, 2):
        if len(groups[a]) >= 2 and len(groups[b]) >= 2:
            u, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
            rows.append({"group A": a, "group B": b, "n A": len(groups[a]), "n B": len(groups[b]),
                         "median A": np.median(groups[a]), "median B": np.median(groups[b]), "U": u, "p": p})
    pw = pd.DataFrame(rows)
    if len(pw):
        # Holm step-down adjustment
        order = np.argsort(pw["p"].to_numpy())
        m, adj, run = len(pw), np.empty(len(pw)), 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * pw["p"].iloc[i]))
            adj[i] = run
        pw["p (Holm)"] = adj
        if not summary and len(pw) == 1:
            summary = f"Mann–Whitney U = {pw['U'].iloc[0]:.1f}, p = {pw['p'].iloc[0]:.3g} (n = {len(d)})"
    return summary or "Each group needs at least two donors for a test.", pw


def correlate(df: pd.DataFrame, x: str, y: str) -> str:
    d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 4:
        return "Needs at least four donors with both values."
    rho, p = stats.spearmanr(d[x], d[y])
    return f"Spearman ρ = {rho:.2f}, p = {p:.3g} (n = {len(d)})"


def paired_compartments(df: pd.DataFrame, value: str) -> str:
    """Wilcoxon signed-rank: endocrine vs exocrine within the same donor."""
    w = df.pivot_table(index="donor", columns="Compartment", values=value, aggfunc="mean")
    if not {"Endocrine", "Exocrine"} <= set(w.columns):
        return ""
    w = w[["Endocrine", "Exocrine"]].dropna()
    if len(w) < 5:
        return "Endocrine vs exocrine needs at least five donors."
    s, p = stats.wilcoxon(w["Endocrine"], w["Exocrine"])
    return (f"Endocrine vs exocrine, same donor (Wilcoxon signed-rank): median {w['Endocrine'].median():.2f} "
            f"vs {w['Exocrine'].median():.2f}, p = {p:.3g} (n = {len(w)} donors)")


__all__ = ["METHOD", "STAGE_NAMES", "compare_groups", "planes_of", "correlate", "discover", "discover_archive", "discover_entries",
           "donor_key", "load_donor_info", "measure_section", "measured", "paired_compartments",
           "save_result", "stage_of"]
