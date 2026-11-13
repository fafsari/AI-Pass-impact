"""Ki-67 density in endocrine (islet) and exocrine pancreas from multiplex fluorescence
or brightfield IHC separated into stains.

Compartments
- tissue: a 'tissue' plane (brightfield: total stain absorbance) above `th.tissue`,
  or, for fluorescence, DAPI above half the nuclei threshold; smoothed at tissue scale
- endocrine: insulin (INS) above `th.endocrine` or glucagon (GCG) above `th.gcg`, closed
  and hole-filled into islets; objects smaller than `min_islet_um2` are dropped
- exocrine: tissue that is not endocrine

Cells
- nuclei: local maxima of smoothed DAPI / counterstain. Slides without a nuclear stain
  (nPOD INS/GCG/Ki-67 IHC) have no nuclei count, so no Ki-67 index
- Ki-67+ nuclei: connected Ki-67 objects at least `min_nucleus_um2` in size,
  assigned to a compartment by their centre

Densities are reported per mm² of each compartment, with the Ki-67 index
(Ki-67+ nuclei as % of all nuclei) where nuclei are stained. All sizes are in microns
so results do not depend on the resolution the image was read at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import disk, remove_small_objects

ROLES = ("dapi", "ins", "gcg", "ki67")


@dataclass
class Thresholds:
    """Intensity thresholds on the 0..1 normalised channels, plus size limits in microns."""

    dapi: float = 0.15
    endocrine: float = 0.3  # insulin (and glucagon when `gcg` is None)
    ki67: float = 0.35
    min_islet_um2: float = 300.0
    min_nucleus_um2: float = 12.0
    tissue: float | None = None  # on the 'tissue' plane; None: tissue from DAPI (fluorescence)
    gcg: float | None = None  # glucagon threshold; None: same as `endocrine`


@dataclass
class Result:
    metrics: pd.DataFrame  # one row per compartment
    islets: pd.DataFrame  # one row per islet (regions only)
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # tissue, endocrine, exocrine, ki67
    ki67_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (row, col)


def tissue_of(planes: dict[str, np.ndarray], th: Thresholds, sigma_px: float) -> np.ndarray:
    """Raw tissue mask (before closing/filling): the tissue plane, or DAPI for fluorescence."""
    if th.tissue is not None and planes.get("tissue") is not None:
        return ndi.gaussian_filter(planes["tissue"], sigma_px) > th.tissue
    return ndi.gaussian_filter(planes["dapi"], sigma_px) > th.dapi * 0.5


def suggest_thresholds(planes: dict[str, np.ndarray], tissue_th: float | None = None) -> Thresholds:
    """Otsu thresholds from an overview, computed inside the tissue.

    `tissue_th` (brightfield) outlines tissue on the planes' 'tissue' plane instead of DAPI.
    """
    t = Thresholds(tissue=tissue_th)
    dapi = ndi.gaussian_filter(planes["dapi"], 1)
    if dapi.max() > dapi.min():
        t.dapi = float(threshold_otsu(dapi))
    tissue = tissue_of(planes, t, 3)
    # brightfield glucagon (blue) is not auto-set: its faint purple background in nuclei would pull
    # Otsu down and turn nucleus-rich tissue into 'islet'; the app uses a fixed default instead
    for role in ("ins",) if tissue_th is not None else ("ins", "gcg"):
        sig = planes[role][tissue]
        if sig.size and sig.max() > sig.min():
            v = float(max(threshold_otsu(sig), 0.1))
            if role == "ins" or not (planes["ins"].max() > 0):
                t.endocrine = v
            else:
                t.gcg = v
    ki = planes["ki67"][tissue]
    if ki.size and ki.max() > ki.min():
        t.ki67 = float(max(threshold_otsu(ki), 0.15))
    return t


def _um(px: float, mpp: float) -> float:
    return px / mpp  # microns -> pixels


def measure(planes: dict[str, np.ndarray], mpp: float, th: Thresholds,
            per_islet: bool = True) -> Result:
    """Segment compartments and count Ki-67+ nuclei in one image region.

    `planes` maps each role in ROLES to a float 0..1 image, all the same shape, plus an
    optional 'tissue' plane (used when `th.tissue` is set). A role plane that is all zero
    means the marker is not on the slide; an all-zero 'dapi' means nuclei are not stained.
    `mpp` is microns per pixel of those images.
    """
    px_area_mm2 = (mpp / 1000) ** 2
    dapi, ins, gcg, ki = (planes[r] for r in ROLES)
    gcg_th = th.endocrine if th.gcg is None else th.gcg

    tissue = tissue_of(planes, th, _um(5, mpp))
    tissue = ndi.binary_fill_holes(ndi.binary_closing(tissue, disk(max(1, int(_um(10, mpp))))))
    tissue = remove_small_objects(tissue, max_size=max(1, int(2000 / mpp ** 2)) - 1)

    sig = _um(2, mpp)
    ins_pos = ndi.gaussian_filter(ins, sig) > th.endocrine
    gcg_pos = ndi.gaussian_filter(gcg, sig) > gcg_th
    endo = (ins_pos | gcg_pos) & tissue
    endo = ndi.binary_closing(endo, disk(max(1, int(_um(4, mpp)))))
    endo = ndi.binary_fill_holes(endo)
    endo = remove_small_objects(endo, max_size=max(1, int(th.min_islet_um2 / mpp ** 2)) - 1) & tissue
    exo = tissue & ~endo

    # all nuclei: peaks of smoothed DAPI / counterstain (none if nuclei are not stained)
    if dapi.max() > 0:
        dmask = ndi.gaussian_filter(dapi, _um(1, mpp)) > th.dapi
        peaks = peak_local_max(ndi.gaussian_filter(dapi, _um(2, mpp)), min_distance=max(1, int(_um(3, mpp))),
                               labels=dmask.astype(np.uint8), exclude_border=False)
    else:
        peaks = None

    # Ki-67+ nuclei
    kmask = (ndi.gaussian_filter(ki, _um(1, mpp)) > th.ki67) & tissue
    kmask = remove_small_objects(kmask, max_size=max(1, int(th.min_nucleus_um2 / mpp ** 2)) - 1)
    klab, nk = ndi.label(kmask)
    kpts = np.array(ndi.center_of_mass(kmask, klab, range(1, nk + 1))).reshape(-1, 2)

    def count(points, mask):
        if points is None:
            return np.nan
        if len(points) == 0:
            return 0
        r = np.clip(points[:, 0].round().astype(int), 0, mask.shape[0] - 1)
        c = np.clip(points[:, 1].round().astype(int), 0, mask.shape[1] - 1)
        return int(mask[r, c].sum())

    rows = []
    for name, m in (("Endocrine", endo), ("Exocrine", exo), ("Total tissue", tissue)):
        area = float(m.sum() * px_area_mm2)
        n_all, n_ki = count(peaks, m), count(kpts, m)
        rows.append({"Compartment": name, "Area (mm²)": area, "Nuclei": n_all, "Ki-67+ nuclei": n_ki,
                     "Ki-67+ per mm²": n_ki / area if area > 0 else np.nan,
                     "Ki-67 index (%)": 100 * n_ki / n_all if n_all and n_all == n_all else np.nan,
                     "Ki-67+ area (%)": 100 * float((kmask & m).sum()) / m.sum() if m.sum() else np.nan})
    metrics = pd.DataFrame(rows)

    islets = pd.DataFrame()
    if per_islet:
        lab, _ = ndi.label(endo)
        recs = []
        for p in regionprops(lab):
            sl = p.slice
            m = p.image
            recs.append({"Islet": p.label, "Area (µm²)": p.area * mpp ** 2,
                         "INS+ (%)": 100 * float(ins_pos[sl][m].mean()),
                         "GCG+ (%)": 100 * float(gcg_pos[sl][m].mean()),
                         "Ki-67+ nuclei": count(kpts, lab == p.label),
                         "Centre (row, col)": (int(p.centroid[0]), int(p.centroid[1]))})
        islets = pd.DataFrame(recs)
    return Result(metrics, islets, {"tissue": tissue, "endocrine": endo, "exocrine": exo, "ki67": kmask}, kpts)


def combine(tile_metrics: list[pd.DataFrame]) -> pd.DataFrame:
    """Sum counts and areas over tiles and recompute the densities."""
    df = pd.concat(tile_metrics).groupby("Compartment", sort=False)[["Area (mm²)", "Nuclei", "Ki-67+ nuclei"]] \
        .sum(min_count=1)  # nuclei stay empty (NaN) when the slide has no nuclear stain
    df["Ki-67+ per mm²"] = df["Ki-67+ nuclei"] / df["Area (mm²)"].replace(0, np.nan)
    df["Ki-67 index (%)"] = 100 * df["Ki-67+ nuclei"] / df["Nuclei"].replace(0, np.nan)
    return df.reset_index()


__all__ = ["ROLES", "Result", "Thresholds", "combine", "measure", "suggest_thresholds", "tissue_of"]
