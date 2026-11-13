"""Digital staining of H&E with INS, GCG and Ki-67.

Paint (current approach)
    The immunostain, registered onto the consecutive H&E section, is painted onto the
    H&E in bright highlight colours (or IHC chromogen colours): the H&E carries the IHC
    ground truth. See paint_stains().

For a later model (H&E -> INS, GCG, Ki-67):

Export
    Registered tiles from a section pair, written to `<out>/tiles/`:
    - `<tile>_he.png`  H&E, RGB uint8
    - `<tile>_fl.tif`  fluorescence, uint8, channels DAPI, INS, GCG, Ki-67 (CYX)
    plus `manifest.csv` with donor, stage, tile position and registration score.
    Split train/test by donor, never by tile: tiles from one donor are correlated.

Evaluate
    A model writes `<tile>_pred.tif` (channels INS, GCG, Ki-67; uint8 or 0..1 float)
    into a predictions folder. Each prediction is compared with the real stain:
    - image agreement per marker: Pearson r and SSIM
    - the project's success criterion: islet mask overlap (Dice) and Ki-67 density in
      endocrine and exocrine areas, measured the same way on real and predicted stain.
      Real DAPI is used for nuclei in both, so differences come from the predicted markers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity

from impactproject.quantify import Thresholds, measure
from impactproject.registration import read_pair, tissue_mask, hematoxylin
from impactproject.slides import Section

FL_ORDER = ("dapi", "ins", "gcg", "ki67")
PRED_ORDER = ("ins", "gcg", "ki67")


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# chromogen colours (optical density per R, G, B) used to paint markers onto H&E
PAINT_OD = {"ins": _unit([0.174, 0.809, 0.561]),   # red chromogen, as insulin on the nPOD IHC
            "gcg": _unit([0.850, 0.500, 0.150]),   # blue chromogen, as glucagon on the nPOD IHC
            "ki67": _unit([0.268, 0.570, 0.776])}  # DAB brown
# stain amount (OD) painted at full signal; close to the real IHC: insulin ~0.6, Ki-67 nuclei ~0.9
PAINT_STRENGTH = {"ins": 0.6, "gcg": 0.6, "ki67": 0.9}
# bright highlight colours (RGB) that stand out on pink/purple H&E: islets cyan, glucagon lime, Ki-67 yellow
HIGHLIGHT = {"ins": (0, 229, 255), "gcg": (140, 255, 60), "ki67": (255, 225, 0)}
PAINT_ORDER = ("ins", "gcg", "ki67")  # Ki-67 last, so dividing nuclei stay on top of islets


def paint_stains(he_rgb: np.ndarray, planes: dict[str, np.ndarray], masks: dict[str, np.ndarray],
                 strength: float = 1.0, style: str = "highlight") -> np.ndarray:
    """Digitally stain an H&E image with markers from the registered immunostain.

    `planes` maps 'ins', 'gcg', 'ki67' to 0..1 marker signal aligned to `he_rgb`; `masks`
    maps the same keys to where the marker is painted (islet mask, Ki-67 nuclei), so
    background is not painted.

    style 'highlight' (default): each marker is blended over the H&E in a bright colour
    (HIGHLIGHT), more opaque where the stain is stronger. style 'chromogen': each marker is
    added as absorbance (Beer–Lambert) in its IHC chromogen colour, like a real double stain.
    """
    keys = [k for k in PAINT_ORDER if planes.get(k) is not None and masks.get(k) is not None]
    if style == "chromogen":
        od = -np.log10(np.clip(he_rgb.astype(np.float32) / 255.0, 1 / 255.0, 1.0))
        for key in keys:
            amount = np.where(masks[key], np.clip(planes[key], 0, 1), 0).astype(np.float32)
            od += amount[..., None] * (PAINT_STRENGTH[key] * strength) * PAINT_OD[key].astype(np.float32)
        return np.clip(255.0 * 10.0 ** (-od), 0, 255).astype(np.uint8)
    out = he_rgb.astype(np.float32)
    for key in keys:
        # every painted pixel gets at least a light tint; strong stain is close to opaque
        a = np.where(masks[key], 0.35 + 0.65 * np.clip(planes[key], 0, 1), 0) * 0.75 * strength
        a = np.clip(a, 0, 0.95).astype(np.float32)[..., None]
        out = out * (1 - a) + np.array(HIGHLIGHT[key], np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def tile_grid(he: Section, he_overview: np.ndarray, tile_um: float, min_tissue: float) -> list[tuple]:
    """Level-0 boxes of tiles covering the H&E tissue."""
    mpp = he.mpp or 0.5
    size = tile_um / mpp
    s = he_overview.shape[1] / he.width
    mask = tissue_mask(hematoxylin(he_overview))
    boxes = []
    for top in np.arange(0, he.height - size + 1, size):
        for left in np.arange(0, he.width - size + 1, size):
            r0, r1 = int(top * s), max(int(top * s) + 1, int((top + size) * s))
            c0, c1 = int(left * s), max(int(left * s) + 1, int((left + size) * s))
            if mask[r0:r1, c0:c1].mean() >= min_tissue:
                boxes.append((left, top, left + size, top + size))
    return boxes


def export_pairs(he: Section, fl: Section, T: np.ndarray, roles: dict[str, int], he_overview: np.ndarray,
                 out_dir: Path, donor: str, stage: str, tile_px: int = 512, tile_um: float = 256,
                 min_tissue: float = 0.5, min_score: float = 0.2, max_tiles: int | None = None,
                 progress=None, reg_ch: int | None = None, signal: str = "nuclei") -> pd.DataFrame:
    """Write registered H&E / immunostain tile pairs and append them to the manifest.

    A role mapped to -1 (marker not on this slide) is written as an empty channel.
    `reg_ch` / `signal`: the channel and signal used for registration (default: DAPI, nuclei).
    """
    import tifffile

    reg_ch = roles["dapi"] if reg_ch is None else reg_ch
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    boxes = tile_grid(he, he_overview, tile_um, min_tissue)
    if max_tiles:
        rng = np.random.default_rng(0)
        boxes = [boxes[i] for i in sorted(rng.choice(len(boxes), min(max_tiles, len(boxes)), replace=False))]
    chans = [roles[r] if roles[r] >= 0 else reg_ch for r in FL_ORDER]
    missing = [k for k, r in enumerate(FL_ORDER) if roles[r] < 0]
    rows = []
    for n, box in enumerate(boxes, 1):
        pair = read_pair(he, fl, T, box, tile_px, chans, reg_ch, refine=True, signal=signal)
        pair.fl[..., missing] = 0
        keep = pair.score >= min_score
        name = f"{donor}_{he.name.split('.')[0]}_{int(box[0])}_{int(box[1])}"
        if keep:
            Image.fromarray(pair.he).save(tiles_dir / f"{name}_he.png")
            tifffile.imwrite(tiles_dir / f"{name}_fl.tif",
                             (pair.fl.transpose(2, 0, 1) * 255).round().astype(np.uint8),
                             metadata={"axes": "CYX", "Channel": {"Name": list(FL_ORDER)}})
        rows.append({"tile": name, "donor": donor, "stage": stage, "he_section": he.name,
                     "fl_section": fl.name, "left": int(box[0]), "top": int(box[1]),
                     "size_px_level0": int(box[2] - box[0]), "mpp": pair.mpp,
                     "registration_ncc": round(pair.score, 4), "refined": pair.refined, "kept": keep})
        if progress:
            progress(n / len(boxes), f"Tile {n}/{len(boxes)}")
    df = pd.DataFrame(rows)
    manifest = out_dir / "manifest.csv"
    if manifest.exists():
        old = pd.read_csv(manifest)
        df = pd.concat([old[~old["tile"].isin(df["tile"])], df], ignore_index=True)
    df.to_csv(manifest, index=False)
    return df


def _load(path: Path, n: int) -> np.ndarray:
    import tifffile
    a = tifffile.imread(path).astype(np.float32)
    if a.ndim == 2:
        a = a[None]
    if a.shape[0] != n and a.shape[-1] == n:
        a = np.moveaxis(a, -1, 0)
    return a / 255.0 if a.max() > 1.0 else a


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else np.nan


def evaluate(pairs_dir: Path, pred_dir: Path, th: Thresholds) -> pd.DataFrame:
    """Compare every `<tile>_pred.tif` in `pred_dir` with the real stain of that tile."""
    manifest = pd.read_csv(pairs_dir / "manifest.csv")
    manifest = manifest[manifest["kept"]]
    rows = []
    for _, m in manifest.iterrows():
        pred_path = pred_dir / f"{m['tile']}_pred.tif"
        if not pred_path.exists():
            continue
        real = _load(pairs_dir / "tiles" / f"{m['tile']}_fl.tif", len(FL_ORDER))
        pred = _load(pred_path, len(PRED_ORDER))
        rec = {"tile": m["tile"], "donor": m["donor"], "stage": m["stage"]}
        for k, role in enumerate(PRED_ORDER):
            r, p = real[FL_ORDER.index(role)], pred[k]
            rec[f"{role} r"] = float(np.corrcoef(r.ravel(), p.ravel())[0, 1]) if r.std() and p.std() else np.nan
            rec[f"{role} SSIM"] = float(structural_similarity(r, p, data_range=1.0))
        mpp = float(m["mpp"]) if pd.notna(m["mpp"]) else 0.5
        planes_r = dict(zip(FL_ORDER, real))
        planes_p = {"dapi": planes_r["dapi"], **dict(zip(PRED_ORDER, pred))}
        a, b = measure(planes_r, mpp, th, per_islet=False), measure(planes_p, mpp, th, per_islet=False)
        rec["Islet Dice"] = dice(a.masks["endocrine"], b.masks["endocrine"])
        for comp in ("Endocrine", "Exocrine"):
            ra = a.metrics.set_index("Compartment").loc[comp]
            rb = b.metrics.set_index("Compartment").loc[comp]
            rec[f"{comp} Ki-67+ real"] = int(ra["Ki-67+ nuclei"])
            rec[f"{comp} Ki-67+ predicted"] = int(rb["Ki-67+ nuclei"])
            rec[f"{comp} area real (mm²)"] = float(ra["Area (mm²)"])
            rec[f"{comp} area predicted (mm²)"] = float(rb["Area (mm²)"])
        rows.append(rec)
    return pd.DataFrame(rows)


def donor_summary(per_tile: pd.DataFrame) -> pd.DataFrame:
    """Ki-67 density per donor and compartment, real vs predicted, summed over tiles."""
    out = []
    for donor, g in per_tile.groupby("donor"):
        rec = {"donor": donor, "stage": g["stage"].iloc[0], "tiles": len(g)}
        for comp in ("Endocrine", "Exocrine"):
            for src in ("real", "predicted"):
                area = g[f"{comp} area {src} (mm²)"].sum()
                rec[f"{comp} Ki-67+/mm² {src}"] = g[f"{comp} Ki-67+ {src}"].sum() / area if area else np.nan
        rec["Islet Dice (mean)"] = g["Islet Dice"].mean()
        out.append(rec)
    return pd.DataFrame(out)


__all__ = ["FL_ORDER", "HIGHLIGHT", "PAINT_OD", "PRED_ORDER", "dice", "donor_summary", "evaluate", "export_pairs",
           "paint_stains", "tile_grid"]
