"""Register an immunostained section (fluorescence or brightfield IHC) to its
consecutive H&E section.

Signal matched between the two sections:
- 'nuclei' (fluorescence): DAPI against H&E hematoxylin
- 'density' (brightfield IHC): total stain absorbance of both slides. nPOD INS/GCG/Ki-67
  slides have no nuclear counterstain, so tissue, lumens, ducts and septa are matched
  instead of nuclei.

1. Coarse (whole section, ~8 µm/px): try rotations every 10° with and without a
   flip, aligning tissue centroids, and keep the best tissue overlap. Then refine
   with an affine ECC fit, which absorbs small scale and shear differences between
   consecutive sections.
2. Fine (one region): the immunostain is read with a margin around the region, the
   offset is found by template matching on a smoothed copy (catches shifts of a few
   hundred µm left by the coarse fit), then refined by ECC from coarse to fine. Each
   stage is kept only if it improves the match.

A transform is a 3x3 matrix T mapping H&E level-0 pixels to fluorescence level-0
pixels. The fluorescence image is always resampled into the H&E frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2hed

from impactproject.slides import Box, Section

COARSE_MPP = 8.0  # microns per pixel for whole-section registration


def hematoxylin(rgb: np.ndarray) -> np.ndarray:
    """Hematoxylin optical density from an RGB H&E image, scaled to 0..1."""
    h = rgb2hed(rgb.astype(np.float32) / 255.0)[..., 0]
    lo, hi = np.percentile(h, [1, 99.5])
    return np.clip((h - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def density(rgb: np.ndarray) -> np.ndarray:
    """Total stain absorbance of an RGB image, scaled to 0..1 (tissue bright, glass dark)."""
    od = -np.log(np.clip(rgb.astype(np.float32) / 255.0, 1e-3, 1.0)).sum(-1)
    lo, hi = np.percentile(od, [1, 99.5])
    return np.clip((od - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def he_signal(rgb: np.ndarray, signal: str) -> np.ndarray:
    return density(rgb) if signal == "density" else hematoxylin(rgb)


def tissue_mask(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Foreground mask of a 0..1 image: smoothed Otsu threshold, holes filled."""
    from skimage.filters import threshold_otsu
    s = ndi.gaussian_filter(img, sigma)
    if s.max() <= s.min():
        return np.zeros(img.shape, bool)
    m = s > threshold_otsu(s)
    m = ndi.binary_closing(m, iterations=2)
    return ndi.binary_fill_holes(m)


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Normalised cross-correlation of two images (optionally inside a mask)."""
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _h(m23: np.ndarray) -> np.ndarray:
    return np.vstack([m23, [0, 0, 1]]).astype(np.float64)


def _scale(s: float) -> np.ndarray:
    return np.diag([s, s, 1.0])


def _translate(x: float, y: float) -> np.ndarray:
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)


def warp_into(img: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resample `img` into a target frame of `size` (w, h); M maps target px -> img px."""
    w, h = size
    planes = img if img.ndim == 3 else img[..., None]
    out = np.stack([cv2.warpAffine(planes[..., k], M[:2].astype(np.float32), (w, h),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for k in range(planes.shape[2])], -1)
    return out if img.ndim == 3 else out[..., 0]


def _ecc(template: np.ndarray, moving: np.ndarray, M: np.ndarray, motion=cv2.MOTION_AFFINE,
         iters: int = 200) -> np.ndarray | None:
    """ECC refinement; returns the refined 3x3 (target px -> moving px) or None."""
    warp = M[:2].astype(np.float32).copy()
    try:
        _, warp = cv2.findTransformECC(template.astype(np.float32), moving.astype(np.float32), warp,
                                       motion, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6),
                                       None, 5)
    except cv2.error:
        return None
    return _h(warp)


@dataclass
class Registration:
    """Result of registering a fluorescence section (moving) to an H&E section (fixed)."""

    T: np.ndarray  # 3x3, H&E level-0 px -> fluorescence level-0 px
    score_before: float  # NCC of nuclear signals at the best rigid start
    score_after: float  # NCC after affine refinement
    he_overview: np.ndarray  # RGB overview used for registration
    he_scale: float  # overview px per H&E level-0 px

    def with_adjustment(self, rot_deg: float = 0, dx_um: float = 0, dy_um: float = 0,
                        scale: float = 1.0, he_mpp: float | None = None, he_size=(0, 0)) -> np.ndarray:
        """Apply a manual nudge in H&E space (about the slide centre) and return the new T."""
        cx, cy = he_size[0] / 2, he_size[1] / 2
        px = 1.0 / (he_mpp or 1.0)
        a = np.deg2rad(rot_deg)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        N = _translate(cx, cy) @ R @ _scale(1 / scale) @ _translate(-cx, -cy) @ _translate(-dx_um * px, -dy_um * px)
        return self.T @ N


def _overview(sec: Section, mpp_target: float, max_px: int = 2048) -> tuple[np.ndarray, float]:
    mpp = sec.mpp or (sec.width / 1024)  # unknown size: ~1024 px wide overview
    w = int(min(max_px, max(256, sec.width * mpp / mpp_target)))
    return sec.read(None, w), w / sec.width


def register_sections(he: Section, fl: Section, dapi: int, mpp_target: float = COARSE_MPP,
                      progress=None, signal: str = "nuclei") -> Registration:
    """Coarse whole-section registration of the immunostain to H&E.

    `dapi` is the immunostain channel to match: DAPI ('nuclei') or total absorbance ('density').
    """
    he_img, s_he = _overview(he, mpp_target)
    fl_full, s_fl = _overview(fl, mpp_target)
    fixed = he_signal(he_img, signal)
    moving = fl_full[..., dapi]
    mf, mm = tissue_mask(fixed), tissue_mask(moving)
    if mf.sum() == 0 or mm.sum() == 0:
        raise ValueError("Could not find tissue in one of the sections.")

    # physical scale between the two overviews (1 if both mpp are known)
    k = 1.0
    if not (he.mpp and fl.mpp):
        k = np.sqrt(mm.sum() / mf.sum())  # match tissue areas when sizes are unknown
    cf = np.array(ndi.center_of_mass(mf))[::-1]
    cm = np.array(ndi.center_of_mass(mm))[::-1]

    small = 256 / max(fixed.shape)
    fs = cv2.resize(mf.astype(np.float32), None, fx=small, fy=small, interpolation=cv2.INTER_AREA)
    ms_img = mm.astype(np.float32)
    scored = []
    candidates = [(a, f) for f in (False, True) for a in range(0, 360, 10)]
    for n, (ang, flip) in enumerate(candidates):
        a = np.deg2rad(ang)
        O = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * k
        if flip:
            O = O @ np.diag([-1, 1])
        M = np.eye(3)
        M[:2, :2] = O
        M[:2, 2] = cm - O @ cf
        Ms = M @ _scale(1 / small)  # small-fixed px -> moving px
        scored.append((ncc(fs, warp_into(ms_img, Ms, (fs.shape[1], fs.shape[0]))), M))
        if progress:
            progress((n + 1) / len(candidates) * 0.5, "Searching rotation and flip")

    # Tissue outlines are often nearly symmetric, so several starts fit the outline
    # equally well. Refine the best few on the nuclear signal and keep the best match.
    # Candidates are refined on a ~512 px copy (fast); only the winner is refined at full size.
    size = (fixed.shape[1], fixed.shape[0])
    fixed_s, moving_s = ndi.gaussian_filter(fixed, 1.5), ndi.gaussian_filter(moving, 1.5)
    d = min(1.0, 512 / max(fixed.shape))
    D, Di = _scale(d), _scale(1 / d)

    def shrink(img):
        return cv2.resize(img, None, fx=d, fy=d, interpolation=cv2.INTER_AREA)

    f_sm, m_sm, mask_sm = shrink(fixed_s), shrink(moving_s), shrink(mf.astype(np.float32)) > 0.5
    size_sm = (f_sm.shape[1], f_sm.shape[0])
    scored.sort(key=lambda s: -s[0])
    top = scored[:8]
    best = (-2.0, None, None)
    for n, (_, M0) in enumerate(top):
        M0s = D @ M0 @ Di  # same transform in small-image pixels
        start = ncc(f_sm, warp_into(m_sm, M0s, size_sm), mask_sm)
        # rigid first (only when both pixel sizes are known: it cannot change scale), then affine
        M1 = _ecc(f_sm, m_sm, M0s, cv2.MOTION_EUCLIDEAN, 100) if k == 1.0 else None
        M1 = _ecc(f_sm, m_sm, M1 if M1 is not None else M0s)
        after = ncc(f_sm, warp_into(m_sm, M1, size_sm), mask_sm) if M1 is not None else -2.0
        if after < start:
            M1, after = M0s, start
        if after > best[0]:
            best = (after, M1, M0)
        if progress:
            progress(0.5 + 0.4 * (n + 1) / len(top), "Refining the best candidates")
    M0 = best[2]
    M1 = Di @ best[1] @ D
    start = ncc(fixed, warp_into(moving, M0, size), mf)
    after = ncc(fixed, warp_into(moving, M1, size), mf)
    if progress:
        progress(0.95, "Final refinement")
    M2 = _ecc(fixed_s, moving_s, M1, iters=50)
    if M2 is not None:
        a2 = ncc(fixed, warp_into(moving, M2, size), mf)
        if a2 > after:
            M1, after = M2, a2
    if after < start:
        M1, after = M0, start
    # overview px -> level-0 px on both sides
    T = _scale(1 / s_fl) @ M1 @ _scale(s_he)
    if progress:
        progress(1.0, "Done")
    return Registration(T, start, after, he_img, s_he)


@dataclass
class RegionPair:
    """A region of the H&E section with the fluorescence channels resampled onto it."""

    box: Box  # H&E level-0 pixels
    he: np.ndarray  # uint8 RGB
    fl: np.ndarray  # float32 HxWxC, channels in the order requested
    channels: list[int]
    mpp: float | None  # microns per pixel of these arrays
    score: float  # NCC of DAPI vs hematoxylin in the region
    refined: bool


def _local_fit(fixed: np.ndarray, moving: np.ndarray, pad: int, mpp: float | None,
               mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Transform (region px -> padded px) aligning `moving` (padded by `pad`) to `fixed`.

    Starts from the coarse registration (a pure offset of `pad`), then template matching on
    a ~8 µm smoothed copy, then ECC at ~3 µm and at full resolution. Returns the best
    transform and its NCC.
    """
    h, w = fixed.shape
    um = 1.0 / (mpp or 1.0)  # px per µm
    sm = ndi.gaussian_filter

    def score(R):
        return ncc(fixed, warp_into(moving, R, (w, h)), mask)

    best_R = _translate(pad, pad)
    best = score(best_R)

    # 1. offset by template matching on a smoothed, downsampled copy (~4 µm/px)
    d = min(1.0, 0.25 / um, 600 / max(h + 2 * pad, w + 2 * pad))
    f_s = cv2.resize(sm(fixed, 8 * um), None, fx=d, fy=d, interpolation=cv2.INTER_AREA)
    m_s = cv2.resize(sm(moving, 8 * um), None, fx=d, fy=d, interpolation=cv2.INTER_AREA)
    if pad > 0 and m_s.shape[0] > f_s.shape[0] and m_s.shape[1] > f_s.shape[1]:
        res = cv2.matchTemplate(m_s.astype(np.float32), f_s.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, _, _, (x, y) = cv2.minMaxLoc(res)
        R = _translate(x / d, y / d)
        s = score(R)
        if s > best:
            best_R, best = R, s

    # 2. ECC from coarse to fine; small images keep it fast
    for sigma_um, size in ((3.0, 800), (1.0, 1600)):
        d = min(1.0, size / max(h, w))
        D, Di = _scale(d), _scale(1 / d)
        f_s = cv2.resize(sm(fixed, sigma_um * um), None, fx=d, fy=d, interpolation=cv2.INTER_AREA)
        m_s = cv2.resize(sm(moving, sigma_um * um), None, fx=d, fy=d, interpolation=cv2.INTER_AREA)
        R = _ecc(f_s, m_s, D @ best_R @ Di, iters=80)
        if R is not None:
            R = Di @ R @ D
            s = score(R)
            if s > best:
                best_R, best = R, s
    return best_R, best


def read_pair(he: Section, fl: Section, T: np.ndarray, box: Box, out_width: int,
              channels: list[int], dapi: int, refine: bool = True, signal: str = "nuclei") -> RegionPair:
    """Read an H&E region and the matching immunostain channels, aligned pixel to pixel.

    `dapi` is the immunostain channel matched against the H&E (see register_sections).
    """
    he_img = he.read(box, out_width)
    h, w = he_img.shape[:2]
    s = w / (box[2] - box[0])
    mpp = he.mpp / s if he.mpp else None
    # margin read around the region so the local fit can find offsets left by the coarse fit
    pad = int(min(0.25 * max(w, h), 300 / (mpp or 1.0))) if refine else 0
    # padded region px -> H&E level-0 -> immunostain level-0
    P = T @ _translate(box[0], box[1]) @ _scale(1 / s) @ _translate(-pad, -pad)
    W, H = w + 2 * pad, h + 2 * pad
    corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], float).T
    q = P @ corners
    fbox = (q[0].min(), q[1].min(), q[0].max(), q[1].max())
    zoom = np.sqrt(abs(np.linalg.det(P[:2, :2])))  # immunostain px per region px
    fw = max(8, int(round((fbox[2] - fbox[0]) / zoom)))
    want = sorted(set(channels) | {dapi})
    fl_img = fl.read(fbox, fw, want)
    sf = fw / (fbox[2] - fbox[0])
    M = _scale(sf) @ _translate(-fbox[0], -fbox[1]) @ P  # padded px -> fl-region px
    padded = warp_into(fl_img, M, (W, H))
    fixed = he_signal(he_img, signal)
    mask = tissue_mask(fixed)
    R0 = _translate(pad, pad)  # region px -> padded px, coarse registration only
    R, refined = R0, False
    if refine:
        R, _ = _local_fit(fixed, padded[..., want.index(dapi)], pad, mpp, mask)
        refined = not np.allclose(R, R0)
    warped = warp_into(padded, R, (w, h))
    score = ncc(fixed, warped[..., want.index(dapi)], mask)
    out = np.stack([warped[..., want.index(c)] for c in channels], -1)
    return RegionPair(box, he_img, out, channels, mpp, score, refined)


__all__ = ["COARSE_MPP", "Registration", "RegionPair", "density", "he_signal", "hematoxylin", "ncc",
           "read_pair", "register_sections", "tissue_mask", "warp_into"]
