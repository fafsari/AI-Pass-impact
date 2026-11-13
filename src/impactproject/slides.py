"""Whole-slide readers with one interface for archive items and local files.

A section is read at whatever resolution the task needs: a small overview for
registration, a region at near full resolution for viewing and quantification.
The whole slide is never loaded into memory.

Fluorescence values are mapped to 0..1 with fixed per-channel ranges taken from
the slide overview, so every region and tile of a section is on the same scale
and one threshold means the same thing everywhere.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Box = tuple[float, float, float, float]  # left, top, right, bottom in level-0 pixels


@dataclass
class Section:
    """One tissue section: an H&E image (RGB) or a multiplex fluorescence image."""

    name: str
    width: int
    height: int
    mpp: float | None  # microns per level-0 pixel, if the file records it
    channels: list[str]  # ["R", "G", "B"] for brightfield
    rgb: bool
    ranges: dict[int, tuple[float, float]] = field(default_factory=dict)  # fluorescence normalisation

    # subclasses implement _read_raw(box, out_w, out_h, channels) -> HxWxC array
    def _read_raw(self, box: Box, out_w: int, out_h: int, channels: list[int]) -> np.ndarray:
        raise NotImplementedError

    def read(self, box: Box | None = None, out_width: int = 1024,
             channels: list[int] | None = None) -> np.ndarray:
        """Region `box` (default: whole slide) resampled to `out_width` pixels wide.

        Brightfield returns uint8 RGB. Fluorescence returns float32 in 0..1, one
        plane per requested channel, using the section's fixed ranges.
        """
        box = box or (0, 0, self.width, self.height)
        l, t, r, b = box
        out_w = max(1, int(out_width))
        out_h = max(1, int(round(out_w * (b - t) / max(1e-6, r - l))))
        if self.rgb:
            return self._read_raw(box, out_w, out_h, [0, 1, 2]).astype(np.uint8)
        channels = list(range(len(self.channels))) if channels is None else channels
        raw = self._read_raw(box, out_w, out_h, channels).astype(np.float32)
        if not self.ranges:
            self.calibrate()
        out = np.empty_like(raw)
        for k, c in enumerate(channels):
            lo, hi = self.ranges[c]
            out[..., k] = np.clip((raw[..., k] - lo) / max(hi - lo, 1e-6), 0, 1)
        return out

    def calibrate(self, width: int = 1536):
        """Fix each fluorescence channel's display/measurement range from the overview."""
        if self.rgb:
            return
        chans = list(range(len(self.channels)))
        ov = self._read_raw((0, 0, self.width, self.height), width,
                            max(1, round(width * self.height / self.width)), chans).astype(np.float32)
        for c in chans:
            plane = ov[..., c]
            fg = plane[plane > np.percentile(plane, 50)]  # skip empty background
            lo = float(np.percentile(plane, 1))
            hi = float(np.percentile(fg if fg.size else plane, 99.8))
            self.ranges[c] = (lo, hi if hi > lo else lo + 1)

    def scale_for(self, out_width: int, box: Box | None = None) -> float:
        """Output pixels per level-0 pixel when reading `box` at `out_width`."""
        box = box or (0, 0, self.width, self.height)
        return out_width / (box[2] - box[0])


# ---------------------------------------------------------------- local files
def _ome_channels_and_mpp(tif) -> tuple[list[str], float | None]:
    names, mpp = [], None
    if tif.ome_metadata:
        root = ET.fromstring(tif.ome_metadata)
        ns = {"o": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        pix = root.find(".//o:Pixels", ns) if ns else root.find(".//Pixels")
        if pix is not None:
            if pix.get("PhysicalSizeX"):
                mpp = float(pix.get("PhysicalSizeX"))
                if pix.get("PhysicalSizeXUnit") in ("nm",):
                    mpp /= 1000
            chans = pix.findall("o:Channel", ns) if ns else pix.findall("Channel")
            names = [c.get("Name") or f"Channel {i + 1}" for i, c in enumerate(chans)]
    return names, mpp


def _svs_mpp(tif) -> float | None:
    desc = tif.pages[0].description or ""
    m = re.search(r"MPP\s*=\s*([\d.]+)", desc)
    return float(m.group(1)) if m else None


class LocalSection(Section):
    """OME-TIFF, SVS, NDPI or other (pyramidal) TIFF on local disk, read lazily with zarr."""

    def __init__(self, path: str | Path):
        import tifffile
        import zarr

        self.path = Path(path)
        tif = tifffile.TiffFile(self.path)
        series = tif.series[0]
        axes = series.axes.upper()
        names, mpp = _ome_channels_and_mpp(tif)
        mpp = mpp or _svs_mpp(tif)
        store = series.aszarr()
        z = zarr.open(store, mode="r")
        self._levels = [z[k] for k in sorted(z.keys(), key=int)] if hasattr(z, "keys") else [z]
        self._axes = axes
        shape0 = dict(zip(axes, self._levels[0].shape))
        rgb = "S" in axes and shape0.get("S") in (3, 4)
        n_ch = shape0.get("C", 1)
        if rgb:
            channels = ["R", "G", "B"]
        else:
            channels = names if len(names) == n_ch else [f"Channel {i + 1}" for i in range(n_ch)]
        super().__init__(self.path.name, shape0["X"], shape0["Y"], mpp, channels, rgb)
        self._tif = tif  # keep the file handle open for the zarr store

    def _read_raw(self, box, out_w, out_h, channels):
        l, t, r, b = box
        need = (r - l) / out_w  # level-0 pixels per output pixel
        level, down = self._levels[0], 1.0
        for lv in self._levels:  # coarsest level that is still at least as fine as needed
            d = self.width / dict(zip(self._axes, lv.shape))["X"]
            if d <= need * 1.01 and d >= down:
                level, down = lv, d
        sl = []
        for ax, n in zip(self._axes, level.shape):
            if ax == "X":
                sl.append(slice(max(0, int(l / down)), min(n, int(np.ceil(r / down)))))
            elif ax == "Y":
                sl.append(slice(max(0, int(t / down)), min(n, int(np.ceil(b / down)))))
            elif ax == "C" and not self.rgb:
                sl.append(channels)
            elif ax == "S":
                sl.append(slice(0, 3))
            else:
                sl.append(0)  # T, Z, etc.: first plane
        arr = np.asarray(level[tuple(sl)])
        kept = [ax for ax, s in zip(self._axes, sl) if not isinstance(s, int)]
        order = [kept.index("Y"), kept.index("X")] + ([kept.index("S")] if self.rgb else
                                                       [kept.index("C")] if "C" in kept else [])
        arr = arr.transpose(order) if len(order) == arr.ndim else arr
        if arr.ndim == 2:
            arr = arr[..., None]
        # pad if the box ran past the slide edge, then resample to the output size
        want_h = int(round((b - t) / down)); want_w = int(round((r - l) / down))
        pad_t, pad_l = max(0, -int(t / down)), max(0, -int(l / down))
        canvas = np.zeros((max(want_h, 1), max(want_w, 1), arr.shape[2]), arr.dtype)
        h, w = min(arr.shape[0], canvas.shape[0] - pad_t), min(arr.shape[1], canvas.shape[1] - pad_l)
        canvas[pad_t:pad_t + h, pad_l:pad_l + w] = arr[:h, :w]
        if self.rgb and canvas.max() == 0:
            canvas[:] = 255
        out = np.stack([cv2.resize(canvas[..., k].astype(np.float32), (out_w, out_h),
                                   interpolation=cv2.INTER_AREA) for k in range(canvas.shape[2])], -1)
        return out


# ---------------------------------------------------------------- archive items
class ArchiveSection(Section):
    """A large_image item on a Digital Slide Archive server (Athena, FUSION)."""

    def __init__(self, archive, item: dict):
        self.archive, self.item_id = archive, item["_id"]
        info = archive.gc.get(f"item/{self.item_id}/tiles")
        frames = info.get("frames") or []
        names = info.get("channels") or []
        rgb = not frames and info.get("bandCount", 3) >= 3
        mpp = info["mm_x"] * 1000 if info.get("mm_x") else None
        channels = ["R", "G", "B"] if rgb else (names or [f"Channel {i + 1}" for i in range(max(1, len(frames)))])
        super().__init__(item["name"], info["sizeX"], info["sizeY"], mpp, channels, rgb)

    def _get(self, params):
        resp = self.archive.gc.get(f"item/{self.item_id}/tiles/region", params, jsonResp=False)
        return Image.open(io.BytesIO(resp.content))

    def _read_raw(self, box, out_w, out_h, channels):
        l, t, r, b = box
        base = {"left": int(l), "top": int(t), "right": int(r), "bottom": int(b),
                "width": out_w, "height": out_h, "encoding": "PNG"}
        if self.rgb:
            img = np.asarray(self._get(base).convert("RGB"))
            return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        planes = []
        for c in channels:
            # 'min'/'max' are computed by the server over the whole frame, so every
            # region of this channel is scaled identically
            img = self._get({**base, "frame": c, "style": '{"min": "min", "max": "max"}'})
            p = np.asarray(img.convert("L"), dtype=np.float32)
            planes.append(cv2.resize(p, (out_w, out_h), interpolation=cv2.INTER_AREA))
        return np.stack(planes, -1)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# Stain colours as optical-density vectors (R, G, B). The blue channel uses the standard
# Ruifrok & Johnston hematoxylin vector and takes up any blue/purple chromogen; DAB is the
# standard vector; the red chromogen was measured on an nPOD INS/Ki-67 double stain
# (insulin in red, Ki-67 in DAB), close to skimage's Fast Red.
BLUE, DAB, RED = "Blue (hematoxylin or GCG)", "DAB (brown)", "Red chromogen"
STAIN_OD = {BLUE: _unit([0.650, 0.704, 0.286]),
            DAB: _unit([0.268, 0.570, 0.776]),
            RED: _unit([0.174, 0.809, 0.561])}
# Fixed display/measurement ranges in stain amount (skimage separate_stains units), so
# the same threshold means the same stain intensity on every slide. On the nPOD slides:
# counterstained nuclei ~0.08, Ki-67 nuclei DAB ~0.15 (background ~0.015), insulin red ~0.11.
# Tissue = total absorbance over R, G and B: glass ~0.011, tissue ~0.06.
STAIN_RANGE = {BLUE: (0.0, 0.15), DAB: (0.0, 0.25), RED: (0.0, 0.20)}
TISSUE_RANGE = (0.0, 0.10)

# How nPOD stained the IHC slides (markers are listed in the file name):
# - INS + Ki-67 (older slides): insulin red, Ki-67 brown (DAB), nuclei counterstained blue
# - INS + GCG + Ki-67 (newer slides, e.g. 6593): insulin red, glucagon blue, Ki-67 brown,
#   NO nuclear counterstain (faint purple in nuclei is background of the glucagon stain)
MARKER_KEYS = {"ins": ("ins", "insulin"), "gcg": ("gcg", "glucagon", "glu"), "ki67": ("ki67", "ki-67", "mki67")}


def markers_in_name(name: str) -> set[str]:
    """Markers listed in a slide file name, e.g. 'ND6445_PB_INS_GCG_Ki67_131292.svs' -> ins, gcg, ki67."""
    tokens = set(re.split(r"[^a-z0-9-]+", name.lower()))
    return {m for m, keys in MARKER_KEYS.items() if tokens & set(keys)}


def brightfield_roles(name: str) -> dict[str, int]:
    """Channel of each role (dapi = nuclei, ins, gcg, ki67) for an nPOD IHC slide, from its name.

    -1 means not on this slide. With glucagon on the slide, blue is glucagon and there is no
    nuclear stain; without it, blue is the hematoxylin counterstain.
    """
    found = markers_in_name(name)
    if not found:  # name lists no markers: assume the common INS + Ki-67 double stain
        found = {"ins", "ki67"}
    gcg = "gcg" in found
    return {"dapi": -1 if gcg else 0,
            "ins": 2 if "ins" in found else -1,
            "gcg": 0 if gcg else -1,
            "ki67": 1 if "ki67" in found else -1}


class StainSeparated(Section):
    """A brightfield immunostain presented as channels, like fluorescence.

    Colour deconvolution (Ruifrok & Johnston) splits the RGB image into a blue stain
    (hematoxylin counterstain, or the glucagon chromogen on INS/GCG/Ki-67 slides), DAB
    (brown, Ki-67) and a red chromogen (insulin). The standard H-E-DAB set is not used: it
    puts red chromogen into DAB, so insulin would be counted as Ki-67. A fourth channel,
    Tissue, is the total absorbance of all stains: it outlines the tissue and is used for
    registration whether or not the nuclei are stained. Stain channels use fixed ranges
    (STAIN_RANGE). The original colours are kept as Red/Green/Blue channels with a fixed
    0..255 range so the real slide can still be displayed.
    """

    TISSUE = 3
    RGB_CHANNELS = (4, 5, 6)
    STAINS = tuple(STAIN_OD)

    def __init__(self, base: Section):
        self.base = base
        self._inv = np.linalg.inv(np.array([STAIN_OD[s] for s in self.STAINS]))
        super().__init__(base.name, base.width, base.height, base.mpp,
                         list(self.STAINS) + ["Tissue (all stains)", "Red", "Green", "Blue"], False)

    def _read_raw(self, box, out_w, out_h, channels):
        rgb = self.base._read_raw(box, out_w, out_h, [0, 1, 2])
        # same OD scaling as skimage.color.separate_stains (log base 1e-6), so STAIN_RANGE
        # matches values measured with skimage
        od = np.log(np.clip(rgb / 255.0, 1e-6, 1)) / np.log(1e-6)
        stains = np.maximum(od @ self._inv, 0)
        planes = [stains[..., 0], stains[..., 1], stains[..., 2], od.sum(-1),
                  rgb[..., 0], rgb[..., 1], rgb[..., 2]]
        return np.stack([planes[c] for c in channels], -1).astype(np.float32)

    def calibrate(self, width: int = 1536):
        self.ranges = {i: STAIN_RANGE[s] for i, s in enumerate(self.STAINS)}
        self.ranges[self.TISSUE] = TISSUE_RANGE
        for c in self.RGB_CHANNELS:
            self.ranges[c] = (0.0, 255.0)


def guess_channel(channels: list[str], *keys: str) -> int | None:
    """Index of the first channel whose name contains any of `keys` (case-insensitive)."""
    for i, n in enumerate(channels):
        if any(k.lower() in n.lower() for k in keys):
            return i
    return None


__all__ = ["ArchiveSection", "Box", "LocalSection", "Section", "StainSeparated", "brightfield_roles",
           "guess_channel", "markers_in_name"]
