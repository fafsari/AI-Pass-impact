"""Client for Digital Slide Archive (Girder) servers such as Athena.

Everything goes through the server's REST API: slide thumbnails, regions at full
resolution and annotation layers. Whole slides are never downloaded; only small
object crops are cached on local disk.
"""

from __future__ import annotations

import hashlib
import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from girder_client import GirderClient, HttpError
from PIL import Image

ATHENA_API = "https://athena.rc.ufl.edu/api/v1"
FUSION_API = "https://fusionpub.rc.ufl.edu/api/v1"
DEFAULT_API = FUSION_API
CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache"


@dataclass(frozen=True)
class Element:
    """One annotated object (e.g. a glomerulus) on a slide."""

    item_id: str
    slide: str
    layer: str
    element_id: str
    bbox: tuple[int, int, int, int]  # left, top, right, bottom in full-res pixels
    outline: tuple[tuple[float, float], ...]  # polygon for drawing overlays


def _bbox_and_outline(el: dict):
    t = el.get("type")
    if t == "polyline" and el.get("points"):
        pts = [(p[0], p[1]) for p in el["points"]]
    elif t in ("rectangle", "ellipse") and "center" in el:
        cx, cy = el["center"][:2]
        w, h = el.get("width", 0) / 2, el.get("height", 0) / 2
        pts = [(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h), (cx - w, cy + h)]
    elif t == "circle" and "center" in el:
        cx, cy = el["center"][:2]
        r = el.get("radius", 0)
        pts = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    else:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
    if bbox[2] - bbox[0] < 4 or bbox[3] - bbox[1] < 4:
        return None
    return bbox, tuple(pts)


class Archive:
    """A connection to one slide archive, anonymous or authenticated.

    Credentials are passed straight to Girder for a session token and are not
    stored anywhere by this class.
    """

    def __init__(self, api_url: str = DEFAULT_API, username: str | None = None,
                 password: str | None = None, api_key: str | None = None):
        self.api_url = api_url.rstrip("/")
        self.gc = GirderClient(apiUrl=self.api_url)
        if api_key:
            self.gc.authenticate(apiKey=api_key)
        elif username:
            self.gc.authenticate(username=username, password=password)
        self.user = self.gc.get("user/me")
        host = urlparse(self.api_url).hostname or "server"
        self.cache_dir = CACHE_ROOT / host

    @property
    def who(self) -> str:
        return self.user["login"] if self.user else "anonymous (public data only)"

    # --- browsing ---------------------------------------------------------
    def collections(self) -> list[dict]:
        return self.gc.get("collection", {"limit": 0, "sort": "name"})

    def folders(self, parent_type: str, parent_id: str) -> list[dict]:
        return self.gc.get("folder", {"parentType": parent_type, "parentId": parent_id,
                                      "limit": 0, "sort": "name"})

    def folder(self, folder_id: str) -> dict:
        return self.gc.get(f"folder/{folder_id}")

    def items(self, folder_id: str) -> list[dict]:
        return self.gc.get("item", {"folderId": folder_id, "limit": 0, "sort": "name"})

    def slides(self, folder_id: str) -> list[dict]:
        return [i for i in self.items(folder_id) if "largeImage" in i]

    def walk(self, folder_id: str) -> list[dict]:
        """This folder and all folders below it, each with its slides.

        Returns dicts with `path` (folder names below the start, () for the start
        itself), `id`, `slides` (large_image items) and `others` (items the server
        has not turned into slides, e.g. uploads still being processed).
        """
        out, todo = [], [((), folder_id)]
        while todo:
            path, fid = todo.pop(0)
            items = self.items(fid)
            slides = [i for i in items if "largeImage" in i]
            out.append({"path": path, "id": fid, "slides": slides, "others": len(items) - len(slides)})
            todo += [(path + (f["name"],), f["_id"]) for f in self.folders("folder", fid)]
        return sorted(out, key=lambda d: [p.lower() for p in d["path"]])

    def tile_info(self, item_id: str) -> dict:
        return self.gc.get(f"item/{item_id}/tiles")

    def layers(self, item_id: str) -> list[dict]:
        """Annotation layers on a slide (without their elements)."""
        out = []
        for a in self.gc.get("annotation", {"itemId": item_id, "limit": 0}):
            out.append({"id": a["_id"], "name": a["annotation"].get("name", "unnamed"),
                        "count": a.get("_elementCount", 0)})
        return out

    def elements(self, item_id: str, slide: str, layer: dict) -> list[Element]:
        ann = self.gc.get(f"annotation/{layer['id']}")
        out = []
        for i, el in enumerate(ann["annotation"].get("elements", [])):
            geo = _bbox_and_outline(el)
            if geo:
                out.append(Element(item_id, slide, layer["name"], el.get("id", str(i)), *geo))
        return out

    # --- pixels -----------------------------------------------------------
    def _image(self, path: str, params: dict, retries: int = 3) -> Image.Image:
        for attempt in range(retries + 1):
            try:
                resp = self.gc.get(path, params, jsonResp=False)
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except HttpError as e:
                # 502/503/504: the server is busy or restarting; back off and retry
                if e.status not in (502, 503, 504) or attempt == retries:
                    raise
                time.sleep(2 ** (attempt + 1))

    def thumbnail(self, item_id: str, width: int = 1024) -> Image.Image:
        """Slide overview, cached on disk.

        Athena sometimes refuses new thumbnails for anonymous users (401) while still
        serving regions, so fall back to a downsampled region covering the whole slide.
        """
        path = self.cache_dir / item_id / f"thumb_{width}.jpg"
        if path.exists():
            return Image.open(path).convert("RGB")
        try:
            img = self._image(f"item/{item_id}/tiles/thumbnail",
                              {"width": width, "encoding": "JPEG", "jpegQuality": 90})
        except HttpError as e:
            if e.status not in (401, 403):
                raise
            img = self._image(f"item/{item_id}/tiles/region",
                              {"width": width, "encoding": "JPEG", "jpegQuality": 90})
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, quality=90)
        return img

    def crop(self, el: Element, size: int = 224, pad: float = 0.15) -> Image.Image:
        """Square crop centred on the object, with some context, resized to `size`."""
        key = hashlib.sha1(f"{el.item_id}/{el.layer}/{el.element_id}/{size}/{pad}".encode()).hexdigest()
        path = self.cache_dir / el.item_id / f"{key}.jpg"
        if path.exists():
            return Image.open(path).convert("RGB")
        l, t, r, b = el.bbox
        cx, cy = (l + r) / 2, (t + b) / 2
        half = max(r - l, b - t) * (0.5 + pad)
        img = self._image(f"item/{el.item_id}/tiles/region", {
            "left": max(0, int(cx - half)), "top": max(0, int(cy - half)),
            "right": int(cx + half), "bottom": int(cy + half),
            "width": size, "height": size, "encoding": "JPEG", "jpegQuality": 95,
        })
        img = img.resize((size, size))
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, quality=95)
        # return the saved copy so first and later (cached) runs see identical pixels
        return Image.open(path).convert("RGB")

    def crops(self, els: list[Element], size: int = 224, workers: int = 4, progress=None):
        """Fetch many crops in parallel; `progress(done, total)` is called as they arrive."""
        out = [None] * len(els)
        with ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(self.crop, e, size): i for i, e in enumerate(els)}
            for n, f in enumerate(futures, 1):
                out[futures[f]] = f.result()
                if progress:
                    progress(n, len(els))
        return out


__all__ = ["ATHENA_API", "FUSION_API", "DEFAULT_API", "Archive", "Element", "HttpError"]
