"""
app.py
======
MANIFOLD desktop application.

A PyWebView window hosting the navigable vector-space interface. The Python
side owns the audio analysis and the high-dimensional space; the JS side
owns the map, the navigation, and Web Audio sonification of the four MANIFOLD
gestures.

Run:
    pip install pywebview librosa scikit-learn numpy scipy soundfile
    python app.py [path/to/audio/folder]

If no folder is given it looks for ./clips, and if that's missing it offers a
folder picker.
"""

from __future__ import annotations
import os
import sys
import json
import threading
import webview  # pywebview

from manifold_engine import (
    Manifold, load_directory, extract_embedding, encoder_embedding,
)


class Api:
    """Methods on this class are callable from JS as window.pywebview.api.*"""

    def __init__(self):
        self.manifold: Manifold | None = None
        self._lock = threading.Lock()
        self.encoder = extract_embedding      # default: librosa features
        self.encoder_name = "features"

    # ---- space construction ----

    def use_encoder(self, name: str) -> dict:
        """Select the embedding function: 'features' or 'encodec'."""
        if name == "encodec":
            self.encoder = encoder_embedding
            self.encoder_name = "encodec"
        else:
            self.encoder = extract_embedding
            self.encoder_name = "features"
        return {"ok": True, "encoder": self.encoder_name}

    def pick_folder(self) -> str | None:
        win = webview.windows[0]
        result = win.create_file_dialog(webview.FOLDER_DIALOG)
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else result

    def load_folder(self, folder: str) -> dict:
        """Analyze every clip in a folder and build the manifold."""
        with self._lock:
            try:
                self.manifold = load_directory(folder, encoder=self.encoder)
            except ImportError as e:
                # EnCodec deps missing -> fall back to features, report it
                self.encoder = extract_embedding
                self.encoder_name = "features"
                try:
                    self.manifold = load_directory(folder, encoder=self.encoder)
                except Exception as e2:
                    return {"ok": False, "error": str(e2)}
                return {"ok": True, "encoder": "features",
                        "warning": f"encodec unavailable ({e}); used features",
                        **self.space_summary()}
            except Exception as e:
                return {"ok": False, "error": str(e)}
            return {"ok": True, "encoder": self.encoder_name,
                    **self.space_summary()}

    def space_summary(self) -> dict:
        m = self.manifold
        if not m:
            return {"ok": False, "error": "no space loaded"}
        return {
            "ok": True,
            "n_points": m.n_points,
            "n_dims": m.n_dims,
            "points": [
                {"index": i, "name": m.meta[i]["name"],
                 "coord2d": m.coords2d[i].tolist()}
                for i in range(m.n_points)
            ],
            # bounds so the frontend can scale the map
            "bounds": {
                "xmin": float(m.coords2d[:, 0].min()),
                "xmax": float(m.coords2d[:, 0].max()),
                "ymin": float(m.coords2d[:, 1].min()),
                "ymax": float(m.coords2d[:, 1].max()),
            },
        }

    # ---- per-gesture data ----

    def point(self, index: int) -> dict:
        """Payload for dwelling on a real clip (chord-of-many + seam)."""
        if not self.manifold:
            return {"ok": False, "error": "no space"}
        return {"ok": True, **self.manifold.point_payload(int(index))}

    def navigate(self, x: float, y: float) -> dict:
        """Payload for an arbitrary navigated location (projection-shimmer)."""
        if not self.manifold:
            return {"ok": False, "error": "no space"}
        return {"ok": True, **self.manifold.projection_at((float(x), float(y)))}

    def distances(self, index: int) -> dict:
        """All distances from a point (for distance-as-consonance)."""
        if not self.manifold:
            return {"ok": False, "error": "no space"}
        return {"ok": True,
                "distances": self.manifold.distances_from(int(index)),
                "nearest": self.manifold.nearest(int(index))}


def resolve_folder() -> str | None:
    # first non-flag argument is the folder
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args and os.path.isdir(args[0]):
        return args[0]
    if os.path.isdir("clips"):
        return os.path.abspath("clips")
    return None


def main():
    api = Api()
    # encoder choice: --encodec for the learned latent, default is features
    if "--encodec" in sys.argv:
        api.use_encoder("encodec")
    here = os.path.dirname(os.path.abspath(__file__))
    index = os.path.join(here, "web", "index.html")

    window = webview.create_window(
        "MANIFOLD — navigable vector space",
        index,
        js_api=api,
        width=1180, height=820, min_size=(900, 640),
        background_color="#0a0a12",
    )

    # Auto-load a folder on startup if we can find one.
    folder = resolve_folder()

    def on_start():
        if folder:
            res = api.load_folder(folder)
            window.evaluate_js(
                f"window.__manifoldBoot({json.dumps(res)})"
            )
        else:
            window.evaluate_js("window.__manifoldBoot({'ok': false, 'error': 'no folder'})")

    webview.start(on_start, debug=False)


if __name__ == "__main__":
    main()
