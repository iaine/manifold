"""
manifold_engine.py
==================
The vector-space engine for MANIFOLD.

Turns audio clips into points in a high-dimensional space, then provides the
projections, distances, and per-axis data the frontend needs to sonify the
four core MANIFOLD gestures:

  - chord-of-many   : the full dimensionality of a point (every axis = a partial)
  - dwell           : holding a single point, hearing its foreground vs. bed
  - translation-seam: the discrepancy between an audio clip's "own" spectral
                      tuning and the space's "learned" axes (epistemic compression)
  - distance        : similarity between points, rendered later as consonance

This module is deliberately ML-light: the "embedding" is a real,
high-dimensional audio feature vector (mel-spectrogram statistics + MFCCs +
spectral descriptors), standardized and given a learned-axis basis via PCA.
That gives us a genuine high-dim space with genuine learned directions,
without requiring a heavy neural codec download.
"""

from __future__ import annotations
import os
import glob
import numpy as np
import librosa
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ----------------------------------------------------------------------
# Feature extraction: a clip -> a high-dimensional vector
# ----------------------------------------------------------------------

EMBED_SR = 22050            # analysis sample rate
N_MELS = 64                 # mel bands -> the bulk of the "bed"
N_MFCC = 20                 # cepstral coefficients -> compact timbral identity


def extract_embedding(path: str, sr: int = EMBED_SR) -> tuple[np.ndarray, dict]:
    """
    Load one audio file and return (vector, meta).

    The vector concatenates several feature families so the resulting space
    has real, interpretable high dimensionality. We summarize each
    time-varying feature by mean and standard deviation across frames, so a
    whole clip collapses to a single point -- the MANIFOLD "becoming a point".
    """
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"empty audio: {path}")

    # Mel spectrogram (power) -> dB
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Spectral descriptors
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    def stats(M):
        return np.concatenate([M.mean(axis=1), M.std(axis=1)])

    vector = np.concatenate([
        stats(mel_db),     # 128 dims
        stats(mfcc),       # 40 dims
        stats(cent),       # 2
        stats(bw),         # 2
        stats(rolloff),    # 2
        stats(flatness),   # 2
        stats(zcr),        # 2
        stats(rms),        # 2
    ]).astype(np.float64)

    # The "own tuning" of the clip: a small spectral signature derived from
    # the clip itself (mel peak frequencies), used later for the seam gesture.
    mel_profile = mel_db.mean(axis=1)
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=sr / 2)

    meta = {
        "path": path,
        "name": os.path.splitext(os.path.basename(path))[0],
        "duration": float(len(y) / sr),
        "mel_profile": mel_profile.tolist(),
        "mel_freqs": mel_freqs.tolist(),
        "raw_dim": int(vector.shape[0]),
    }
    return vector, meta


# ----------------------------------------------------------------------
# Alternative encoder: a learned neural latent (EnCodec)
# ----------------------------------------------------------------------
#
# This returns the SAME (vector, meta) contract as extract_embedding above,
# so it is a true drop-in. The difference is conceptual and, for MANIFOLD,
# the whole point: extract_embedding builds a space out of hand-designed
# audio features, whereas this builds it out of a *learned* latent -- the
# terms a neural codec decided were worth keeping. Navigating this space is
# navigating the model's own description of the audio, not ours.
#
# EnCodec encodes a waveform into discrete codes across several residual
# vector-quantizer codebooks. To collapse a whole clip into a single point
# (MANIFOLD's "becoming a point") we summarize the continuous latent
# embedding over time by mean and standard deviation, exactly as the feature
# encoder summarizes its frames.

# Lazily-initialized so importing this module never forces a torch import.
_ENCODEC = {"model": None, "sr": None}


def _load_encodec(bandwidth: float = 6.0):
    """Load and cache the 24 kHz EnCodec model. Imported lazily on first use."""
    if _ENCODEC["model"] is not None:
        return _ENCODEC["model"], _ENCODEC["sr"]
    try:
        import torch  # noqa: F401
        from encodec import EncodecModel
    except ImportError as e:
        raise ImportError(
            "encoder_embedding requires torch, torchaudio, and encodec.\n"
            "Install with:  pip install torch torchaudio encodec"
        ) from e
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model.eval()
    _ENCODEC["model"] = model
    _ENCODEC["sr"] = model.sample_rate  # 24000
    return model, _ENCODEC["sr"]


def encoder_embedding(path: str, bandwidth: float = 6.0
                      ) -> tuple[np.ndarray, dict]:
    """
    Encode one audio file into a point in EnCodec's learned latent space.

    Drop-in replacement for extract_embedding: returns (vector, meta) with the
    same keys, so load_directory(..., encoder=encoder_embedding) just works.

    The vector is the time-summarized continuous latent (the pre-quantization
    embedding), giving a real high-dimensional learned representation. The
    mel_profile / mel_freqs for the translation-seam are still derived with
    librosa from the clip itself, because the seam is precisely the tension
    between the clip's OWN spectral tuning and wherever the learned space
    places it -- so we deliberately keep an independent, audio-side reading.
    """
    import torch

    model, target_sr = _load_encodec(bandwidth)

    # Load + conform to the model's expectations: mono, model sample rate.
    # We load with librosa (soundfile under the hood) rather than
    # torchaudio.load, because recent torchaudio routes loading through
    # torchcodec, which is an extra dependency. librosa is already required.
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError(f"empty audio: {path}")
    wav = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)  # [1, 1, time]

    with torch.no_grad():
        # The continuous latent before quantization: shape [1, D, T_frames].
        # This is the richest learned representation EnCodec exposes pre-codes.
        latent = model.encoder(wav)            # [1, D, T]
        lat = latent.squeeze(0).cpu().numpy()  # [D, T]

    # Collapse time -> a single point: mean and std per latent channel.
    # D is typically 128, so the resulting vector is ~256-dim.
    vector = np.concatenate([lat.mean(axis=1), lat.std(axis=1)]).astype(np.float64)

    # Independent audio-side spectral reading for the seam (see docstring).
    y22, _ = librosa.load(path, sr=EMBED_SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y22, sr=EMBED_SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_profile = mel_db.mean(axis=1)
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=EMBED_SR / 2)

    meta = {
        "path": path,
        "name": os.path.splitext(os.path.basename(path))[0],
        "duration": float(len(y) / target_sr),
        "mel_profile": mel_profile.tolist(),
        "mel_freqs": mel_freqs.tolist(),
        "raw_dim": int(vector.shape[0]),
        "encoder": "encodec_24khz",
        "bandwidth": float(bandwidth),
        "latent_frames": int(lat.shape[1]),
    }
    return vector, meta


# ----------------------------------------------------------------------
# The space: a collection of points + learned axes
# ----------------------------------------------------------------------

@dataclass
class Manifold:
    vectors: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    meta: list = field(default_factory=list)
    scaler: StandardScaler | None = None
    pca: PCA | None = None
    coords2d: np.ndarray | None = None     # projection for the map
    standardized: np.ndarray | None = None # zero-mean unit-var vectors

    @property
    def n_points(self) -> int:
        return self.vectors.shape[0]

    @property
    def n_dims(self) -> int:
        return self.vectors.shape[1] if self.vectors.ndim == 2 else 0

    def build(self, vectors: list[np.ndarray], meta: list[dict]):
        self.vectors = np.vstack(vectors)
        self.meta = meta
        # Standardize: every learned axis gets equal voice (no dim dominates
        # the chord just because it has bigger units).
        self.scaler = StandardScaler()
        self.standardized = self.scaler.fit_transform(self.vectors)
        # Learned axes: PCA gives an orthogonal basis of "directions the data
        # cares about" -- the space's own terms, not the audio's.
        n_comp = min(self.standardized.shape[0], self.standardized.shape[1])
        self.pca = PCA(n_components=n_comp)
        projected = self.pca.fit_transform(self.standardized)
        # 2D map coordinates = first two principal components.
        self.coords2d = projected[:, :2] if projected.shape[1] >= 2 else \
            np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
        return self

    # ---- data the frontend needs, per gesture ----

    def point_payload(self, i: int) -> dict:
        """Everything needed to sonify dwelling on point i."""
        std_vec = self.standardized[i]
        explained = self.pca.explained_variance_ratio_
        proj = self.pca.transform(std_vec.reshape(1, -1))[0]
        # The chord-of-many uses the FULL standardized vector (n_dims, e.g. 180)
        # -- every original learned dimension becomes one partial, far more than
        # the ear can resolve. The PCA projection (capped at n_points) is only
        # for the 2D map and the foreground ranking.
        # foreground = the few dims with largest absolute value for THIS point
        # (its most defining coordinates) -> countable partials over the bed.
        fg_order = np.argsort(np.abs(std_vec))[::-1]
        foreground_idx = fg_order[:6].tolist()
        return {
            "index": i,
            "name": self.meta[i]["name"],
            "n_dims": int(self.n_dims),
            "coord2d": self.coords2d[i].tolist(),
            # FULL high-dim coordinates -> every axis becomes a partial
            "vector": std_vec.tolist(),
            # low-dim projection retained for the map / shimmer
            "projection": proj.tolist(),
            "explained_variance": explained.tolist(),
            "foreground_axes": [int(x) for x in foreground_idx],
            # the clip's "own tuning" for the seam gesture
            "mel_profile": self.meta[i]["mel_profile"],
            "mel_freqs": self.meta[i]["mel_freqs"],
        }

    def distance(self, i: int, j: int) -> float:
        """Euclidean distance in standardized space -> consonance later."""
        return float(np.linalg.norm(self.standardized[i] - self.standardized[j]))

    def distances_from(self, i: int) -> list[float]:
        d = np.linalg.norm(self.standardized - self.standardized[i], axis=1)
        return d.tolist()

    def nearest(self, i: int, k: int = 5) -> list[dict]:
        d = np.array(self.distances_from(i))
        order = np.argsort(d)
        out = []
        for j in order:
            if j == i:
                continue
            out.append({"index": int(j), "name": self.meta[j]["name"],
                        "distance": float(d[j])})
            if len(out) >= k:
                break
        return out

    def projection_at(self, point2d: tuple[float, float]) -> dict:
        """
        For free navigation: given an arbitrary (x, y) on the 2D map, return
        a synthetic point's payload by inverse-projecting back to high-dim
        space. This is how the listener 'moves through' the manifold between
        the real clips -- projection-shimmer territory.
        """
        xy = np.array([point2d[0], point2d[1]])
        # Build a full-D projected vector: known first 2 comps, zeros elsewhere.
        full = np.zeros(self.pca.n_components_)
        full[0], full[1] = xy[0], xy[1]
        std_vec = self.pca.inverse_transform(full.reshape(1, -1))[0]
        proj = full
        # nearest real neighbours to this empty-space location
        d = np.linalg.norm(self.coords2d - xy, axis=1)
        order = np.argsort(d)[:3]
        neighbours = [{"index": int(j), "name": self.meta[j]["name"],
                       "distance": float(d[j])} for j in order]
        return {
            "index": -1,
            "name": "(navigated point)",
            "n_dims": int(self.n_dims),
            "coord2d": xy.tolist(),
            "vector": std_vec.tolist(),
            "projection": proj.tolist(),
            "explained_variance": self.pca.explained_variance_ratio_.tolist(),
            "foreground_axes": [int(x) for x in np.argsort(np.abs(std_vec))[::-1][:6]],
            "neighbours": neighbours,
        }


# ----------------------------------------------------------------------
# Loading a directory of audio into a Manifold
# ----------------------------------------------------------------------

AUDIO_EXT = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.aiff", "*.aif")


def load_directory(folder: str, encoder=extract_embedding) -> Manifold:
    """
    Build a Manifold from every audio clip in a folder.

    encoder: the per-clip embedding function, returning (vector, meta).
             Defaults to the librosa feature extractor (extract_embedding,
             no heavy deps). Pass encoder_embedding to navigate EnCodec's
             learned latent space instead:

                 load_directory(folder, encoder=encoder_embedding)
    """
    paths = []
    for ext in AUDIO_EXT:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"no audio files in {folder}")
    vectors, meta = [], []
    for p in paths:
        try:
            v, m = encoder(p)
            vectors.append(v)
            meta.append(m)
        except ImportError:
            # Missing encoder dependency (e.g. torch/encodec) is not a bad
            # clip -- propagate so the caller can fall back to another encoder.
            raise
        except Exception as e:
            print(f"  skip {p}: {e}")
    if not vectors:
        raise RuntimeError("no clips could be encoded")
    return Manifold().build(vectors, meta)
