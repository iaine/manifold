# MANIFOLD — a navigable vector space you can hear

A desktop application that turns a folder of audio clips into points in a
high-dimensional space, then lets you **navigate that space and hear it**. It
is an implementation of the MANIFOLD sonic language: audio ceasing to be sound
and becoming *position*, in a space too large to perceive, rephrased in terms
it did not choose.

The vector space is built from **real audio embeddings** — every clip is
analysed into a 180-dimensional feature vector (mel-spectrogram statistics,
MFCCs, spectral descriptors), standardized, and given a learned-axis basis via
PCA. Python owns the space; the window's frontend owns the map and the
Web Audio sonification.

## Install

```bash
pip install pywebview librosa scikit-learn numpy scipy soundfile
```

On Linux, PyWebView also needs a GTK or Qt backend:

```bash
# either
pip install pywebview[qt]
# or use system GTK web views (Debian/Ubuntu)
sudo apt install python3-gi gir1.2-webkit2-4.1
```

## Run

```bash
python app.py                 # looks for ./clips, feature encoder
python app.py /path/to/audio  # or point it at any folder of clips
python app.py --encodec       # navigate EnCodec's learned latent instead
```

If no folder is found, the window offers a folder picker. Supported formats:
wav, mp3, flac, ogg, m4a, aiff. Three or more clips give the space useful
structure; eight to a few dozen works well.

**Click "enable audio" first** (browsers and embedded web views require a user
gesture before sound can start), then click a point to dwell, or move the
cursor to navigate the empty space between points.

## Two encoders: feature space vs. learned latent

The space can be built two ways, and the difference is the whole MANIFOLD
point — whose *terms* is the audio being described in?

- **`features`** (default) — a 180-dim vector of librosa descriptors
  (mel-spectrogram stats, MFCCs, spectral shape). Hand-designed axes, no heavy
  dependencies. You navigate *our* description of the audio.
- **`encodec`** — the continuous latent of Meta's EnCodec 24 kHz codec,
  time-summarized to a ~256-dim point. Learned axes the model chose. You
  navigate *the model's* description of the audio.

Enable the learned latent with `python app.py --encodec`, which requires:

```bash
pip install torch torchaudio encodec
```

The first `--encodec` run downloads the EnCodec weights (a one-time network
fetch). If torch/encodec aren't installed, the app falls back to the feature
encoder automatically and reports it, so it always runs.

Both encoders return the identical `(vector, meta)` contract, so everything
downstream — projection, distances, navigation, and all four sonification
gestures — is unchanged. The translation-seam deliberately keeps an
*independent* librosa spectral reading of each clip even in EnCodec mode,
because the seam is precisely the tension between the clip's own spectral
tuning and wherever the learned space places it.

## The four gestures, and where they live in the code

| Gesture | What you do | What you hear | Code |
|---------|-------------|---------------|------|
| **Chord-of-many** | click a point | every one of the 180 dimensions becomes a partial — a handful of countable foreground tones over a bed of hundreds too dense to resolve | `sonify.js → dwell()`; meter in `manifold.js → drawChord()` |
| **Dwell** | hold on a point | the bank sustains; you inhabit a single coordinate | `sonify.js → dwell()` |
| **Translation-seam** | automatic, per point | the clip's own brightest spectral peaks sound against the space's placement; the beating between them is epistemic compression made audible | `sonify.js → _buildSeam()` |
| **Distance-as-consonance** | click a name in "nearest" | two tones form an interval — near points beat slow and consonant, far points clash | `sonify.js → pingDistance()` |
| **Projection-shimmer** | move the cursor between points | the bed re-voices as you navigate empty space; same kind of object, different face | `manifold.js mousemove → api.navigate()` |

## Architecture

```
app.py              PyWebView shell + JS-bridge API (load, point, navigate, distances)
manifold_engine.py  audio -> 180-dim vectors; PCA basis; projections; distances
web/index.html      layout, design tokens, the bed (uncountable-dimension field)
web/sonify.js       Web Audio engine: the four gestures
web/manifold.js     map rendering, navigation, bridge wiring
clips/              put audio here (sample clips included)
```

The Python side never makes sound and the JS side never sees raw audio — the
bridge passes only vectors, projections, and distances. That division is the
point: by the time audio reaches your ears here, it is no longer a recording,
it is a position being sonified.

## Notes

- The default "embedding" is deliberately ML-light (librosa features + PCA) so
  the app runs without a neural-codec download. To navigate a *learned* latent,
  pass `--encodec` (or call `load_directory(folder, encoder=encoder_embedding)`);
  `encoder_embedding` in `manifold_engine.py` returns EnCodec's latent vector
  with the same contract, so nothing else changes.
- The bed is subsampled for the ear and CPU: all foreground partials always
  sound; the bed is thinned by the density control. You are *meant* to fail to
  count it.
