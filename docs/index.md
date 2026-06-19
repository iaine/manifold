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

[Installation](INSTALLATION)

[Usage](SOUND_OPTIONS)