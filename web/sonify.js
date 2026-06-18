/* sonify.js ---------------------------------------------------------------
   The MANIFOLD sonification engine, in Web Audio.

   Maps the four core gestures from the MANIFOLD spec to sound:

     chord-of-many   -> an additive bank: every high-dim coordinate becomes
                        one partial. A countable foreground of loud partials
                        sits over a bed of hundreds too dense to resolve.
     dwell           -> the bank holds steady on the selected point.
     translation-seam-> two detuned copies of the foreground: one tuned to the
                        clip's OWN spectral profile, one to the space's learned
                        position. The beating between them IS the seam.
     distance        -> when comparing two points, their fundamentals form an
                        interval; near points beat slowly/consonant, far clash.

   Everything is parameter-driven from the Python embedding payloads.
-------------------------------------------------------------------------- */

class ManifoldSynth {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.bedGain = null;
    this.seamGain = null;
    this.voices = [];          // active partial oscillators
    this.seamVoices = [];
    this.armed = false;
    this.bedDensity = 0.7;
    this.seamDepth = 0.6;
    this._cur = null;          // current payload, for re-voicing
  }

  arm() {
    if (this.armed) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.55;
    // gentle limiter so a 180-partial bank can't clip
    const comp = this.ctx.createDynamicsCompressor();
    comp.threshold.value = -24; comp.ratio.value = 12; comp.knee.value = 6;
    this.bedGain = this.ctx.createGain();  this.bedGain.gain.value = 0.7;
    this.seamGain = this.ctx.createGain(); this.seamGain.gain.value = 0.0;
    this.bedGain.connect(this.master);
    this.seamGain.connect(this.master);
    this.master.connect(comp);
    comp.connect(this.ctx.destination);
    this.armed = true;
  }

  setMaster(v){ if(this.master) this.master.gain.value = v; }
  setBedDensity(v){ this.bedDensity = v; if(this._cur) this.dwell(this._cur, true); }
  setSeamDepth(v){ this.seamDepth = v; }

  // map a standardized coordinate (~ -3..+3) to a pitch in Hz.
  // foreground partials land in a hearable midrange; the bed spreads wide.
  _coordToFreq(value, idx, total) {
    // base log-frequency spread across the bank
    const lo = Math.log2(70), hi = Math.log2(9000);
    const pos = idx / Math.max(1, total - 1);
    const base = lo + (hi - lo) * pos;
    // the coordinate value perturbs the partial — this is the point's identity
    const cents = value * 90;               // ±3σ -> ~±270 cents
    return Math.pow(2, base + cents / 1200);
  }

  _clearVoices(list, when) {
    list.forEach(v => {
      try {
        v.gain.gain.cancelScheduledValues(when);
        v.gain.gain.setTargetAtTime(0.0001, when, 0.08);
        v.osc.stop(when + 0.5);
      } catch (e) {}
    });
  }

  /* ---- DWELL + CHORD-OF-MANY ---------------------------------------- */
  dwell(payload, isUpdate=false) {
    if (!this.armed) return;
    this._cur = payload;
    const t = this.ctx.currentTime;
    const vec = payload.vector || [];
    const total = vec.length;
    const fg = new Set(payload.foreground_axes || []);

    this._clearVoices(this.voices, t);
    const fresh = [];

    // how many bed partials to actually sound, scaled by density control.
    // we ALWAYS render every foreground partial; the bed is subsampled so
    // the ear meets "too many to count" without the CPU melting.
    const bedBudget = Math.floor(40 + this.bedDensity * (total - 40));

    for (let i = 0; i < total; i++) {
      const isFg = fg.has(i);
      if (!isFg) {
        // subsample the bed deterministically
        if ((i % Math.max(1, Math.round(total / bedBudget))) !== 0) continue;
      }
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      const freq = this._coordToFreq(vec[i], i, total);
      osc.frequency.value = freq;
      osc.type = isFg ? 'triangle' : 'sine';
      // foreground is loud and countable; bed is faint and massed
      const amp = isFg ? 0.10 : 0.012 * this.bedDensity;
      g.gain.value = 0.0001;
      osc.connect(g);
      g.connect(isFg ? this.master : this.bedGain);
      osc.start(t);
      g.gain.setTargetAtTime(amp, t, isFg ? 0.05 : 0.4); // bed fades in slow
      fresh.push({osc, gain:g, fg:isFg, baseFreq:freq, value:vec[i]});
    }
    this.voices = fresh;

    // arm the seam for this point
    this._buildSeam(payload, t);
  }

  /* ---- TRANSLATION-SEAM --------------------------------------------- */
  /* The foreground partials, duplicated and bent toward the clip's own
     spectral peaks. The interference between the space-tuned foreground and
     the audio-tuned copy is the audible "mistuning" of epistemic compression. */
  _buildSeam(payload, t) {
    this._clearVoices(this.seamVoices, t);
    const fresh = [];
    const profile = payload.mel_profile || [];
    const freqs = payload.mel_freqs || [];
    if (!profile.length) { this.seamVoices = fresh; return; }

    // find the clip's own loudest spectral peaks (its "own tuning")
    const idx = profile.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0])
                       .slice(0,6).map(p=>p[1]);
    idx.forEach((mi, k) => {
      const ownFreq = freqs[mi] || 200;
      if (ownFreq < 40 || ownFreq > 10000) return;
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.frequency.value = ownFreq;
      osc.type = 'sine';
      g.gain.value = 0.0001;
      osc.connect(g); g.connect(this.seamGain);
      osc.start(t);
      g.gain.setTargetAtTime(0.06, t, 0.2);
      fresh.push({osc, gain:g, baseFreq:ownFreq});
    });
    this.seamVoices = fresh;
    // seam loudness is the discrepancy knob
    this.seamGain.gain.setTargetAtTime(this.seamDepth * 0.5, t, 0.3);
  }

  /* ---- PROJECTION-SHIMMER ------------------------------------------- */
  /* Called on free navigation: same point identity, but the bed balance
     shifts with position — re-voicing the uncountable without moving the
     foreground much. We just re-dwell with the navigated payload. */
  navigateTo(payload) {
    this.dwell(payload);
  }

  /* ---- DISTANCE-AS-CONSONANCE --------------------------------------- */
  /* Briefly sound a second point's fundamental against the current one so
     the listener hears similarity as interval. distance in [0..maxd]. */
  pingDistance(distance, maxd) {
    if (!this.armed) return;
    const t = this.ctx.currentTime;
    const norm = Math.min(1, distance / (maxd || 1));
    // near -> small interval (consonant); far -> wide detune (clash)
    const baseF = 220;
    const ratio = 1 + norm * 0.5;          // up to a rough tritone-ish clash
    [baseF, baseF * ratio].forEach((f,i)=>{
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.frequency.value = f; osc.type='sine';
      g.gain.value = 0.0001;
      osc.connect(g); g.connect(this.master); osc.start(t);
      g.gain.setTargetAtTime(0.12, t, 0.02);
      g.gain.setTargetAtTime(0.0001, t + 0.5, 0.2);
      osc.stop(t + 1.1);
    });
  }

  silence() {
    if (!this.armed) return;
    const t = this.ctx.currentTime;
    this._clearVoices(this.voices, t);
    this._clearVoices(this.seamVoices, t);
    this.voices = []; this.seamVoices = [];
  }
}

window.ManifoldSynth = ManifoldSynth;
