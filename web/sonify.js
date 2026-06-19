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
    this.paused = false;
    this.bedDensity = 0.7;
    this.seamDepth = 0.6;
    this.mode = 'additive';    // sonification type: additive | granular | spectral
    this._cur = null;          // current payload, for re-voicing
    // gesture mute state: true = silenced. Each gesture routes through its own
    // gain node so muting is independent and instant.
    this.muted = { chord:false, seam:false, shimmer:false, distance:false };
  }

  arm() {
    if (this.armed) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.55;
    // gentle limiter so a 180-partial bank can't clip
    const comp = this.ctx.createDynamicsCompressor();
    comp.threshold.value = -24; comp.ratio.value = 12; comp.knee.value = 6;

    // Per-gesture gain nodes so each can be muted independently.
    //   chordGain   <- chord-of-many foreground + bed (the held point)
    //   bedGain     <- the uncountable bed, nested under chordGain
    //   seamGain    <- the translation-seam
    //   distanceGain<- the distance-as-consonance pings
    // Projection-shimmer has no node of its own: it IS the chord re-voiced
    // during navigation, so it is gated in software (see navigateTo).
    this.chordGain = this.ctx.createGain(); this.chordGain.gain.value = 1.0;
    this.bedGain  = this.ctx.createGain();  this.bedGain.gain.value = 0.7;
    this.seamGain = this.ctx.createGain();  this.seamGain.gain.value = 0.0;
    this.distanceGain = this.ctx.createGain(); this.distanceGain.gain.value = 1.0;

    this.bedGain.connect(this.chordGain);   // bed nested under the chord mute
    this.chordGain.connect(this.master);
    this.seamGain.connect(this.master);
    this.distanceGain.connect(this.master);
    this.master.connect(comp);
    comp.connect(this.ctx.destination);
    this.armed = true;
    // apply any mutes chosen before audio was enabled
    this._applyMutes();
  }

  setMaster(v){ if(this.master) this.master.gain.value = v; }
  setBedDensity(v){ this.bedDensity = v; if(this._cur) this.dwell(this._cur, true); }
  setSeamDepth(v){ this.seamDepth = v; if(!this.muted.seam) this._applyMutes(); }
  setMode(name){
    this.mode = name;
    // re-voice the held point in the new sonification type immediately
    if (this.armed && this._cur) this.dwell(this._cur, true);
  }

  /* ---- GESTURE MIXER: mute toggles --------------------------------- */
  setMuted(gesture, isMuted){
    if (!(gesture in this.muted)) return;
    this.muted[gesture] = isMuted;
    this._applyMutes();
  }
  toggleMuted(gesture){
    if (!(gesture in this.muted)) return this.muted[gesture];
    this.muted[gesture] = !this.muted[gesture];
    this._applyMutes();
    return this.muted[gesture];
  }
  _applyMutes(){
    if (!this.armed) return;
    const t = this.ctx.currentTime;
    // chord (and its nested bed) on/off
    this.chordGain.gain.setTargetAtTime(this.muted.chord ? 0.0001 : 1.0, t, 0.05);
    // seam respects both its mute and the seam-depth knob
    const seamLevel = this.muted.seam ? 0.0001 : this.seamDepth * 0.5;
    this.seamGain.gain.setTargetAtTime(seamLevel, t, 0.05);
    // distance pings
    this.distanceGain.gain.setTargetAtTime(this.muted.distance ? 0.0001 : 1.0, t, 0.05);
    // shimmer is gated in software, nothing to set here
  }

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
        if (v.lfo) v.lfo.stop(when + 0.5);   // granular mode tremolo gate
      } catch (e) {}
    });
  }

  /* ---- DWELL + CHORD-OF-MANY ---------------------------------------- */
  dwell(payload, isUpdate=false) {
    if (!this.armed) return;
    // Interacting with the space lifts a pause: a new point can't sound on a
    // suspended clock. Resume, and let the UI know so its label resyncs.
    if (this.paused) {
      this.resume().then(() => { if (this.onResume) this.onResume(); });
    }
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
      const freq = this._coordToFreq(vec[i], i, total);
      const voice = this._makeVoice(this.mode, freq, vec[i], isFg, i, total, t);
      if (voice) fresh.push(voice);
    }
    this.voices = fresh;

    // arm the seam for this point
    this._buildSeam(payload, t);
  }

  /* ---- per-partial voicing, by sonification type -------------------- */
  /* Each mode renders one dimension of the point differently, but the
     foreground/bed split and the meaning (every axis = one partial) is the
     same. Returns a voice record {osc|src, gain, ...} or null. */
  _makeVoice(mode, freq, value, isFg, i, total, t) {
    // foreground -> chordGain (mutable as 'chord'); bed -> bedGain (nested
    // under chordGain, so muting the chord mutes the bed too).
    const dest = isFg ? this.chordGain : this.bedGain;

    if (mode === 'granular') {
      // GRANULAR: each dimension is a fast retriggering grain -> the point as
      // a stuttering cloud. Grain rate rises with |value| (more defining dims
      // chatter faster). Implemented as an oscillator through a tremolo gate.
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      const lfo = this.ctx.createOscillator();
      const lfoGain = this.ctx.createGain();
      osc.type = isFg ? 'sawtooth' : 'sine';
      osc.frequency.value = freq;
      const rate = 6 + Math.min(28, Math.abs(value) * 10); // grains/sec
      lfo.frequency.value = isFg ? rate : rate * 0.5;
      lfo.type = 'square';
      lfoGain.gain.value = isFg ? 0.09 : 0.012 * this.bedDensity;
      // gate the amplitude with the square LFO -> on/off grains
      g.gain.value = 0.0001;
      lfo.connect(lfoGain);
      lfoGain.connect(g.gain);
      osc.connect(g); g.connect(dest);
      osc.start(t); lfo.start(t);
      return {osc, lfo, gain:g, fg:isFg, baseFreq:freq, value};
    }

    if (mode === 'spectral') {
      // SPECTRAL: each dimension is a narrow band of filtered noise centred on
      // its frequency -> the point as a textured resonance, not a pitch. The
      // bed becomes a shimmering wash; foreground bands ring clearly.
      const src = this._noiseSource();
      const bp = this.ctx.createBiquadFilter();
      const g = this.ctx.createGain();
      bp.type = 'bandpass';
      bp.frequency.value = freq;
      bp.Q.value = isFg ? 18 : 6;          // foreground rings tighter
      const amp = isFg ? 0.32 : 0.05 * this.bedDensity;
      g.gain.value = 0.0001;
      src.connect(bp); bp.connect(g); g.connect(dest);
      src.start(t);
      g.gain.setTargetAtTime(amp, t, isFg ? 0.08 : 0.5);
      return {osc:src, gain:g, fg:isFg, baseFreq:freq, value, _noise:true};
    }

    // ADDITIVE (default): every dimension a sustained partial -> the dense
    // chord-of-many. A handful of countable foreground tones over a bed.
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.frequency.value = freq;
    osc.type = isFg ? 'triangle' : 'sine';
    const amp = isFg ? 0.10 : 0.012 * this.bedDensity;
    g.gain.value = 0.0001;
    osc.connect(g); g.connect(dest);
    osc.start(t);
    g.gain.setTargetAtTime(amp, t, isFg ? 0.05 : 0.4);
    return {osc, gain:g, fg:isFg, baseFreq:freq, value};
  }

  // shared looping white-noise buffer source for spectral mode
  _noiseSource() {
    if (!this._noiseBuf) {
      const len = this.ctx.sampleRate * 2;
      const buf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
      const d = buf.getChannelData(0);
      for (let k = 0; k < len; k++) d[k] = Math.random() * 2 - 1;
      this._noiseBuf = buf;
    }
    const src = this.ctx.createBufferSource();
    src.buffer = this._noiseBuf;
    src.loop = true;
    return src;
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
    // seam loudness is the discrepancy knob, unless the gesture is muted
    const seamLevel = this.muted.seam ? 0.0001 : this.seamDepth * 0.5;
    this.seamGain.gain.setTargetAtTime(seamLevel, t, 0.3);
  }

  /* ---- PROJECTION-SHIMMER ------------------------------------------- */
  /* Called on free navigation: same point identity, but the bed balance
     shifts with position — re-voicing the uncountable without moving the
     foreground much. We just re-dwell with the navigated payload.
     If shimmer is muted, navigation does not re-voice — the held point
     stays put while you move the cursor. */
  navigateTo(payload) {
    if (this.muted.shimmer) return;
    this.dwell(payload);
  }

  /* ---- DISTANCE-AS-CONSONANCE --------------------------------------- */
  /* Briefly sound a second point's fundamental against the current one so
     the listener hears similarity as interval. distance in [0..maxd]. */
  pingDistance(distance, maxd) {
    if (!this.armed) return;
    if (this.muted.distance) return;       // gesture muted
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
      osc.connect(g); g.connect(this.distanceGain); osc.start(t);
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

  /* ---- PAUSE: freeze the held sound in place ---- */
  /* Suspends the audio clock, so the chord-of-many holds mid-air and resumes
     exactly where it was. The point you were dwelling on is preserved. */
  async pause() {
    if (!this.armed || this.paused) return;
    if (this.ctx.state === 'running') {
      await this.ctx.suspend();
    }
    this.paused = true;
  }

  async resume() {
    if (!this.armed || !this.paused) return;
    if (this.ctx.state === 'suspended') {
      await this.ctx.resume();
    }
    this.paused = false;
  }

  async togglePause() {
    if (this.paused) { await this.resume(); return false; }
    await this.pause(); return true;
  }

  /* ---- STOP: silence everything and let go of the point ---- */
  /* Clears all voices and forgets the current payload, so the space falls
     silent. Click a point (or navigate) to begin sounding again. If paused,
     we resume the clock first so the stop actually takes effect. */
  async stop() {
    if (!this.armed) return;
    if (this.paused) { await this.resume(); }
    this.silence();
    this._cur = null;
  }
}

window.ManifoldSynth = ManifoldSynth;
