/* manifold.js -------------------------------------------------------------
   Controller: renders the map, handles navigation, and wires the Python
   bridge payloads to the sonification engine.
-------------------------------------------------------------------------- */

const synth = new ManifoldSynth();
const state = {
  points: [], bounds: null, nDims: 0,
  hover: null, held: null, maxDist: 1, ready: false,
};

const bed = document.getElementById('bed');
const map = document.getElementById('map');
const overlay = document.getElementById('overlay');
const bedCtx = bed.getContext('2d');
const mapCtx = map.getContext('2d');
const ovCtx = overlay.getContext('2d');

function resize() {
  for (const c of [bed, map, overlay]) {
    const r = c.parentElement.getBoundingClientRect();
    c.width = r.width * devicePixelRatio;
    c.height = r.height * devicePixelRatio;
    c.style.width = r.width + 'px';
    c.style.height = r.height + 'px';
    c.getContext('2d').setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  }
}
window.addEventListener('resize', () => { resize(); drawMap(); });

/* ---- coordinate mapping: high-dim PCA plane -> screen ---- */
function toScreen(coord) {
  const r = map.getBoundingClientRect();
  const pad = 80;
  const {xmin,xmax,ymin,ymax} = state.bounds;
  const x = pad + (coord[0]-xmin)/(xmax-xmin||1) * (r.width-2*pad);
  const y = pad + (coord[1]-ymin)/(ymax-ymin||1) * (r.height-2*pad);
  return [x, y];
}
function fromScreen(px, py) {
  const r = map.getBoundingClientRect();
  const pad = 80;
  const {xmin,xmax,ymin,ymax} = state.bounds;
  const x = xmin + (px-pad)/(r.width-2*pad) * (xmax-xmin||1);
  const y = ymin + (py-pad)/(r.height-2*pad) * (ymax-ymin||1);
  return [x, y];
}

/* ---- THE BED: ambient particle field = uncountable dimensions ---- */
let bedParticles = [];
function seedBed() {
  const r = bed.getBoundingClientRect();
  const n = 260;
  bedParticles = Array.from({length:n}, () => ({
    x: Math.random()*r.width, y: Math.random()*r.height,
    z: Math.random(),                       // depth -> size & speed
    ph: Math.random()*Math.PI*2,
  }));
}
function drawBed(time) {
  const r = bed.getBoundingClientRect();
  bedCtx.clearRect(0,0,r.width,r.height);
  for (const p of bedParticles) {
    const tw = 0.5 + 0.5*Math.sin(time*0.0008 + p.ph);
    const sz = (0.4 + p.z*1.4);
    bedCtx.globalAlpha = 0.06 + tw*0.10*p.z;
    bedCtx.fillStyle = '#8a6cff';
    bedCtx.beginPath();
    bedCtx.arc(p.x, p.y, sz, 0, Math.PI*2);
    bedCtx.fill();
    // slow drift so the bed never settles -> can't be counted
    p.x += Math.sin(time*0.0002 + p.ph)*0.06*p.z;
    p.y += Math.cos(time*0.00017 + p.ph)*0.05*p.z;
  }
  bedCtx.globalAlpha = 1;
}

/* ---- the points themselves ---- */
function drawMap() {
  if (!state.ready) return;
  const r = map.getBoundingClientRect();
  mapCtx.clearRect(0,0,r.width,r.height);
  // faint links from held point to its neighbours (distance lattice)
  if (state.held && state.held.coord2d) {
    const [hx,hy] = toScreen(state.held.coord2d);
    for (const pt of state.points) {
      const [sx,sy] = toScreen(pt.coord2d);
      mapCtx.strokeStyle = 'rgba(95,227,216,0.05)';
      mapCtx.beginPath(); mapCtx.moveTo(hx,hy); mapCtx.lineTo(sx,sy); mapCtx.stroke();
    }
  }
  for (const pt of state.points) {
    const [x,y] = toScreen(pt.coord2d);
    const isHeld = state.held && state.held.index === pt.index;
    mapCtx.beginPath();
    mapCtx.arc(x,y, isHeld?7:4, 0, Math.PI*2);
    mapCtx.fillStyle = isHeld ? '#5fe3d8' : 'rgba(201,206,224,0.55)';
    mapCtx.fill();
    if (isHeld) {
      mapCtx.beginPath(); mapCtx.arc(x,y,14,0,Math.PI*2);
      mapCtx.strokeStyle='rgba(95,227,216,0.4)'; mapCtx.stroke();
    }
    // label
    mapCtx.fillStyle = 'rgba(107,112,144,0.8)';
    mapCtx.font = '10px IBM Plex Mono, monospace';
    mapCtx.fillText(pt.name, x+9, y+3);
  }
}

/* ---- overlay: the navigation cursor as a 'held position' ---- */
function drawOverlay() {
  const r = overlay.getBoundingClientRect();
  ovCtx.clearRect(0,0,r.width,r.height);
  if (state.hover) {
    const {px,py} = state.hover;
    ovCtx.strokeStyle = 'rgba(232,161,60,0.5)';
    ovCtx.beginPath(); ovCtx.arc(px,py,18,0,Math.PI*2); ovCtx.stroke();
    ovCtx.beginPath();
    ovCtx.moveTo(px-26,py); ovCtx.lineTo(px+26,py);
    ovCtx.moveTo(px,py-26); ovCtx.lineTo(px,py+26);
    ovCtx.strokeStyle='rgba(232,161,60,0.25)'; ovCtx.stroke();
  }
}

/* ---- chord-of-many meter on the panel ---- */
function drawChord(payload) {
  const cv = document.querySelector('#chord canvas');
  const r = cv.parentElement.getBoundingClientRect();
  cv.width = r.width*devicePixelRatio; cv.height = r.height*devicePixelRatio;
  const c = cv.getContext('2d');
  c.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  c.clearRect(0,0,r.width,r.height);
  const vec = payload.vector || [];
  const fg = new Set(payload.foreground_axes || []);
  const n = vec.length;
  for (let i=0;i<n;i++){
    const x = (i/n)*r.width;
    const v = Math.min(1, Math.abs(vec[i])/3);
    const h = v * r.height;
    c.fillStyle = fg.has(i) ? '#5fe3d8' : 'rgba(138,108,255,0.45)';
    c.fillRect(x, r.height-h, Math.max(1, r.width/n - 0.5), h);
  }
}

/* ---- panel update ---- */
function updatePanel(payload) {
  document.getElementById('p-name').textContent = payload.name;
  document.getElementById('p-dims').textContent = payload.n_dims;
  document.getElementById('p-fg').textContent = (payload.foreground_axes||[]).length;
  drawChord(payload);

  // seam mark: offset of the clip's brightest own-peak from space centre
  const seamMark = document.getElementById('seam-mark');
  let off = 0;
  if (payload.mel_profile && payload.mel_profile.length) {
    const prof = payload.mel_profile;
    const peak = prof.indexOf(Math.max(...prof));
    off = (peak / prof.length) - 0.5;     // -0.5..+0.5
  }
  seamMark.style.left = (50 + off*90) + '%';

  // nearest list
  if (state.held && state.held.index >= 0) {
    window.pywebview.api.distances(payload.index).then(res => {
      if (!res.ok) return;
      state.maxDist = Math.max(...res.distances);
      const box = document.getElementById('nearest');
      box.innerHTML = '';
      res.nearest.forEach(n => {
        const row = document.createElement('div');
        row.className = 'nearest-row';
        row.innerHTML = `<span>${n.name}</span><span class="d">${n.distance.toFixed(1)}</span>`;
        row.onclick = () => { holdPoint(n.index);
          synth.pingDistance(n.distance, state.maxDist); };
        box.appendChild(row);
      });
    });
  }
}

/* ---- interactions ---- */
let navThrottle = 0;
map.addEventListener('mousemove', (e) => {
  if (!state.ready) return;
  const r = map.getBoundingClientRect();
  const px = e.clientX - r.left, py = e.clientY - r.top;
  state.hover = {px, py};
  document.getElementById('hud-coord').textContent =
     fromScreen(px,py).map(v=>v.toFixed(2)).join(' , ');
  // throttle bridge calls during free navigation (projection-shimmer)
  const now = performance.now();
  if (now - navThrottle > 90) {
    navThrottle = now;
    const [mx,my] = fromScreen(px,py);
    window.pywebview.api.navigate(mx,my).then(res => {
      if (res.ok) { synth.navigateTo(res); updatePanel(res); }
    });
  }
});

function holdPoint(index) {
  window.pywebview.api.point(index).then(res => {
    if (!res.ok) return;
    state.held = res;
    synth.dwell(res);
    updatePanel(res);
    drawMap();
  });
}
map.addEventListener('click', (e) => {
  if (!state.ready) return;
  const r = map.getBoundingClientRect();
  const px = e.clientX - r.left, py = e.clientY - r.top;
  // hit-test points
  let best=null, bestD=22;
  for (const pt of state.points) {
    const [sx,sy] = toScreen(pt.coord2d);
    const d = Math.hypot(sx-px, sy-py);
    if (d < bestD) { bestD=d; best=pt; }
  }
  if (best) holdPoint(best.index);
});

/* ---- controls ---- */
document.getElementById('vol').oninput = e => synth.setMaster(e.target.value/100);
document.getElementById('bed-density').oninput = e => synth.setBedDensity(e.target.value/100);
document.getElementById('seam-depth').oninput = e => synth.setSeamDepth(e.target.value/100);

const armBtn = document.getElementById('audio-arm');
const pauseBtn = document.getElementById('audio-pause');
const stopBtn = document.getElementById('audio-stop');

armBtn.onclick = () => {
  synth.arm();
  armBtn.textContent = 'audio live';
  armBtn.classList.add('armed');
  pauseBtn.disabled = false;
  stopBtn.disabled = false;
  if (state.held) synth.dwell(state.held);
};

pauseBtn.onclick = async () => {
  if (!synth.armed) return;
  const nowPaused = await synth.togglePause();
  pauseBtn.textContent = nowPaused ? 'resume' : 'pause';
  pauseBtn.classList.toggle('paused', nowPaused);
};

stopBtn.onclick = async () => {
  if (!synth.armed) return;
  await synth.stop();
  // stopping clears the paused state too; reset the pause button
  pauseBtn.textContent = 'pause';
  pauseBtn.classList.remove('paused');
};

// when clicking/navigating the map lifts a pause, resync the button label
synth.onResume = () => {
  pauseBtn.textContent = 'pause';
  pauseBtn.classList.remove('paused');
};

/* ---- boot: receive the space from Python ---- */
function boot(summary) {
  const bootEl = document.getElementById('boot');
  if (!summary || !summary.ok) {
    document.getElementById('boot-msg').textContent =
      (summary && summary.error === 'no folder')
      ? 'no audio folder found — choose one to build a space'
      : ('could not load: ' + (summary ? summary.error : 'unknown'));
    return;
  }
  state.points = summary.points;
  state.bounds = summary.bounds;
  state.nDims = summary.n_dims;
  state.ready = true;
  document.getElementById('hud-dims').textContent =
     `${summary.n_points} points · ${summary.n_dims} dimensions`;
  bootEl.style.display = 'none';
  resize(); seedBed(); drawMap();
  if (summary.points.length) holdPoint(summary.points[0].index);
}
window.__manifoldBoot = boot;

document.getElementById('boot-pick').onclick = () => {
  window.pywebview.api.pick_folder().then(folder => {
    if (!folder) return;
    document.getElementById('boot-msg').textContent = 'analyzing clips…';
    window.pywebview.api.load_folder(folder).then(boot);
  });
};

/* ---- render loop ---- */
function loop(t){ drawBed(t); drawOverlay(); requestAnimationFrame(loop); }
resize(); seedBed(); requestAnimationFrame(loop);
