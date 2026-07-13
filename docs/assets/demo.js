/**
 * EchoMap interactive demo - synthetic acoustic SLAM loop in the browser.
 * Mirrors the Python pipeline: chirp → echo → TOA/DOA → classify → map → policy.
 */

const MATERIALS = ["drywall", "wood", "glass", "metal", "carpet", "concrete"];
const MATERIAL_COLORS = {
  drywall: "#c8b48c",
  wood: "#8b5a2b",
  glass: "#7ec8e8",
  metal: "#b8c0c8",
  carpet: "#a07850",
  concrete: "#6e7278",
};

const CHIRP_MODES = {
  GEOMETRY: { f0: 2000, f1: 8000, durationMs: 15, label: "GEOMETRY", tag: "tag-geo", desc: "Long-range wall detection (2-8 kHz)" },
  MATERIAL: { f0: 8000, f1: 20000, durationMs: 8, label: "MATERIAL", tag: "tag-mat", desc: "Surface discrimination (8-20 kHz)" },
  GLASS_PROBE: { f0: 12000, f1: 24000, durationMs: 8, label: "GLASS-PROBE", tag: "tag-glass", desc: "Glass/mirror probe (12-24 kHz, oblique)" },
};

const MODE_ORDER = ["GEOMETRY", "MATERIAL", "GLASS_PROBE"];

const MAP_CELLS = 48;
const MAP_SIZE_M = 6.0;
const CELL_M = MAP_SIZE_M / MAP_CELLS;
const SPEED_OF_SOUND = 343;
const SAMPLE_RATE = 48000;
const ECHO_SAMPLES = 512;

const CONF_THRESHOLD = 0.7;
const STOP_THRESHOLD = 0.85;
const APPROACH_M = 0.75;

const echoCanvas = document.getElementById("echo-canvas");
const echoCtx = echoCanvas.getContext("2d");
const mapCanvas = document.getElementById("map-canvas");
const mapCtx = mapCanvas.getContext("2d");
const modeGrid = document.getElementById("mode-grid");
const resultBlock = document.getElementById("result-block");
const predictionWord = document.getElementById("prediction-word");
const predictionConf = document.getElementById("prediction-conf");
const confidenceFill = document.getElementById("confidence-fill");
const scopeStatus = document.getElementById("scope-status");
const demoHint = document.getElementById("demo-hint");
const stepLog = document.getElementById("step-log");
const statsEl = document.getElementById("demo-stats");

const btnExplore = document.getElementById("btn-explore");
const btnStep = document.getElementById("btn-step");
const btnReset = document.getElementById("btn-reset");
const btnMute = document.getElementById("btn-mute");

let occupancy;
let material;
let materialConf;
let robot = { x: 3.0, y: 3.0, heading: 0 };
let echoBuffer = new Float32Array(ECHO_SAMPLES);
let corrBuffer = new Float32Array(ECHO_SAMPLES);
let animating = false;
let exploring = false;
let exploreTimer = null;
let muted = true;
let audioCtx = null;
let stepCount = 0;
let lastResult = null;

const ROOM = {
  walls: [
    { x0: 0.4, y0: 0.4, x1: 5.6, y1: 0.4, mat: 0 },
    { x0: 5.6, y0: 0.4, x1: 5.6, y1: 5.6, mat: 2 },
    { x0: 5.6, y0: 5.6, x1: 0.4, y1: 5.6, mat: 4 },
    { x0: 0.4, y0: 5.6, x1: 0.4, y1: 0.4, mat: 1 },
    { x0: 2.0, y0: 1.8, x1: 3.4, y1: 1.8, mat: 3 },
    { x0: 4.0, y0: 3.2, x1: 4.0, y1: 4.6, mat: 5 },
  ],
};

function initGrids() {
  occupancy = new Int8Array(MAP_CELLS * MAP_CELLS).fill(-1);
  material = new Int8Array(MAP_CELLS * MAP_CELLS).fill(-1);
  materialConf = new Float32Array(MAP_CELLS * MAP_CELLS);
  robot = { x: 3.0, y: 3.0, heading: 0 };
  stepCount = 0;
  lastResult = null;
  echoBuffer.fill(0);
  corrBuffer.fill(0);
}

function cellIndex(gx, gy) {
  return gy * MAP_CELLS + gx;
}

function worldToGrid(x, y) {
  const gx = Math.max(0, Math.min(MAP_CELLS - 1, Math.floor(x / CELL_M)));
  const gy = Math.max(0, Math.min(MAP_CELLS - 1, Math.floor(y / CELL_M)));
  return [gx, gy];
}

function rayWallHit(ox, oy, angleRad, maxRange = 5.5) {
  const dx = Math.cos(angleRad);
  const dy = Math.sin(angleRad);
  let bestT = Infinity;
  let bestMat = 0;
  let bestX = ox;
  let bestY = oy;

  for (const w of ROOM.walls) {
    const hit = segmentIntersect(ox, oy, ox + dx * maxRange, oy + dy * maxRange, w.x0, w.y0, w.x1, w.y1);
    if (hit && hit.t < bestT) {
      bestT = hit.t;
      bestMat = w.mat;
      bestX = hit.x;
      bestY = hit.y;
    }
  }

  if (bestT === Infinity) return null;
  const range = Math.hypot(bestX - ox, bestY - oy);
  return { range, mat: bestMat, x: bestX, y: bestY };
}

function segmentIntersect(x1, y1, x2, y2, x3, y3, x4, y4) {
  const den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(den) < 1e-9) return null;
  const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
  const u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den;
  if (t < 0 || t > 1 || u < 0 || u > 1) return null;
  return { t, x: x1 + t * (x2 - x1), y: y1 + t * (y2 - y1) };
}

function generateChirp(modeKey, nSamples = 256) {
  const p = CHIRP_MODES[modeKey];
  const duration = p.durationMs / 1000;
  const out = new Float32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const t = (i / nSamples) * duration;
    const frac = i / (nSamples - 1);
    const phase = 2 * Math.PI * (p.f0 * t + 0.5 * (p.f1 - p.f0) * frac * t);
    const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (nSamples - 1)));
    out[i] = Math.sin(phase) * window * 0.8;
  }
  return out;
}

function synthesizeEcho(modeKey, rangeM, matIdx) {
  const chirp = generateChirp(modeKey, 120);
  const delaySamples = Math.floor((2 * rangeM / SPEED_OF_SOUND) * SAMPLE_RATE / 40);
  const delay = Math.max(40, Math.min(ECHO_SAMPLES - 140, delaySamples));
  const buf = new Float32Array(ECHO_SAMPLES);
  const corr = new Float32Array(ECHO_SAMPLES);

  const reflectivity = [0.55, 0.65, 0.95, 0.9, 0.25, 0.7][matIdx] ?? 0.5;
  const sharpness = [0.6, 0.7, 1.2, 1.0, 0.35, 0.5][matIdx] ?? 0.6;
  const modeGain = modeKey === "GLASS_PROBE" ? (matIdx === 2 ? 1.4 : 0.7) : modeKey === "MATERIAL" ? 1.0 : 0.85;

  for (let i = 0; i < chirp.length && i < 80; i++) {
    buf[i] += chirp[i] * 0.25;
  }
  for (let i = 0; i < chirp.length && delay + i < ECHO_SAMPLES; i++) {
    const decay = Math.exp(-i * 0.012 / sharpness);
    buf[delay + i] += chirp[i] * reflectivity * modeGain * decay;
  }
  for (let i = 0; i < ECHO_SAMPLES; i++) {
    buf[i] += (Math.random() - 0.5) * 0.04;
    if (delay + 40 + i < ECHO_SAMPLES && i < 60) {
      buf[delay + 40 + i] += chirp[i % chirp.length] * reflectivity * 0.15 * Math.exp(-i * 0.03);
    }
  }

  for (let i = 0; i < ECHO_SAMPLES; i++) {
    const d = i - delay;
    corr[i] = Math.exp(-(d * d) / (18 * 18)) * reflectivity * modeGain;
  }

  return { buf, corr, delay, chirp };
}

function classifyMaterial(modeKey, trueMat, rangeM) {
  let confBase;
  if (modeKey === "MATERIAL") confBase = 0.78;
  else if (modeKey === "GLASS_PROBE") confBase = trueMat === 2 ? 0.88 : 0.55;
  else confBase = 0.45;

  confBase += Math.max(0, (2.5 - rangeM) * 0.06);
  confBase += (Math.random() - 0.5) * 0.12;
  confBase = Math.max(0.35, Math.min(0.97, confBase));

  let predicted = trueMat;
  if (Math.random() > confBase) {
    predicted = Math.floor(Math.random() * MATERIALS.length);
  }
  if (modeKey === "GLASS_PROBE" && trueMat === 2 && Math.random() < 0.85) {
    predicted = 2;
    confBase = Math.max(confBase, 0.8);
  }
  return { mat: predicted, conf: confBase, trueMat };
}

function updateOccupancy(rangeM, bearingDeg) {
  const headingRad = ((robot.heading + bearingDeg) * Math.PI) / 180;
  const nSteps = Math.floor(rangeM / CELL_M);
  for (let step = 1; step < nSteps; step++) {
    const dist = step * CELL_M;
    const wx = robot.x + dist * Math.cos(headingRad);
    const wy = robot.y + dist * Math.sin(headingRad);
    const [gx, gy] = worldToGrid(wx, wy);
    const idx = cellIndex(gx, gy);
    if (occupancy[idx] === -1) occupancy[idx] = 0;
  }
  const wx = robot.x + rangeM * Math.cos(headingRad);
  const wy = robot.y + rangeM * Math.sin(headingRad);
  const [gx, gy] = worldToGrid(wx, wy);
  occupancy[cellIndex(gx, gy)] = 1;
  return [gx, gy];
}

function updateMaterialAt(gx, gy, matIdx, conf) {
  const idx = cellIndex(gx, gy);
  const prev = materialConf[idx];
  if (conf >= prev) {
    material[idx] = matIdx;
    materialConf[idx] = conf;
  } else {
    materialConf[idx] = prev * 0.7 + conf * 0.3;
  }
}

function lowestConfidenceWall() {
  let best = null;
  for (let gy = 0; gy < MAP_CELLS; gy++) {
    for (let gx = 0; gx < MAP_CELLS; gx++) {
      const idx = cellIndex(gx, gy);
      if (occupancy[idx] !== 1) continue;
      const conf = materialConf[idx];
      if (best === null || conf < best.conf) {
        best = { gx, gy, conf };
      }
    }
  }
  return best;
}

function findFrontier() {
  let best = null;
  let bestScore = -1;
  for (let gy = 1; gy < MAP_CELLS - 1; gy++) {
    for (let gx = 1; gx < MAP_CELLS - 1; gx++) {
      const idx = cellIndex(gx, gy);
      if (occupancy[idx] !== -1) continue;
      let nearFree = false;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        if (occupancy[cellIndex(gx + dx, gy + dy)] === 0) nearFree = true;
      }
      const wx = (gx + 0.5) * CELL_M;
      const wy = (gy + 0.5) * CELL_M;
      const dist = Math.hypot(wx - robot.x, wy - robot.y);
      if (!nearFree && dist > 1.2) continue;
      if (!nearFree && stepCount > 3) continue;
      const score = dist + (nearFree ? 2 : 0) + Math.random() * 0.3;
      if (score > bestScore) {
        bestScore = score;
        best = { gx, gy, x: wx, y: wy };
      }
    }
  }
  return best;
}

function mapConverged() {
  let walls = 0;
  let high = 0;
  for (let i = 0; i < occupancy.length; i++) {
    if (occupancy[i] === 1) {
      walls++;
      if (materialConf[i] >= STOP_THRESHOLD) high++;
    }
  }
  return walls >= 40 && high / walls >= 0.7;
}

function approachPoint(tx, ty) {
  const dx = tx - robot.x;
  const dy = ty - robot.y;
  const dist = Math.hypot(dx, dy);
  if (dist <= APPROACH_M) return { x: robot.x, y: robot.y };
  const scale = (dist - APPROACH_M) / dist;
  return { x: robot.x + dx * scale, y: robot.y + dy * scale };
}

function selectAction() {
  if (mapConverged()) return null;

  const wall = lowestConfidenceWall();

  if (wall && material[cellIndex(wall.gx, wall.gy)] === 2 && wall.conf < 0.8) {
    const tx = (wall.gx + 0.5) * CELL_M;
    const ty = (wall.gy + 0.5) * CELL_M;
    const approach = approachPoint(tx, ty);
    return {
      mode: "GLASS_PROBE",
      targetX: approach.x,
      targetY: approach.y,
      heading: Math.atan2(ty - robot.y, tx - robot.x) * (180 / Math.PI),
      reason: "glass signature - oblique probe",
      lookAt: { x: tx, y: ty },
    };
  }

  if (wall && wall.conf < CONF_THRESHOLD) {
    const tx = (wall.gx + 0.5) * CELL_M;
    const ty = (wall.gy + 0.5) * CELL_M;
    const approach = approachPoint(tx, ty);
    return {
      mode: "MATERIAL",
      targetX: approach.x,
      targetY: approach.y,
      heading: Math.atan2(ty - robot.y, tx - robot.x) * (180 / Math.PI),
      reason: `material conf ${(wall.conf * 100).toFixed(0)}% < 70% - move closer`,
      lookAt: { x: tx, y: ty },
    };
  }

  const frontier = findFrontier();
  if (frontier) {
    return {
      mode: "GEOMETRY",
      targetX: frontier.x,
      targetY: frontier.y,
      heading: Math.atan2(frontier.y - robot.y, frontier.x - robot.x) * (180 / Math.PI),
      reason: "exploring unmapped frontier",
      lookAt: frontier,
    };
  }

  if (wall) {
    const tx = (wall.gx + 0.5) * CELL_M;
    const ty = (wall.gy + 0.5) * CELL_M;
    const approach = approachPoint(tx, ty);
    return {
      mode: "MATERIAL",
      targetX: approach.x,
      targetY: approach.y,
      heading: Math.atan2(ty - robot.y, tx - robot.x) * (180 / Math.PI),
      reason: "refining wall materials",
      lookAt: { x: tx, y: ty },
    };
  }

  return null;
}

function drawEcho(showCorr = true) {
  const w = echoCanvas.width;
  const h = echoCanvas.height;
  echoCtx.fillStyle = "#080a0e";
  echoCtx.fillRect(0, 0, w, h);

  echoCtx.strokeStyle = "#1a2230";
  echoCtx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (h * i) / 4;
    echoCtx.beginPath();
    echoCtx.moveTo(0, y);
    echoCtx.lineTo(w, y);
    echoCtx.stroke();
  }

  echoCtx.strokeStyle = "#2a3548";
  echoCtx.beginPath();
  echoCtx.moveTo(0, h / 2);
  echoCtx.lineTo(w, h / 2);
  echoCtx.stroke();

  if (showCorr) {
    echoCtx.strokeStyle = "rgba(110, 231, 183, 0.35)";
    echoCtx.lineWidth = 1.5;
    echoCtx.beginPath();
    for (let i = 0; i < ECHO_SAMPLES; i++) {
      const x = (i / (ECHO_SAMPLES - 1)) * w;
      const y = h / 2 - corrBuffer[i] * h * 0.4;
      if (i === 0) echoCtx.moveTo(x, y);
      else echoCtx.lineTo(x, y);
    }
    echoCtx.stroke();
  }

  echoCtx.strokeStyle = "#3d9eff";
  echoCtx.lineWidth = 1.5;
  echoCtx.beginPath();
  for (let i = 0; i < ECHO_SAMPLES; i++) {
    const x = (i / (ECHO_SAMPLES - 1)) * w;
    const y = h / 2 - echoBuffer[i] * h * 0.38;
    if (i === 0) echoCtx.moveTo(x, y);
    else echoCtx.lineTo(x, y);
  }
  echoCtx.stroke();

  let peakI = 0;
  let peakV = 0;
  for (let i = 0; i < ECHO_SAMPLES; i++) {
    if (corrBuffer[i] > peakV) {
      peakV = corrBuffer[i];
      peakI = i;
    }
  }
  if (peakV > 0.05) {
    const px = (peakI / (ECHO_SAMPLES - 1)) * w;
    echoCtx.strokeStyle = "#6ee7b7";
    echoCtx.setLineDash([4, 4]);
    echoCtx.beginPath();
    echoCtx.moveTo(px, 8);
    echoCtx.lineTo(px, h - 8);
    echoCtx.stroke();
    echoCtx.setLineDash([]);
    echoCtx.fillStyle = "#6ee7b7";
    echoCtx.font = "11px IBM Plex Mono, monospace";
    echoCtx.fillText("TOA peak", px + 6, 16);
  }
}

function drawMap() {
  const w = mapCanvas.width;
  const h = mapCanvas.height;
  const cellW = w / MAP_CELLS;
  const cellH = h / MAP_CELLS;

  mapCtx.fillStyle = "#1a1f28";
  mapCtx.fillRect(0, 0, w, h);

  for (let gy = 0; gy < MAP_CELLS; gy++) {
    for (let gx = 0; gx < MAP_CELLS; gx++) {
      const idx = cellIndex(gx, gy);
      const occ = occupancy[idx];
      if (occ === -1) {
        mapCtx.fillStyle = "#2a3140";
      } else if (occ === 0) {
        mapCtx.fillStyle = "#e8edf2";
      } else {
        const m = material[idx];
        mapCtx.fillStyle = m >= 0 ? MATERIAL_COLORS[MATERIALS[m]] : "#c45c5c";
      }
      mapCtx.fillRect(gx * cellW, gy * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  mapCtx.strokeStyle = "rgba(255,255,255,0.12)";
  mapCtx.lineWidth = 2;
  for (const wall of ROOM.walls) {
    mapCtx.beginPath();
    mapCtx.moveTo((wall.x0 / MAP_SIZE_M) * w, (wall.y0 / MAP_SIZE_M) * h);
    mapCtx.lineTo((wall.x1 / MAP_SIZE_M) * w, (wall.y1 / MAP_SIZE_M) * h);
    mapCtx.stroke();
  }

  const rx = (robot.x / MAP_SIZE_M) * w;
  const ry = (robot.y / MAP_SIZE_M) * h;
  const headingRad = (robot.heading * Math.PI) / 180;

  if (lastResult) {
    mapCtx.strokeStyle = "rgba(61, 158, 255, 0.5)";
    mapCtx.lineWidth = 1.5;
    mapCtx.beginPath();
    mapCtx.moveTo(rx, ry);
    mapCtx.lineTo(
      rx + Math.cos(headingRad) * (lastResult.range / MAP_SIZE_M) * w,
      ry + Math.sin(headingRad) * (lastResult.range / MAP_SIZE_M) * h
    );
    mapCtx.stroke();
  }

  mapCtx.fillStyle = "#3d9eff";
  mapCtx.beginPath();
  mapCtx.arc(rx, ry, 5, 0, Math.PI * 2);
  mapCtx.fill();

  mapCtx.strokeStyle = "#6ee7b7";
  mapCtx.lineWidth = 2;
  mapCtx.beginPath();
  mapCtx.moveTo(rx, ry);
  mapCtx.lineTo(rx + Math.cos(headingRad) * 12, ry + Math.sin(headingRad) * 12);
  mapCtx.stroke();
}

function updateStats() {
  let known = 0;
  let walls = 0;
  let labeled = 0;
  for (let i = 0; i < occupancy.length; i++) {
    if (occupancy[i] !== -1) known++;
    if (occupancy[i] === 1) {
      walls++;
      if (material[i] >= 0) labeled++;
    }
  }
  const pct = ((known / occupancy.length) * 100).toFixed(0);
  statsEl.textContent = `step ${stepCount} · map ${pct}% · walls ${labeled}/${walls}`;
}

function logStep(text) {
  const line = document.createElement("div");
  line.className = "log-line";
  line.textContent = text;
  stepLog.prepend(line);
  while (stepLog.children.length > 8) stepLog.removeChild(stepLog.lastChild);
}

function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
}

function playChirp(modeKey) {
  if (muted) return;
  ensureAudio();
  const p = CHIRP_MODES[modeKey];
  const duration = p.durationMs / 1000;
  const n = Math.floor(SAMPLE_RATE * duration);
  const buffer = audioCtx.createBuffer(1, n, SAMPLE_RATE);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const frac = i / (n - 1);
    const phase = 2 * Math.PI * (p.f0 * t + 0.5 * (p.f1 - p.f0) * frac * t);
    const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    const audiblePhase = phase * (modeKey === "GEOMETRY" ? 1 : 0.35);
    data[i] = Math.sin(audiblePhase) * window * 0.35;
  }
  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(audioCtx.destination);
  src.start();
}

async function runChirp(modeKey, opts = {}) {
  if (animating) return null;
  animating = true;

  const mode = CHIRP_MODES[modeKey];
  scopeStatus.textContent = "emitting";
  scopeStatus.classList.add("active");
  demoHint.textContent = `Emitting ${mode.label} chirp…`;
  highlightMode(modeKey);
  playChirp(modeKey);

  const chirp = generateChirp(modeKey, 100);
  for (let t = 0; t < 20; t++) {
    for (let i = 0; i < ECHO_SAMPLES - 1; i++) echoBuffer[i] = echoBuffer[i + 1];
    echoBuffer[ECHO_SAMPLES - 1] = chirp[t % chirp.length] * (1 - t / 25) + (Math.random() - 0.5) * 0.02;
    corrBuffer.fill(0);
    drawEcho(false);
    await delay(16);
  }

  const bearing = opts.bearing ?? 0;
  const angleRad = ((robot.heading + bearing) * Math.PI) / 180;
  const hit = rayWallHit(robot.x, robot.y, angleRad);
  if (!hit) {
    scopeStatus.textContent = "no echo";
    demoHint.textContent = "No wall hit in this direction - try another pose.";
    animating = false;
    return null;
  }

  scopeStatus.textContent = "listening";
  const { buf, corr } = synthesizeEcho(modeKey, hit.range, hit.mat);

  for (let t = 0; t < ECHO_SAMPLES; t += 8) {
    for (let i = 0; i < 8 && t + i < ECHO_SAMPLES; i++) {
      echoBuffer[t + i] = buf[t + i];
      corrBuffer[t + i] = corr[t + i];
    }
    drawEcho(true);
    await delay(8);
  }
  echoBuffer.set(buf);
  corrBuffer.set(corr);
  drawEcho(true);

  await delay(200);

  const cls = classifyMaterial(modeKey, hit.mat, hit.range);
  const [gx, gy] = updateOccupancy(hit.range, bearing);
  updateMaterialAt(gx, gy, cls.mat, cls.conf);

  const [rgx, rgy] = worldToGrid(robot.x, robot.y);
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      const x = rgx + dx;
      const y = rgy + dy;
      if (x < 0 || y < 0 || x >= MAP_CELLS || y >= MAP_CELLS) continue;
      const idx = cellIndex(x, y);
      if (occupancy[idx] === -1) occupancy[idx] = 0;
    }
  }

  lastResult = {
    range: hit.range,
    bearing,
    mat: cls.mat,
    conf: cls.conf,
    mode: modeKey,
  };

  resultBlock.hidden = false;
  predictionWord.textContent = MATERIALS[cls.mat];
  const pct = Math.round(cls.conf * 100);
  predictionConf.textContent = `${pct}%`;
  confidenceFill.style.width = `${pct}%`;

  drawMap();
  updateStats();

  scopeStatus.textContent = "classified";
  scopeStatus.classList.remove("active");
  demoHint.textContent = `${mode.label} → ${MATERIALS[cls.mat]} (${pct}%) · range ${hit.range.toFixed(2)} m · bearing ${bearing.toFixed(0)}°`;

  animating = false;
  return lastResult;
}

async function runPolicyStep() {
  if (animating) return;
  const action = selectAction();
  if (!action) {
    demoHint.textContent = "Map converged - exploration complete.";
    scopeStatus.textContent = "done";
    logStep(`step ${stepCount}: map converged`);
    stopExplore();
    return;
  }

  robot.x = clamp(action.targetX, 0.6, 5.4);
  robot.y = clamp(action.targetY ?? robot.y, 0.6, 5.4);
  robot.heading = action.heading;

  let bearing = 0;
  if (action.lookAt) {
    const desired = (Math.atan2(action.lookAt.y - robot.y, action.lookAt.x - robot.x) * 180) / Math.PI;
    robot.heading = desired;
    bearing = 0;
  }

  stepCount += 1;
  const result = await runChirp(action.mode, { bearing });
  if (result) {
    logStep(
      `step ${stepCount}: ${action.mode} - ${action.reason} → ${MATERIALS[result.mat]} (${Math.round(result.conf * 100)}%) @ ${result.range.toFixed(2)}m`
    );
  }
}

async function runManualMode(modeKey) {
  if (animating || exploring) return;
  const bearings = modeKey === "GEOMETRY" ? [-20, 0, 20] : modeKey === "GLASS_PROBE" ? [35] : [0];
  for (const b of bearings) {
    stepCount += 1;
    const result = await runChirp(modeKey, { bearing: b });
    if (result) {
      logStep(
        `manual: ${modeKey} → ${MATERIALS[result.mat]} (${Math.round(result.conf * 100)}%) @ ${result.range.toFixed(2)}m`
      );
    }
    robot.heading += b === 0 ? 15 : 0;
  }
}

function startExplore() {
  if (exploring) {
    stopExplore();
    return;
  }
  exploring = true;
  btnExplore.textContent = "Pause";
  btnExplore.classList.add("active");
  demoHint.textContent = "Adaptive policy exploring…";
  const loop = async () => {
    if (!exploring) return;
    await runPolicyStep();
    if (exploring) exploreTimer = setTimeout(loop, 400);
  };
  loop();
}

function stopExplore() {
  exploring = false;
  if (exploreTimer) clearTimeout(exploreTimer);
  exploreTimer = null;
  btnExplore.textContent = "Auto explore";
  btnExplore.classList.remove("active");
}

function resetDemo() {
  stopExplore();
  initGrids();
  resultBlock.hidden = true;
  predictionWord.textContent = "-";
  predictionConf.textContent = "0%";
  confidenceFill.style.width = "0%";
  scopeStatus.textContent = "idle";
  scopeStatus.classList.remove("active");
  demoHint.textContent = "Pick a chirp mode, or hit Auto explore to watch the adaptive policy map the room.";
  stepLog.innerHTML = "";
  clearModeHighlight();
  drawEcho(false);
  drawMap();
  updateStats();
}

function initModes() {
  MODE_ORDER.forEach((key) => {
    const mode = CHIRP_MODES[key];
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "mode-btn";
    btn.dataset.mode = key;
    btn.innerHTML = `<span class="mode-tag ${mode.tag}">${mode.label}</span>${mode.desc}`;
    btn.addEventListener("click", () => runManualMode(key));
    modeGrid.appendChild(btn);
  });
}

function highlightMode(modeKey) {
  modeGrid.querySelectorAll(".mode-btn").forEach((b) => {
    b.classList.toggle("selected", b.dataset.mode === modeKey);
  });
}

function clearModeHighlight() {
  modeGrid.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("selected"));
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function idleWaveform() {
  if (animating) {
    requestAnimationFrame(idleWaveform);
    return;
  }
  for (let i = 0; i < ECHO_SAMPLES - 1; i++) echoBuffer[i] = echoBuffer[i + 1] * 0.98;
  echoBuffer[ECHO_SAMPLES - 1] = (Math.random() - 0.5) * 0.03;
  if (!lastResult) corrBuffer.fill(0);
  else {
    for (let i = 0; i < ECHO_SAMPLES; i++) corrBuffer[i] *= 0.995;
  }
  drawEcho(!!lastResult);
  requestAnimationFrame(idleWaveform);
}

function bindControls() {
  btnExplore.addEventListener("click", () => {
    ensureAudio();
    startExplore();
  });
  btnStep.addEventListener("click", async () => {
    ensureAudio();
    stopExplore();
    await runPolicyStep();
  });
  btnReset.addEventListener("click", resetDemo);
  btnMute.addEventListener("click", () => {
    muted = !muted;
    btnMute.textContent = muted ? "Sound: off" : "Sound: on";
    btnMute.classList.toggle("active", !muted);
    if (!muted) ensureAudio();
  });
}

initGrids();
initModes();
bindControls();
drawEcho(false);
drawMap();
updateStats();
idleWaveform();
