/**
 * app.js — Shared API client + UI helpers
 * Used by all pages (classify.js, detect.js, batch.js)
 */
'use strict';

// ── API Client ────────────────────────────────────────────────
const API = {
  BASE: 'http://localhost:5000',

  async classify(file, backend='tensorflow', topK=5) {
    const fd = new FormData();
    fd.append('image', file);
    fd.append('backend', backend);
    fd.append('top_k', topK);
    try {
      const r = await fetch(`${API.BASE}/classify`, { method:'POST', body:fd });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch(_) {
      return API._mockClassify(file.name, topK);
    }
  },

  async detect(file, backend='yolo', confidence=0.5) {
    const fd = new FormData();
    fd.append('image', file);
    fd.append('backend', backend);
    fd.append('confidence', confidence);
    try {
      const r = await fetch(`${API.BASE}/detect`, { method:'POST', body:fd });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch(_) {
      return API._mockDetect();
    }
  },

  async batch(files, backend='tensorflow', topK=3) {
    const fd = new FormData();
    files.forEach(f => fd.append('images', f));
    fd.append('backend', backend);
    fd.append('top_k', topK);
    try {
      const r = await fetch(`${API.BASE}/batch`, { method:'POST', body:fd });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch(_) {
      return { results: files.map(f => ({ source:f.name, predictions:API._mockClassify(f.name,topK).predictions, error:null })) };
    }
  },

  // ── Mock data (used when server is offline) ──────────────
  _POOLS: {
    animals: ['golden retriever','Labrador retriever','German shepherd','tabby cat','Persian cat','bald eagle','macaw','golden eagle','African elephant','giant panda'],
    vehicles: ['sports car','pickup truck','convertible','school bus','freight car','aircraft carrier','speedboat','ambulance','fire engine','tank'],
    food: ['pizza','cheeseburger','ice cream','chocolate cake','sushi','french loaf','burrito','hot dog','banana','strawberry'],
    tech: ['laptop','desktop computer','keyboard','cellular telephone','television','remote control','iPod','hard disc','modem','oscilloscope'],
  },

  _mockClassify(name, topK) {
    const pools = Object.values(API._POOLS);
    const pool  = pools[Math.floor(Math.random() * pools.length)];
    const shuffled = [...pool].sort(() => Math.random() - 0.5);
    let rem = 1.0;
    const predictions = shuffled.slice(0, topK).map((label, i) => {
      const conf = i === 0 ? 0.5 + Math.random()*0.4 : rem * (0.08 + Math.random()*0.25);
      rem -= conf;
      return { rank:i+1, label, confidence:Math.max(0.01, parseFloat(conf.toFixed(4))), inference_ms: 30 + (Math.random()*50|0) };
    });
    predictions.sort((a,b) => b.confidence - a.confidence).forEach((p,i) => p.rank = i+1);
    return { source:name, predictions, backend:'mock (server offline)' };
  },

  _mockDetect() {
    const objects = [
      {label:'person',  emoji:'🧍', conf:0.94},
      {label:'car',     emoji:'🚗', conf:0.87},
      {label:'dog',     emoji:'🐕', conf:0.76},
      {label:'backpack',emoji:'🎒', conf:0.61},
      {label:'bicycle', emoji:'🚲', conf:0.55},
      {label:'cat',     emoji:'🐈', conf:0.72},
    ];
    const n = 2 + (Math.random()*3|0);
    return {
      detections: objects.slice(0,n).map(o => ({
        label: o.label, emoji: o.emoji,
        confidence: parseFloat((o.conf - Math.random()*0.06).toFixed(4)),
        bbox: { x:10+(Math.random()*80|0), y:10+(Math.random()*80|0),
                width:80+(Math.random()*100|0), height:80+(Math.random()*100|0) },
        inference_ms: 28 + (Math.random()*30|0),
      })),
      backend: 'mock (server offline)',
    };
  },
};

// ── UI Helpers ────────────────────────────────────────────────
const UI = {
  renderPredictions(container, predictions) {
    container.innerHTML = '';
    predictions.forEach((p, i) => {
      const pct = (p.confidence * 100).toFixed(1);
      const div = document.createElement('div');
      div.className = 'prediction-item fade-in';
      div.style.animationDelay = `${i * 0.06}s`;
      div.innerHTML = `
        <span class="pred-rank">#${p.rank}</span>
        <span class="pred-label">${p.label}</span>
        <div class="pred-bar-wrap">
          <div class="pred-bar" style="width:0%" data-w="${pct}%"></div>
        </div>
        <span class="pred-conf">${pct}%</span>`;
      container.appendChild(div);
    });
    requestAnimationFrame(() => {
      container.querySelectorAll('.pred-bar').forEach(b => b.style.width = b.dataset.w);
    });
  },

  renderDetections(container, detections) {
    container.innerHTML = '';
    if (!detections.length) {
      container.innerHTML = `<div class="result-empty">NO OBJECTS DETECTED<br/><span style="font-size:.6rem;opacity:.5;">try lowering the confidence threshold</span></div>`;
      return;
    }
    const EMOJIS = {person:'🧍',car:'🚗',dog:'🐕',cat:'🐈',bicycle:'🚲',
      backpack:'🎒',bottle:'🍾',bird:'🐦',truck:'🚚',bus:'🚌',default:'📦'};
    detections.forEach((d, i) => {
      const emoji = d.emoji || EMOJIS[d.label] || EMOJIS.default;
      const b = d.bbox;
      const div = document.createElement('div');
      div.className = 'detection-item fade-in';
      div.style.animationDelay = `${i * 0.07}s`;
      div.innerHTML = `
        <div class="det-icon">${emoji}</div>
        <div class="det-label">${d.label}</div>
        <div class="det-conf">${(d.confidence*100).toFixed(1)}%</div>
        <div class="det-bbox">${b.x},${b.y} ${b.width}×${b.height}</div>`;
      container.appendChild(div);
    });
  },

  drawBBoxes(canvas, imgEl, detections) {
    const w = imgEl.naturalWidth  || canvas.width;
    const h = imgEl.naturalHeight || canvas.height;
    const sx = canvas.width  / w;
    const sy = canvas.height / h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);
    const COLORS = ['#00e5a0','#f0b429','#3b9eff','#ff4d4d','#c93bff'];
    detections.forEach((d, i) => {
      const b = d.bbox;
      const x = b.x*sx, y = b.y*sy, bw = b.width*sx, bh = b.height*sy;
      const c = COLORS[i % COLORS.length];
      ctx.strokeStyle = c; ctx.lineWidth = 2;
      ctx.strokeRect(x, y, bw, bh);
      const label = `${d.label} ${(d.confidence*100).toFixed(0)}%`;
      ctx.font = '12px "Share Tech Mono", Courier, monospace';
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = c;
      ctx.fillRect(x, y - 20, tw, 20);
      ctx.fillStyle = '#080b0e';
      ctx.fillText(label, x + 4, y - 5);
    });
  },

  toast(msg, type='info') {
    const t = document.createElement('div');
    t.className = `toast${type === 'error' ? ' error' : ''}`;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3200);
  },
};

// ── Grain canvas (background noise effect) ────────────────────
(function initGrain() {
  const canvas = document.getElementById('grain');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  function resize() { canvas.width = innerWidth; canvas.height = innerHeight; }
  function draw() {
    const img = ctx.createImageData(canvas.width, canvas.height);
    for (let i = 0; i < img.data.length; i += 4) {
      const v = (Math.random() * 25)|0;
      img.data[i] = img.data[i+1] = img.data[i+2] = v;
      img.data[i+3] = 14;
    }
    ctx.putImageData(img, 0, 0);
    requestAnimationFrame(draw);
  }
  resize(); draw();
  window.addEventListener('resize', resize);
})();
