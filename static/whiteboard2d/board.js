/* 2D whiteboard draw + audio-synced path playback (standalone UI). */
(function () {
  const KMH_PER_MPS = 3.6;
  let currentUnit = document.body.dataset.speedUnit || "mps";

  function bindFreqSlider(sliderId, labelId) {
    const slider = document.getElementById(sliderId);
    const label = document.getElementById(labelId);
    if (!slider || !label) return;
    slider.addEventListener("input", () => {
      label.textContent = slider.value;
    });
  }

  bindFreqSlider("freq_max", "freq-max-value");
  bindFreqSlider("result-freq-max", "result-freq-max-value");

  function updateLabels(unit) {
    const suffix = unit === "kmph" ? "km/h" : "m/s";
    const v1 = document.getElementById("v1-label");
    const v2 = document.getElementById("v2-label");
    if (v1) v1.textContent = "Speed v₁ (" + suffix + ")";
    if (v2) v2.textContent = "Speed along path v₂ (" + suffix + ")";
  }

  document.querySelectorAll('input[name="speed_unit"]').forEach((radio) => {
    radio.addEventListener("change", (event) => {
      const newUnit = event.target.value;
      if (newUnit === currentUnit) return;
      ["v1", "v2"].forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        let value = parseFloat(el.value);
        if (Number.isNaN(value)) return;
        if (currentUnit === "mps" && newUnit === "kmph") value *= KMH_PER_MPS;
        else if (currentUnit === "kmph" && newUnit === "mps") value /= KMH_PER_MPS;
        el.value = Number(value.toFixed(3));
      });
      currentUnit = newUnit;
      updateLabels(newUnit);
    });
  });

  const canvas = document.getElementById("path2d-canvas");
  const pathInput = document.getElementById("path_json");
  const statsEl = document.getElementById("path2d-stats");
  const form = document.getElementById("path2d-form");
  const micXEl = document.getElementById("mic_x");
  const micYEl = document.getElementById("mic_y");
  if (canvas && pathInput && form) {
    const ctx = canvas.getContext("2d");
    const WORLD = { xmin: -50, xmax: 50, ymin: -35, ymax: 35 };
    const PAD = { l: 52, r: 18, t: 18, b: 42 };
    let strokes = [];
    let drawing = false;
    let current = null;

    function plotRect() {
      return { x: PAD.l, y: PAD.t, w: canvas.width - PAD.l - PAD.r, h: canvas.height - PAD.t - PAD.b };
    }
    function worldToCanvas(wx, wy) {
      const p = plotRect();
      return [
        p.x + ((wx - WORLD.xmin) / (WORLD.xmax - WORLD.xmin)) * p.w,
        p.y + ((WORLD.ymax - wy) / (WORLD.ymax - WORLD.ymin)) * p.h,
      ];
    }
    function canvasToWorld(cx, cy) {
      const p = plotRect();
      const wx = WORLD.xmin + ((cx - p.x) / p.w) * (WORLD.xmax - WORLD.xmin);
      const wy = WORLD.ymax - ((cy - p.y) / p.h) * (WORLD.ymax - WORLD.ymin);
      return {
        x: Math.min(WORLD.xmax, Math.max(WORLD.xmin, wx)),
        y: Math.min(WORLD.ymax, Math.max(WORLD.ymin, wy)),
      };
    }
    function eventWorld(ev) {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      return canvasToWorld((ev.clientX - rect.left) * scaleX, (ev.clientY - rect.top) * scaleY);
    }
    function flatPoints() {
      const out = [];
      for (const stroke of strokes) for (const p of stroke) out.push(p);
      return out;
    }
    function pathLengthM() {
      const pts = flatPoints();
      let L = 0;
      for (let i = 1; i < pts.length; i++) {
        L += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
      }
      return L;
    }
    function drawGrid() {
      const p = plotRect();
      ctx.save();
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.beginPath();
      ctx.rect(p.x, p.y, p.w, p.h);
      ctx.clip();
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      for (let x = Math.ceil(WORLD.xmin / 2) * 2; x <= WORLD.xmax; x += 2) {
        const [cx] = worldToCanvas(x, 0);
        ctx.beginPath();
        ctx.moveTo(cx, p.y);
        ctx.lineTo(cx, p.y + p.h);
        ctx.stroke();
      }
      for (let y = Math.ceil(WORLD.ymin / 2) * 2; y <= WORLD.ymax; y += 2) {
        const [, cy] = worldToCanvas(0, y);
        ctx.beginPath();
        ctx.moveTo(p.x, cy);
        ctx.lineTo(p.x + p.w, cy);
        ctx.stroke();
      }
      ctx.strokeStyle = "#cbd5e1";
      ctx.lineWidth = 1.25;
      for (let x = Math.ceil(WORLD.xmin / 10) * 10; x <= WORLD.xmax; x += 10) {
        const [cx] = worldToCanvas(x, 0);
        ctx.beginPath();
        ctx.moveTo(cx, p.y);
        ctx.lineTo(cx, p.y + p.h);
        ctx.stroke();
      }
      for (let y = Math.ceil(WORLD.ymin / 10) * 10; y <= WORLD.ymax; y += 10) {
        const [, cy] = worldToCanvas(0, y);
        ctx.beginPath();
        ctx.moveTo(p.x, cy);
        ctx.lineTo(p.x + p.w, cy);
        ctx.stroke();
      }
      ctx.strokeStyle = "#64748b";
      ctx.lineWidth = 1.5;
      const [ox, oy] = worldToCanvas(0, 0);
      ctx.beginPath();
      ctx.moveTo(p.x, oy);
      ctx.lineTo(p.x + p.w, oy);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(ox, p.y);
      ctx.lineTo(ox, p.y + p.h);
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = "#334155";
      ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let x = Math.ceil(WORLD.xmin / 10) * 10; x <= WORLD.xmax; x += 10) {
        const [cx] = worldToCanvas(x, 0);
        ctx.fillText(String(x), cx, p.y + p.h + 8);
      }
      ctx.fillText("x (m)", p.x + p.w / 2, canvas.height - 16);
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let y = Math.ceil(WORLD.ymin / 10) * 10; y <= WORLD.ymax; y += 10) {
        const [, cy] = worldToCanvas(0, y);
        ctx.fillText(String(y), p.x - 8, cy);
      }
      ctx.save();
      ctx.translate(14, p.y + p.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText("y (m)", 0, 0);
      ctx.restore();
      const micX = parseFloat(micXEl && micXEl.value) || 0;
      const micY = parseFloat(micYEl && micYEl.value) || 0;
      const [mx, my] = worldToCanvas(micX, micY);
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(mx - 8, my);
      ctx.lineTo(mx + 8, my);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(mx, my - 8);
      ctx.lineTo(mx, my + 8);
      ctx.stroke();
      ctx.fillStyle = "#111827";
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText("mic", mx + 10, my - 4);
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.strokeStyle = "#1d4ed8";
      ctx.lineWidth = 2.5;
      for (const stroke of strokes) {
        if (stroke.length < 2) continue;
        ctx.beginPath();
        const [sx, sy] = worldToCanvas(stroke[0].x, stroke[0].y);
        ctx.moveTo(sx, sy);
        for (let i = 1; i < stroke.length; i++) {
          const [px, py] = worldToCanvas(stroke[i].x, stroke[i].y);
          ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
      const pts = flatPoints();
      if (pts.length) {
        const [sx, sy] = worldToCanvas(pts[0].x, pts[0].y);
        const [ex, ey] = worldToCanvas(pts[pts.length - 1].x, pts[pts.length - 1].y);
        ctx.fillStyle = "#16a34a";
        ctx.beginPath();
        ctx.arc(sx, sy, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#dc2626";
        ctx.beginPath();
        ctx.arc(ex, ey, 5, 0, Math.PI * 2);
        ctx.fill();
      }
      if (statsEl) statsEl.textContent = "Path length: " + pathLengthM().toFixed(1) + " m";
    }

    function pointerDown(ev) {
      ev.preventDefault();
      drawing = true;
      current = [];
      strokes.push(current);
      current.push(eventWorld(ev));
      canvas.setPointerCapture(ev.pointerId);
      drawGrid();
    }
    function pointerMove(ev) {
      if (!drawing || !current) return;
      ev.preventDefault();
      const w = eventWorld(ev);
      const last = current[current.length - 1];
      if (Math.hypot(w.x - last.x, w.y - last.y) < 0.15) return;
      current.push(w);
      drawGrid();
    }
    function pointerUp(ev) {
      if (!drawing) return;
      drawing = false;
      current = null;
      try {
        canvas.releasePointerCapture(ev.pointerId);
      } catch (_) {}
      drawGrid();
    }

    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointermove", pointerMove);
    canvas.addEventListener("pointerup", pointerUp);
    canvas.addEventListener("pointercancel", pointerUp);
    const clearBtn = document.getElementById("path2d-clear");
    const undoBtn = document.getElementById("path2d-undo");
    if (clearBtn) clearBtn.addEventListener("click", () => { strokes = []; drawGrid(); });
    if (undoBtn) undoBtn.addEventListener("click", () => { strokes.pop(); drawGrid(); });
    if (micXEl) micXEl.addEventListener("input", drawGrid);
    if (micYEl) micYEl.addEventListener("input", drawGrid);
    form.addEventListener("submit", (ev) => {
      const pts = flatPoints();
      if (pts.length < 2) {
        ev.preventDefault();
        alert("Draw a path with at least two points first.");
        return;
      }
      pathInput.value = JSON.stringify(pts.map((p) => ({ x: +p.x.toFixed(4), y: +p.y.toFixed(4) })));
    });
    drawGrid();
  }

  // Audio-synced path playback
  const urlEl = document.getElementById("path2d-animation-url");
  const audio = document.getElementById("path2d-playback-audio");
  const playCanvas = document.getElementById("path2d-playback-canvas");
  if (urlEl && audio && playCanvas) {
    let anim = null;
    const ctx = playCanvas.getContext("2d");
    const PAD = { l: 44, r: 14, t: 14, b: 36 };

    function worldOf() {
      return (anim && anim.world) || { xmin: -50, xmax: 50, ymin: -35, ymax: 35 };
    }
    function plotRect() {
      return {
        x: PAD.l,
        y: PAD.t,
        w: playCanvas.width - PAD.l - PAD.r,
        h: playCanvas.height - PAD.t - PAD.b,
      };
    }
    function toCanvas(wx, wy) {
      const W = worldOf();
      const p = plotRect();
      return [
        p.x + ((wx - W.xmin) / (W.xmax - W.xmin)) * p.w,
        p.y + ((W.ymax - wy) / (W.ymax - W.ymin)) * p.h,
      ];
    }
    function drawAt(timeS) {
      if (!anim) return;
      const p = plotRect();
      const W = worldOf();
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, playCanvas.width, playCanvas.height);
      ctx.save();
      ctx.beginPath();
      ctx.rect(p.x, p.y, p.w, p.h);
      ctx.clip();
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      for (let x = Math.ceil(W.xmin / 10) * 10; x <= W.xmax; x += 10) {
        const [cx] = toCanvas(x, 0);
        ctx.beginPath();
        ctx.moveTo(cx, p.y);
        ctx.lineTo(cx, p.y + p.h);
        ctx.stroke();
      }
      for (let y = Math.ceil(W.ymin / 10) * 10; y <= W.ymax; y += 10) {
        const [, cy] = toCanvas(0, y);
        ctx.beginPath();
        ctx.moveTo(p.x, cy);
        ctx.lineTo(p.x + p.w, cy);
        ctx.stroke();
      }
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 1.2;
      const [ox, oy] = toCanvas(0, 0);
      ctx.beginPath();
      ctx.moveTo(p.x, oy);
      ctx.lineTo(p.x + p.w, oy);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(ox, p.y);
      ctx.lineTo(ox, p.y + p.h);
      ctx.stroke();
      const path = anim.path || [];
      if (path.length > 1) {
        ctx.strokeStyle = "#cbd5e1";
        ctx.lineWidth = 2;
        ctx.beginPath();
        const [sx, sy] = toCanvas(path[0][0], path[0][1]);
        ctx.moveTo(sx, sy);
        for (let i = 1; i < path.length; i++) {
          const [px, py] = toCanvas(path[i][0], path[i][1]);
          ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
      const ts = anim.t;
      let i = 0;
      while (i + 1 < ts.length && ts[i + 1] <= timeS) i++;
      const t0 = ts[Math.max(0, i)];
      const t1 = ts[Math.min(ts.length - 1, i + 1)];
      const a = t1 > t0 ? (timeS - t0) / (t1 - t0) : 0;
      const i1 = Math.min(ts.length - 1, i + 1);
      const vx = anim.x[i] + a * (anim.x[i1] - anim.x[i]);
      const vy = anim.y[i] + a * (anim.y[i1] - anim.y[i]);
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      const [sx0, sy0] = toCanvas(anim.x[0], anim.y[0]);
      ctx.moveTo(sx0, sy0);
      for (let k = 1; k <= i; k++) {
        const [px, py] = toCanvas(anim.x[k], anim.y[k]);
        ctx.lineTo(px, py);
      }
      const [cx, cy] = toCanvas(vx, vy);
      ctx.lineTo(cx, cy);
      ctx.stroke();
      const [mx, my] = toCanvas(anim.mic[0], anim.mic[1]);
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(mx - 7, my);
      ctx.lineTo(mx + 7, my);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(mx, my - 7);
      ctx.lineTo(mx, my + 7);
      ctx.stroke();
      ctx.fillStyle = "#dc2626";
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      ctx.fillStyle = "#334155";
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("x (m)", p.x + p.w / 2, playCanvas.height - 10);
      ctx.fillText(timeS.toFixed(2) + " s", p.x + p.w - 8, 12);
    }
    function tick() {
      drawAt(audio.currentTime || 0);
      if (!audio.paused && !audio.ended) requestAnimationFrame(tick);
    }
    audio.addEventListener("play", () => requestAnimationFrame(tick));
    audio.addEventListener("seeked", () => drawAt(audio.currentTime || 0));
    audio.addEventListener("timeupdate", () => {
      if (audio.paused) drawAt(audio.currentTime || 0);
    });
    fetch(JSON.parse(urlEl.textContent))
      .then((r) => r.json())
      .then((data) => {
        anim = data;
        drawAt(0);
      })
      .catch(() => {});
  }

  document.querySelectorAll(".psd-range-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      const img = document.getElementById("psd-comparison-img");
      if (!img) return;
      const range = btn.dataset.range;
      if (range === "low" && img.dataset.low) img.src = img.dataset.low;
      else if (img.dataset.full) img.src = img.dataset.full;
      document.querySelectorAll(".psd-range-toggle").forEach((b) => {
        b.classList.toggle("bg-blue-600", b === btn);
        b.classList.toggle("text-white", b === btn);
        b.classList.toggle("text-slate-400", b !== btn);
      });
    });
  });
})();
