/* Dimensional vehicle schematics for the 2D whiteboard (custom drawings). */
(function () {
  const NS = "http://www.w3.org/2000/svg";

  function tireSize(spec) {
    const match = String(spec || "").match(/(\d+)\/(\d+)R(\d+)/i);
    if (!match) return { radius: 0.32, width: 0.205 };
    const width = Number(match[1]) / 1000;
    const aspect = Number(match[2]) / 100;
    const rim = Number(match[3]) * 0.0254;
    return { radius: rim / 2 + width * aspect, width };
  }

  function fmt(value, digits) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    return Number(value).toFixed(digits);
  }

  function mm(value) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    return String(Math.round(Number(value) * 1000));
  }

  function el(name, attrs, text) {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs || {}).forEach(([key, value]) => {
      if (value != null) node.setAttribute(key, String(value));
    });
    if (text != null) node.textContent = text;
    return node;
  }

  function mapper(xmin, xmax, zmin, zmax, box) {
    const sx = box.w / Math.max(xmax - xmin, 1e-6);
    const sz = box.h / Math.max(zmax - zmin, 1e-6);
    const s = Math.min(sx, sz);
    const ox = box.x + (box.w - s * (xmax - xmin)) / 2;
    const oz = box.y + (box.h - s * (zmax - zmin)) / 2;
    return {
      x: (X) => ox + (X - xmin) * s,
      z: (Z) => oz + (zmax - Z) * s,
      s,
    };
  }

  function poly(pts, mapX, mapZ) {
    return pts.map(([x, z]) => mapX(x).toFixed(2) + "," + mapZ(z).toFixed(2)).join(" ");
  }

  function dimH(svg, x1, x2, y, label, { above = true, color = "#5a6570" } = {}) {
    const left = Math.min(x1, x2);
    const right = Math.max(x1, x2);
    const mid = (left + right) / 2;
    const tick = 4;
    svg.appendChild(el("line", { x1: left, y1: y, x2: right, y2: y, stroke: color, "stroke-width": 1 }));
    svg.appendChild(el("line", { x1: left, y1: y - tick, x2: left, y2: y + tick, stroke: color, "stroke-width": 1 }));
    svg.appendChild(el("line", { x1: right, y1: y - tick, x2: right, y2: y + tick, stroke: color, "stroke-width": 1 }));
    svg.appendChild(el("text", {
      x: mid,
      y: above ? y - 5 : y + 12,
      fill: "#0f1419",
      "font-size": 10,
      "font-family": "Outfit, ui-sans-serif, sans-serif",
      "font-weight": 600,
      "text-anchor": "middle",
    }, label));
  }

  function dimV(svg, x, y1, y2, label, { left = true, color = "#5a6570", along = false } = {}) {
    const top = Math.min(y1, y2);
    const bot = Math.max(y1, y2);
    const mid = (top + bot) / 2;
    const tick = 4;
    svg.appendChild(el("line", { x1: x, y1: top, x2: x, y2: bot, stroke: color, "stroke-width": 1 }));
    svg.appendChild(el("line", { x1: x - tick, y1: top, x2: x + tick, y2: top, stroke: color, "stroke-width": 1 }));
    svg.appendChild(el("line", { x1: x - tick, y1: bot, x2: x + tick, y2: bot, stroke: color, "stroke-width": 1 }));
    if (along) {
      const tx = left ? x - 9 : x + 9;
      svg.appendChild(el("text", {
        x: tx,
        y: mid,
        fill: "#0f1419",
        "font-size": 10,
        "font-family": "Outfit, ui-sans-serif, sans-serif",
        "font-weight": 600,
        "text-anchor": "middle",
        "dominant-baseline": "middle",
        transform: "rotate(-90 " + tx + " " + mid + ")",
      }, label));
      return;
    }
    svg.appendChild(el("text", {
      x: left ? x - 6 : x + 6,
      y: mid,
      fill: "#0f1419",
      "font-size": 10,
      "font-family": "Outfit, ui-sans-serif, sans-serif",
      "font-weight": 600,
      "text-anchor": left ? "end" : "start",
      "dominant-baseline": "middle",
    }, label));
  }

  function labelView(svg, text, x, y) {
    svg.appendChild(el("text", {
      x,
      y,
      fill: "#5a6570",
      "font-size": 9,
      "font-family": "Barlow Condensed, sans-serif",
      "font-weight": 700,
      "letter-spacing": "0.08em",
    }, text));
  }

  function sourceDot(svg, x, y, color, title) {
    const g = el("g", {});
    g.appendChild(el("title", {}, title));
    g.appendChild(el("circle", { cx: x, cy: y, r: 3.4, fill: color, stroke: "#fff", "stroke-width": 1 }));
    svg.appendChild(g);
  }

  function stations(d, family) {
    const xf = d.WB / 2 + d.FO;
    const xr = -(d.WB / 2 + d.RO);
    const at = (fromFront) => xf - fromFront * d.L;
    const spec = {
      hatch: { hood: 0.15, screen: 0.28, roofEnd: 0.74, tail: 0.9, noseH: 0.43, beltH: 0.54, tailH: 0.7 },
      sedan: { hood: 0.2, screen: 0.34, roofEnd: 0.68, tail: 0.82, noseH: 0.4, beltH: 0.52, tailH: 0.43 },
      liftback: { hood: 0.19, screen: 0.33, roofEnd: 0.7, tail: 0.86, noseH: 0.41, beltH: 0.52, tailH: 0.5 },
      suv: { hood: 0.16, screen: 0.29, roofEnd: 0.76, tail: 0.91, noseH: 0.48, beltH: 0.56, tailH: 0.72 },
      mpv: { hood: 0.12, screen: 0.24, roofEnd: 0.82, tail: 0.94, noseH: 0.52, beltH: 0.58, tailH: 0.8 },
    }[family];
    return {
      xf,
      xr,
      xaF: d.WB / 2,
      xaR: -d.WB / 2,
      xHood: at(spec.hood),
      xScreen: at(spec.screen),
      xRoofR: at(spec.roofEnd),
      xTail: at(spec.tail),
      zRock: Math.min(Math.max(d.GC, d.r * 0.4), d.r * 0.7),
      zBump: Math.min(Math.max(d.GC, d.r * 0.4), d.r * 0.7) + d.r * 0.82,
      zNose: spec.noseH * d.H,
      zBelt: spec.beltH * d.H,
      zTail: spec.tailH * d.H,
    };
  }

  function sideProfile(d, family) {
    const s = stations(d, family);
    const body = [
      [s.xr + 0.04, s.zRock],
      [s.xr, s.zBump],
      [s.xr + 0.02, s.zTail * 0.88],
      [s.xTail, s.zTail],
      [s.xRoofR, d.H],
      [s.xScreen, d.H],
      [s.xHood, s.zNose],
      [s.xf - Math.min(0.08, d.FO * 0.12), s.zNose * 0.9],
      [s.xf, s.zBump],
      [s.xf - 0.04, s.zRock],
    ];
    const glass = [
      [s.xRoofR + 0.06, d.H - 0.06],
      [s.xScreen + 0.04, d.H - 0.06],
      [s.xHood + 0.04, s.zNose + 0.08],
      [s.xRoofR + 0.12, s.zBelt],
    ];
    return { ...s, body, glass };
  }

  function topProfile(d, family) {
    const xf = d.WB / 2 + d.FO;
    const xr = -(d.WB / 2 + d.RO);
    const half = d.W / 2;
    const taperF = family === "mpv" ? 0.1 : 0.16;
    const taperR = family === "sedan" ? 0.12 : 0.08;
    const cabinF = d.WB / 2 - (family === "mpv" ? 0.22 : 0.38);
    const cabinR = -d.WB / 2 - (family === "sedan" ? -0.22 : family === "hatch" ? 0.05 : 0.12);
    const cabinW = half * (family === "mpv" ? 0.78 : 0.72);
    return {
      xf,
      xr,
      body: [
        [xr, -(half - taperR)],
        [xr + d.RO * 0.35, -half],
        [xf - d.FO * 0.45, -half],
        [xf, -(half - taperF)],
        [xf, half - taperF],
        [xf - d.FO * 0.45, half],
        [xr + d.RO * 0.35, half],
        [xr, half - taperR],
      ],
      cabin: [
        [cabinR, -cabinW],
        [cabinF, -cabinW],
        [cabinF, cabinW],
        [cabinR, cabinW],
      ],
    };
  }

  function frontProfile(d, family) {
    const half = d.W / 2;
    const r = d.r;
    const track = d.FT / 2;
    const lip = 0.02;
    const halfW = d.tireW / 2 + lip;
    const zBelt = { mpv: 0.58, suv: 0.55, hatch: 0.52, liftback: 0.5, sedan: 0.5 }[family] * d.H;
    const roofW = half * (family === "mpv" ? 0.78 : family === "suv" ? 0.74 : 0.7);
    const shoulder = half * 0.92;
    const zLow = Math.max(d.GC, r * 0.5);
    const zCrown = r * 1.7;

    function arch(sign, fromOuter) {
      const xOuter = sign * (track + halfW);
      const xInner = sign * (track - halfW);
      const pts = [];
      const n = 8;
      for (let i = 0; i <= n; i += 1) {
        const t = fromOuter ? i / n : 1 - i / n;
        const x = xOuter + (xInner - xOuter) * t;
        const z = zLow + (zCrown - zLow) * Math.sin(Math.PI * t);
        pts.push([x, z]);
      }
      return { pts, xInner };
    }

    const right = arch(1, true);
    const left = arch(-1, false);
    return {
      track,
      body: [
        [-half, zBelt * 0.72],
        [-shoulder, zBelt],
        [-roofW, d.H],
        [roofW, d.H],
        [shoulder, zBelt],
        [half, zBelt * 0.72],
        [half, zLow],
        ...right.pts,
        [right.xInner, d.GC],
        [left.xInner, d.GC],
        ...left.pts,
        [-half, zLow],
      ],
      glass: [
        [-roofW + 0.05, d.H - 0.08],
        [roofW - 0.05, d.H - 0.08],
        [shoulder - 0.08, zBelt + 0.06],
        [-shoulder + 0.08, zBelt + 0.06],
      ],
    };
  }

  function drawSide(d, family, sources) {
    const svg = el("svg", { viewBox: "0 0 440 188", role: "img" });
    const side = sideProfile(d, family);
    const box = { x: 44, y: 18, w: 380, h: 118 };
    const map = mapper(side.xr - 0.06, side.xf + 0.06, -0.06, d.H + 0.12, box);
    labelView(svg, "SIDE", 8, 14);

    svg.appendChild(el("line", {
      x1: map.x(side.xr - 0.04),
      y1: map.z(0),
      x2: map.x(side.xf + 0.04),
      y2: map.z(0),
      stroke: "#c8d0d8",
      "stroke-width": 1,
    }));

    svg.appendChild(el("polygon", {
      points: poly(side.body, map.x, map.z),
      fill: "#dfe6ed",
      stroke: "#1a222c",
      "stroke-width": 1.6,
      "stroke-linejoin": "round",
    }));
    svg.appendChild(el("polygon", {
      points: poly(side.glass, map.x, map.z),
      fill: "#f4f7fb",
      stroke: "#1a222c",
      "stroke-width": 1,
      "stroke-linejoin": "round",
      opacity: 0.95,
    }));

    [side.xaF, side.xaR].forEach((ax) => {
      const cx = map.x(ax);
      const cy = map.z(d.r);
      const ro = d.r * map.s;
      svg.appendChild(el("circle", { cx, cy, r: ro, fill: "#1a222c" }));
      svg.appendChild(el("circle", { cx, cy, r: ro * 0.58, fill: "#e8ecef", stroke: "#1a222c", "stroke-width": 1.1 }));
      svg.appendChild(el("circle", { cx, cy, r: Math.max(1.6, ro * 0.12), fill: "#1a222c" }));
    });

    if (sources.engine) sourceDot(svg, map.x(sources.engine[0]), map.z(sources.engine[2]), "#e11d2e", "Engine");
    if (sources.exhaust) sourceDot(svg, map.x(sources.exhaust[0]), map.z(sources.exhaust[2]), "#2f6fed", "Exhaust");

    const yWB = map.z(0) + 16;
    const yL = yWB + 16;
    dimH(svg, map.x(side.xaR), map.x(side.xaF), yWB, "WB " + fmt(d.WB, 3));
    dimH(svg, map.x(side.xr), map.x(side.xf), yL, "L " + fmt(d.L, 3));
    dimV(svg, 22, map.z(0), map.z(d.H), "H " + fmt(d.H, 3), { along: true });
    return svg;
  }

  function drawTop(d, family) {
    const svg = el("svg", { viewBox: "0 0 360 188", role: "img" });
    const top = topProfile(d, family);
    const half = d.W / 2;
    const box = { x: 30, y: 20, w: 308, h: 122 };
    const spanY = Math.max(half, d.FT / 2, d.RT / 2) + 0.12;
    const map = mapper(top.xr - 0.08, top.xf + 0.08, -spanY - 0.08, spanY + 0.08, box);
    labelView(svg, "TOP", 8, 14);

    svg.appendChild(el("polygon", {
      points: poly(top.body, map.x, map.z),
      fill: "#dfe6ed",
      stroke: "#1a222c",
      "stroke-width": 1.6,
      "stroke-linejoin": "round",
    }));
    svg.appendChild(el("polygon", {
      points: poly(top.cabin, map.x, map.z),
      fill: "#f4f7fb",
      stroke: "#1a222c",
      "stroke-width": 1,
      "stroke-linejoin": "round",
    }));

    const tireW = d.tireW;
    const tireL = d.r * 1.15;
    const axles = [
      [d.WB / 2, d.FT / 2],
      [-d.WB / 2, d.RT / 2],
    ];
    axles.forEach(([ax, trackHalf]) => {
      [-trackHalf, trackHalf].forEach((y) => {
        const x1 = map.x(ax - tireL / 2);
        const y1 = map.z(y + tireW / 2);
        const x2 = map.x(ax + tireL / 2);
        const y2 = map.z(y - tireW / 2);
        svg.appendChild(el("rect", {
          x: Math.min(x1, x2),
          y: Math.min(y1, y2),
          width: Math.abs(x2 - x1),
          height: Math.abs(y2 - y1),
          rx: 2.2,
          fill: "#1a222c",
        }));
      });
    });

    dimH(svg, map.x(-d.WB / 2), map.x(d.WB / 2), 16, "WB " + fmt(d.WB, 3));
    dimH(svg, map.x(top.xr), map.x(top.xf), map.z(-spanY) + 18, "L " + fmt(d.L, 3));
    dimV(svg, 20, map.z(-half), map.z(half), "W " + fmt(d.W, 3), { along: true });
    return svg;
  }

  function drawFrontTire(svg, map, y, radius, width) {
    const cx = map.x(y);
    const top = map.z(radius * 2);
    const bot = map.z(0);
    const half = (width / 2) * map.s;
    const height = bot - top;
    svg.appendChild(el("rect", {
      x: cx - half,
      y: top,
      width: half * 2,
      height,
      rx: Math.min(half, height * 0.16),
      fill: "#1a222c",
    }));
    svg.appendChild(el("circle", {
      cx,
      cy: map.z(radius),
      r: Math.min(half * 0.62, height * 0.16),
      fill: "#e8ecef",
      stroke: "#1a222c",
      "stroke-width": 1,
    }));
  }

  function drawFront(d, family) {
    const svg = el("svg", { viewBox: "0 0 230 200", role: "img" });
    const front = frontProfile(d, family);
    const half = d.W / 2;
    const box = { x: 42, y: 16, w: 168, h: 140 };
    const reach = Math.max(half, front.track + d.tireW / 2) + 0.1;
    const map = mapper(-reach, reach, -0.02, d.H + 0.1, box);
    labelView(svg, "FRONT", 8, 14);

    svg.appendChild(el("line", {
      x1: map.x(-half - 0.04),
      y1: map.z(0),
      x2: map.x(half + 0.04),
      y2: map.z(0),
      stroke: "#c8d0d8",
      "stroke-width": 1,
    }));
    [-front.track, front.track].forEach((y) => {
      drawFrontTire(svg, map, y, d.r, d.tireW);
    });
    svg.appendChild(el("polygon", {
      points: poly(front.body, map.x, map.z),
      fill: "#dfe6ed",
      stroke: "#1a222c",
      "stroke-width": 1.6,
      "stroke-linejoin": "round",
    }));
    svg.appendChild(el("polygon", {
      points: poly(front.glass, map.x, map.z),
      fill: "#f4f7fb",
      stroke: "#1a222c",
      "stroke-width": 1,
      "stroke-linejoin": "round",
    }));

    dimV(svg, 22, map.z(0), map.z(d.H), "H " + fmt(d.H, 3), { along: true });
    dimH(svg, map.x(-half), map.x(half), map.z(0) + 18, "W " + fmt(d.W, 3));
    return svg;
  }

  function renderVehicleSchematic(vehicle) {
    const host = document.getElementById("vehicle-schematic");
    const ident = document.getElementById("schematic-identity");
    if (!host) return;
    host.innerHTML = "";
    const geometry = vehicle && vehicle.geometry;
    if (!geometry || !geometry.dims) {
      host.innerHTML = '<p class="schematic-empty">No dimensional schematic for this vehicle.</p>';
      if (ident) ident.textContent = "";
      return;
    }

    const dims = geometry.dims;
    const tire = tireSize(geometry.tire);
    const d = {
      L: dims.L,
      W: dims.W,
      H: dims.H,
      WB: dims.WB,
      FT: dims.FT,
      RT: dims.RT,
      FO: dims.FO,
      RO: dims.RO,
      GC: dims.GC,
      r: tire.radius,
      tireW: tire.width,
    };
    const family = geometry.body_family || "hatch";
    const sources = geometry.sources || {};

    if (ident) {
      ident.innerHTML = "<strong>" + (geometry.working_identity || vehicle.label || "") + "</strong>";
    }

    const board = document.createElement("div");
    board.className = "schematic-board";
    const views = [drawSide(d, family, sources), drawTop(d, family), drawFront(d, family)];
    views.forEach((svg) => {
      const wrap = document.createElement("div");
      wrap.className = "schematic-view";
      wrap.appendChild(svg);
      board.appendChild(wrap);
    });
    host.appendChild(board);

    const meta = document.createElement("p");
    meta.className = "schematic-meta";
    meta.innerHTML = [
      ["L", mm(d.L)],
      ["W", mm(d.W)],
      ["H", mm(d.H)],
      ["WB", mm(d.WB)],
      ["FO", mm(d.FO)],
      ["RO", mm(d.RO)],
      ["FT", mm(d.FT)],
      ["RT", mm(d.RT)],
      ["GC", mm(d.GC)],
      ["tire", geometry.tire || "—"],
    ].map(([k, v]) => "<span>" + k + " <b>" + v + (k === "tire" ? "" : " mm") + "</b></span>").join("");
    host.appendChild(meta);

    const legend = document.createElement("p");
    legend.className = "schematic-legend";
    legend.innerHTML = '<span><i style="background:#e11d2e"></i>Engine</span><span><i style="background:#2f6fed"></i>Exhaust</span>';
    host.appendChild(legend);
  }

  window.renderVehicleSchematic = renderVehicleSchematic;
})();
