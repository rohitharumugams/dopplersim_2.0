/* 3D path board editor + synced playback (expects THREE + OrbitControls globals). */
(function () {
  function toThree(x, y, z) {
    return new THREE.Vector3(x, z, y);
  }
  function fromThree(v) {
    return { x: v.x, y: v.z, z: v.y };
  }

  function initEditor() {
    const host = document.getElementById("path3d-canvas-host");
    const pathInput = document.getElementById("path_json");
    const form = document.getElementById("path3d-form");
    const statsEl = document.getElementById("path3d-stats");
    const drawZEl = document.getElementById("path3d-draw-z");
    const micXEl = document.getElementById("mic_x");
    const micYEl = document.getElementById("mic_y");
    const micZEl = document.getElementById("mic_z");
    if (!host || !pathInput || !form || !window.THREE) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc);
    const camera = new THREE.PerspectiveCamera(
      50,
      host.clientWidth / Math.max(host.clientHeight, 1),
      0.1,
      500
    );
    camera.position.set(45, 35, 55);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(host.clientWidth, Math.max(host.clientHeight, 1));
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    host.appendChild(renderer.domElement);
    var controls = null;
    if (THREE.OrbitControls) {
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.mouseButtons = {
        LEFT: THREE.MOUSE.PAN,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.ROTATE,
      };
    }

    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const dir = new THREE.DirectionalLight(0xffffff, 0.45);
    dir.position.set(30, 50, 20);
    scene.add(dir);
    scene.add(new THREE.GridHelper(100, 20, 0x94a3b8, 0xcbd5e1));
    scene.add(new THREE.AxesHelper(20));

    const drawPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(100, 100),
      new THREE.MeshBasicMaterial({
        color: 0x3b82f6,
        transparent: true,
        opacity: 0.08,
        side: THREE.DoubleSide,
      })
    );
    drawPlane.rotation.x = -Math.PI / 2;
    scene.add(drawPlane);

    const micMark = new THREE.Group();
    const micMat = new THREE.LineBasicMaterial({ color: 0x111827 });
    [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ].forEach(function (axis) {
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-axis[0] * 1.5, -axis[1] * 1.5, -axis[2] * 1.5),
        new THREE.Vector3(axis[0] * 1.5, axis[1] * 1.5, axis[2] * 1.5),
      ]);
      micMark.add(new THREE.Line(g, micMat));
    });
    scene.add(micMark);

    var points = [];
    var line = null;
    var markers = [];
    var raycaster = new THREE.Raycaster();
    var pointer = new THREE.Vector2();
    var moved = false;

    function pathLength() {
      var L = 0;
      for (var i = 1; i < points.length; i++) {
        var a = points[i - 1];
        var b = points[i];
        L += Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z);
      }
      return L;
    }
    function rebuildPath() {
      if (line) {
        scene.remove(line);
        line.geometry.dispose();
      }
      markers.forEach(function (m) {
        scene.remove(m);
        m.geometry.dispose();
      });
      markers = [];
      if (points.length >= 2) {
        var verts = points.map(function (p) {
          return toThree(p.x, p.y, p.z);
        });
        var geo = new THREE.BufferGeometry().setFromPoints(verts);
        line = new THREE.Line(
          geo,
          new THREE.LineBasicMaterial({ color: 0x1d4ed8 })
        );
        scene.add(line);
      } else {
        line = null;
      }
      points.forEach(function (p, i) {
        var color =
          i === 0 ? 0x16a34a : i === points.length - 1 ? 0xdc2626 : 0x2563eb;
        var m = new THREE.Mesh(
          new THREE.SphereGeometry(0.45, 12, 12),
          new THREE.MeshBasicMaterial({ color: color })
        );
        m.position.copy(toThree(p.x, p.y, p.z));
        scene.add(m);
        markers.push(m);
      });
      if (statsEl) {
        statsEl.textContent =
          "Path length: " +
          pathLength().toFixed(1) +
          " m · " +
          points.length +
          " pts";
      }
    }
    function syncDrawPlane() {
      drawPlane.position.y = parseFloat(drawZEl.value) || 0;
    }
    function syncMic() {
      var mx = parseFloat(micXEl.value) || 0;
      var my = parseFloat(micYEl.value) || 0;
      var mz = parseFloat(micZEl.value) || 0;
      micMark.position.copy(toThree(mx, my, mz));
    }
    renderer.domElement.addEventListener("pointerdown", function () {
      moved = false;
    });
    renderer.domElement.addEventListener("pointermove", function () {
      moved = true;
    });
    renderer.domElement.addEventListener("pointerup", function (ev) {
      if (moved || ev.button !== 0) return;
      var rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      var hits = raycaster.intersectObject(drawPlane);
      if (!hits.length) return;
      var p = fromThree(hits[0].point);
      var z = parseFloat(drawZEl.value) || 0;
      points.push({ x: p.x, y: p.y, z: z });
      rebuildPath();
    });
    var clearBtn = document.getElementById("path3d-clear");
    var undoBtn = document.getElementById("path3d-undo");
    if (clearBtn)
      clearBtn.addEventListener("click", function () {
        points = [];
        rebuildPath();
      });
    if (undoBtn)
      undoBtn.addEventListener("click", function () {
        points.pop();
        rebuildPath();
      });
    if (drawZEl) drawZEl.addEventListener("input", syncDrawPlane);
    [micXEl, micYEl, micZEl].forEach(function (el) {
      if (el) el.addEventListener("input", syncMic);
    });
    form.addEventListener("submit", function (ev) {
      if (points.length < 2) {
        ev.preventDefault();
        alert("Place at least two points in the 3D board first.");
        return;
      }
      pathInput.value = JSON.stringify(
        points.map(function (p) {
          return {
            x: +p.x.toFixed(4),
            y: +p.y.toFixed(4),
            z: +p.z.toFixed(4),
          };
        })
      );
    });
    syncDrawPlane();
    syncMic();
    rebuildPath();
    (function animate() {
      requestAnimationFrame(animate);
      if (controls) controls.update();
      renderer.render(scene, camera);
    })();
    window.addEventListener("resize", function () {
      var w = host.clientWidth;
      var h = host.clientHeight;
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
  }

  function initPlayback() {
    var urlEl = document.getElementById("path2d-animation-url");
    var dimEl = document.getElementById("path-animation-dim");
    var audio = document.getElementById("path2d-playback-audio");
    var host = document.getElementById("path3d-playback-host");
    if (!urlEl || !audio || !host || !window.THREE) return;
    var dim = dimEl ? JSON.parse(dimEl.textContent) : 2;
    if (dim !== 3) return;

    var anim = null;
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc);
    var camera = new THREE.PerspectiveCamera(
      50,
      host.clientWidth / Math.max(host.clientHeight, 1),
      0.1,
      500
    );
    camera.position.set(40, 30, 50);
    var renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(host.clientWidth, Math.max(host.clientHeight, 1));
    host.appendChild(renderer.domElement);
    var controls = THREE.OrbitControls
      ? new THREE.OrbitControls(camera, renderer.domElement)
      : null;
    if (controls) controls.enableDamping = true;
    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    scene.add(new THREE.GridHelper(100, 20, 0x94a3b8, 0xcbd5e1));
    scene.add(new THREE.AxesHelper(15));

    var pathLine = null;
    var trailLine = null;
    var vehicle = null;

    function ensureScene(data) {
      if (pathLine) return;
      var pathPts = (data.path || []).map(function (p) {
        return toThree(p[0], p[1], p[2]);
      });
      if (pathPts.length > 1) {
        pathLine = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(pathPts),
          new THREE.LineBasicMaterial({ color: 0xcbd5e1 })
        );
        scene.add(pathLine);
      }
      trailLine = new THREE.Line(
        new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ color: 0x2563eb })
      );
      scene.add(trailLine);
      vehicle = new THREE.Mesh(
        new THREE.SphereGeometry(0.8, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xdc2626 })
      );
      scene.add(vehicle);
      var mic = new THREE.Mesh(
        new THREE.SphereGeometry(0.5, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0x111827 })
      );
      mic.position.copy(toThree(data.mic[0], data.mic[1], data.mic[2]));
      scene.add(mic);
    }

    function drawAt(timeS) {
      if (!anim) return;
      ensureScene(anim);
      var ts = anim.t;
      var i = 0;
      while (i + 1 < ts.length && ts[i + 1] <= timeS) i++;
      var t0 = ts[Math.max(0, i)];
      var t1 = ts[Math.min(ts.length - 1, i + 1)];
      var a = t1 > t0 ? (timeS - t0) / (t1 - t0) : 0;
      var i1 = Math.min(ts.length - 1, i + 1);
      var x = anim.x[i] + a * (anim.x[i1] - anim.x[i]);
      var y = anim.y[i] + a * (anim.y[i1] - anim.y[i]);
      var z = anim.z[i] + a * (anim.z[i1] - anim.z[i]);
      vehicle.position.copy(toThree(x, y, z));
      var trail = [];
      for (var k = 0; k <= i; k++)
        trail.push(toThree(anim.x[k], anim.y[k], anim.z[k]));
      trail.push(toThree(x, y, z));
      trailLine.geometry.dispose();
      trailLine.geometry = new THREE.BufferGeometry().setFromPoints(trail);
      if (controls) controls.update();
      renderer.render(scene, camera);
    }
    function tick() {
      drawAt(audio.currentTime || 0);
      if (!audio.paused && !audio.ended) requestAnimationFrame(tick);
    }
    audio.addEventListener("play", function () {
      requestAnimationFrame(tick);
    });
    audio.addEventListener("seeked", function () {
      drawAt(audio.currentTime || 0);
    });
    audio.addEventListener("timeupdate", function () {
      if (audio.paused) drawAt(audio.currentTime || 0);
    });
    fetch(JSON.parse(urlEl.textContent))
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        anim = data;
        drawAt(0);
      });
  }

  function boot() {
    if (document.getElementById("path3d-canvas-host")) initEditor();
    if (document.getElementById("path3d-playback-host")) initPlayback();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
