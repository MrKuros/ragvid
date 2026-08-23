// A DOM, a fetch and a WebGL2 context, all stubbed, so index.html's <script>
// can be exercised under plain node. There is no browser here and there will
// not be one in CI, but almost everything the live preview decides -- whether
// to use the canvas at all, which cube goes into which texture unit, what a
// failure falls back to -- is ordinary JavaScript and is exactly the part that
// breaks silently.
//
// The page's <script> is spliced in at PAGE_SOURCE_HERE by tests/test_page.py,
// so this file, the prelude and the checks all share one scope: the page's
// top-level `let`/`const` (S, liveOn, parseCube, ...) are reachable below.

// ---- the smallest DOM that the page can run against -------------------------

const els = new Map();
let nextId = 0;

function mk(id) {
  const listeners = {};
  let kids = null;
  const el = {
    id: id || `anon${++nextId}`,
    hidden: false, disabled: false, textContent: "", innerHTML: "", value: "",
    src: "", title: "", placeholder: "", type: "", checked: false,
    paused: true, currentTime: 0, readyState: 0, videoWidth: 0, videoHeight: 0,
    width: 0, height: 0,
    style: {}, dataset: {}, attrs: {}, listeners,
    classList: {
      set: new Set(),
      add(c) { this.set.add(c); }, remove(c) { this.set.delete(c); },
      contains(c) { return this.set.has(c); },
      toggle(c, on) { on === undefined ? (this.set.has(c) ? this.set.delete(c) : this.set.add(c))
                                       : (on ? this.set.add(c) : this.set.delete(c)); },
    },
    get children() { return kids || (kids = [mk(), mk(), mk()]); },
    addEventListener(ev, fn) { (listeners[ev] = listeners[ev] || []).push(fn); },
    removeEventListener() {},
    dispatch(ev, arg) { (listeners[ev] || []).forEach((fn) => fn(arg || { target: el })); },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return this.attrs[k]; },
    removeAttribute(k) { delete this.attrs[k]; },
    querySelector() { return mk(); },
    querySelectorAll() { return []; },
    append() {}, prepend() {}, insertAdjacentHTML() {},
    setSelectionRange() {}, focus() {}, click() { if (el.onclick) el.onclick({}); },
    showModal() {}, close() {}, load() {},
    play() { el.paused = false; el.dispatch("play"); return Promise.resolve(); },
    pause() { el.paused = true; el.dispatch("pause"); },
    getContext: (kind) => (kind === "webgl2" ? (glAvailable ? fakeGl : null) : null),
    selectedOptions: [{ textContent: "" }],
  };
  return el;
}

function $$(id) {
  if (!els.has(id)) els.set(id, mk(id));
  return els.get(id);
}

globalThis.document = {
  getElementById: $$,
  createElement: () => mk(),
  querySelector: () => null,
  querySelectorAll: () => [],
};
globalThis.addEventListener = () => {};
globalThis.requestAnimationFrame = () => 0;   // never loop: the checks draw by hand
globalThis.confirm = () => false;
globalThis.XMLHttpRequest = function () {
  return { open() {}, send() {}, upload: {}, setRequestHeader() {} };
};
globalThis.FormData = function () { return { append() {} }; };

// ---- a WebGL2 context that records rather than renders -----------------------
// Unknown properties come back as their own name, so `gl.TEXTURE_3D` is the
// string "TEXTURE_3D" and every recorded call reads like the source does.

let glAvailable = true;
let shadersCompile = true;
const glCalls = [];
const uniforms = {};
let draws = 0;
const shaderSources = [];
let boundTex = null, activeUnit = null;

const glBase = {
  createShader: () => ({}),
  shaderSource: (s, src) => shaderSources.push(src),
  compileShader: () => {},
  getShaderParameter: () => shadersCompile,
  getShaderInfoLog: () => "stub: compile refused",
  createProgram: () => ({}),
  attachShader: () => {},
  linkProgram: () => {},
  getProgramParameter: () => true,
  useProgram: () => {},
  getUniformLocation: (p, name) => name,         // the name IS the location here
  createBuffer: () => ({}),
  bindBuffer: () => {}, bufferData: () => {},
  getAttribLocation: () => 0,
  enableVertexAttribArray: () => {}, vertexAttribPointer: () => {},
  createTexture: () => ({ tex: glCalls.push(["createTexture"]) }),
  activeTexture: (u) => { activeUnit = u; },
  bindTexture: (target, tex) => { boundTex = [target, tex]; },
  texParameteri: (target, k, v) => glCalls.push(["texParameteri", target, k, v]),
  texImage2D: () => glCalls.push(["texImage2D"]),
  texImage3D: (target, lvl, ifmt, n) => glCalls.push(["texImage3D", n, activeUnit]),
  uniform1i: (loc, v) => { uniforms[loc] = v; glCalls.push(["uniform1i", loc, v]); },
  uniform4f: (loc, ...v) => { uniforms[loc] = v; glCalls.push(["uniform4f", loc, v]); },
  viewport: () => {},
  drawArrays: () => { draws++; },
  pixelStorei: () => {},
};
const fakeGl = new Proxy(glBase, { get: (t, k) => (k in t ? t[k] : k) });

// ---- fetch -------------------------------------------------------------------

const fetched = [];
let routes = {};

globalThis.fetch = async (url, opts) => {
  fetched.push(url);
  const path = String(url).split("?")[0];
  // Whole URL first, so a check can answer one layer's cube differently from
  // the rest; path otherwise.
  const body = routes[String(url)] !== undefined ? routes[String(url)] : routes[path];
  if (body === undefined) {
    return { ok: false, status: 404,
             headers: { get: () => "application/json" },
             json: async () => ({ error: { type: "NotFound", message: path } }),
             text: async () => "" };
  }
  const json = typeof body !== "string";
  return { ok: true, status: 200,
           headers: { get: () => (json ? "application/json" : "text/plain") },
           json: async () => body, text: async () => body };
};

// ---- assertions --------------------------------------------------------------

let checked = 0;
function ok(cond, what) {
  checked++;
  if (!cond) { console.error(`FAIL: ${what}`); process.exit(1); }
}
const settle = () => new Promise((r) => setTimeout(r, 0));

// PAGE_SOURCE_HERE

// ---- the checks --------------------------------------------------------------

const IDENTITY_CUBE = `TITLE "t"
LUT_3D_SIZE 2
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0
0.000000 0.000000 0.000000
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
1.000000 1.000000 0.000000
0.000000 0.000000 1.000000
1.000000 0.000000 1.000000
0.000000 1.000000 1.000000
1.000000 1.000000 1.000000
`;

const EFFECTS = { denoise: 0, glow: 0, softness: 0, grain: 0, vignette: 0, fringe: 0 };
const SPEC = { look_mix: 1, rationale: "", effects: { ...EFFECTS } };

function state(over = {}) {
  return {
    open: true, api_version: EXPECTED_API, version: 3, source: "/clips/a.mp4",
    name: "a.mp4", duration: 10, planned: true, can_undo: true, history_depth: 1,
    steps: [], spec: { ...SPEC }, intent: null, auto_balance: true, balance: "",
    layers: [], stats: {}, input_lut: null, input_format: null, providers: [],
    provider: "groq", model: "m", configured: true, ...over,
  };
}

function reset(over = {}) {
  fetched.length = 0;
  draws = 0;
  routes = { "/api/state": state(over), "/media/cube": IDENTITY_CUBE };
  const v = $("vid");
  v.videoWidth = 640; v.videoHeight = 360; v.readyState = 2; v.currentTime = 0;
  v.paused = true;
  $("imgGraded").src = "";
  return state(over);
}

// An <img> src is not a fetch, so this is what "the server rendered it" means.
const frameAsked = () => String($("imgGraded").src).startsWith("/media/frame");

// What a page reload would do -- the live path caches its verdicts on purpose.
function forget() {
  glOk = null; videoBroken = false;
  vidFor = null; cubeFor = null; techFor = null; layersFor = null; layerCount = 0;
  for (const k of Object.keys(uniforms)) delete uniforms[k];
  glCalls.length = 0;
}

// region.py's "top" and "center" entries, as /api/state serialises a Region.
const REGION_TOP = { shape: "linear", edge: "top", extent: 0.4, cx: 0.5, cy: 0.5,
                     rx: 0.5, ry: 0.5, softness: 0.4, invert: false };
const REGION_CENTER = { shape: "radial", edge: "top", extent: 0.5, cx: 0.5, cy: 0.5,
                        rx: 0.6, ry: 0.75, softness: 0.7, invert: false };
const layer = (region) => ({ region, spec: { ...SPEC } });

(async () => {
  // First, and deliberately phrased in terms that need nothing new to express:
  // a live canvas means the page stops asking ffmpeg for stills. Run against
  // the page as it was, this is the line that fails.
  glAvailable = true; shadersCompile = true;
  render(reset());
  await settle();
  ok(!frameAsked(), "no ffmpeg round trip while the canvas is live");

  // --- the cube parser, which is the one piece of real arithmetic here -------
  const parsed = parseCube(IDENTITY_CUBE);
  ok(parsed && parsed.size === 2, "a .cube parses to its LUT_3D_SIZE");
  ok(parsed.data.length === 24, "2^3 entries, three floats each");
  // Red varies fastest: entry 1 is (1,0,0), entry 2 is (0,1,0), entry 4 is (0,0,1).
  ok(parsed.data[3] === 1 && parsed.data[4] === 0 && parsed.data[5] === 0,
     "entry 1 is red -- red varies fastest, as lut.py writes it");
  ok(parsed.data[6] === 0 && parsed.data[7] === 1, "entry 2 steps green");
  ok(parsed.data[12] === 0 && parsed.data[14] === 1, "entry 4 steps blue");
  ok(parseCube(IDENTITY_CUBE.replace("DOMAIN_MAX 1.0 1.0 1.0", "DOMAIN_MAX 4.0 4.0 4.0")) === null,
     "a cube whose domain is not 0..1 is refused, not misread");
  ok(parseCube("LUT_3D_SIZE 2\n0 0 0\n") === null, "a truncated cube is refused");

  // --- the ordinary case, still the render above ----------------------------
  ok(liveOn === true, "the canvas is live for a plain colour grade");
  ok($("gl").hidden === false, "the canvas is showing");
  ok($("playBtn").hidden === false && $("clipBtn").hidden === false,
     "play and clipping appear with the live preview");
  ok($("vid").src === "/media/source", "the <video> is pointed at the clip");
  ok(fetched.some((u) => String(u).startsWith("/media/cube?v=3")),
     "the grade cube is fetched at the current version");
  ok(uniforms.useGrade === 1 && uniforms.gradeN === 2, "the grade LUT is bound and in use");
  ok(uniforms.useTech === 0, "no technical LUT on ordinary footage");

  // The LUT textures must not be hardware-filtered: the shader interpolates.
  const lutFilters = glCalls.filter((c) => c[0] === "texParameteri" && c[1] === "TEXTURE_3D"
                                        && String(c[2]).endsWith("_FILTER"));
  ok(lutFilters.length >= 2, "the 3D textures set their filters explicitly");
  ok(lutFilters.every((c) => c[3] === "NEAREST"),
     "3D LUT sampling is NEAREST -- hardware trilinear is the wrong interpolation");

  // --- scrubbing seeks the decoder instead of asking the server -------------
  fetched.length = 0;
  $("scrub").value = "4.5";
  $("scrub").dispatch("input", { target: { value: "4.5" } });
  ok($("vid").currentTime === 4.5, "a drag seeks the open decoder");
  ok(!frameAsked(), "a drag costs no request at all");

  // --- hold to compare bypasses the grade in the same shader ----------------
  $("frame").dispatch("mousedown", { preventDefault() {} });
  ok(uniforms.useGrade === 0, "holding shows the source through the same path");
  $("frame").dispatch("mouseup");
  ok(uniforms.useGrade === 1, "and the grade comes back on release");

  // --- the clipping toggle (C5) --------------------------------------------
  ok(uniforms.marks === 0, "clipping marks start off");
  const before = draws;
  $("clipBtn").onclick();
  ok(uniforms.marks === 1, "the toggle turns the marks on");
  ok(draws > before, "and redraws immediately");
  ok($("clipBtn").classList.contains("on"), "the button shows it is on");
  ok($("clipBtn").getAttribute("aria-pressed") === "true", "and says so to a screen reader");
  $("clipBtn").onclick();
  ok(uniforms.marks === 0, "and off again");

  // --- playback (C2) --------------------------------------------------------
  $("playBtn").onclick();
  ok($("vid").paused === false, "the one button plays");
  ok($("playIcon").getAttribute("d").startsWith("M7 5h3"), "the icon becomes a pause");
  $("playBtn").onclick();
  ok($("vid").paused === true, "and pauses");
  ok($("playIcon").getAttribute("d").startsWith("M8 5l11"), "the icon goes back");

  // --- the camera's log conversion is a second LUT, in the right order ------
  render(reset({ input_lut: "/w/log_slog3.cube", input_format: "slog3", version: 4 }));
  await settle();
  ok(fetched.some((u) => String(u).includes("input=1")), "the technical LUT is fetched too");
  ok(uniforms.useTech === 1, "and applied");
  const units = glCalls.filter((c) => c[0] === "texImage3D").map((c) => c[2]);
  ok(units.includes("TEXTURE01") && units.includes("TEXTURE02"),
     "technical on unit 1, grade on unit 2 -- never the same texture twice");

  // --- regional layers composite in the shader, not on the server ----------
  forget();
  render(reset({ layers: [layer(REGION_TOP), layer(REGION_CENTER)], version: 20 }));
  await settle();
  ok(liveOn === true, "a regional grade stays live -- the shader composites it");
  ok(!frameAsked(), "and costs no ffmpeg round trip");
  ok(fetched.some((u) => String(u).startsWith("/media/cube?layer=0&v=20")),
     "layer 0's cube is fetched at the current version");
  ok(fetched.some((u) => String(u).startsWith("/media/cube?layer=1&v=20")),
     "and layer 1's");
  ok(uniforms.nLayers === 2, "both layers are switched on");
  ok(uniforms["layN[0]"] === 2 && uniforms["layN[1]"] === 2, "each layer knows its cube size");
  const layerUnits = glCalls.filter((c) => c[0] === "texImage3D").map((c) => c[2]);
  ok(layerUnits.includes("TEXTURE03") && layerUnits.includes("TEXTURE04"),
     "layers land on units 3 and 4 -- never on top of the grade's own");

  // The packing the shader reads. `top` is dir (0,1), offset 0, so
  // u = dot(uv, dir) + off is uv.y, which is region.Region.mask's "top".
  ok(String(uniforms["rA[0]"]) === "0,1,0,0.4", "a linear region packs to (dir, offset, extent)");
  ok(String(uniforms["rB[0]"]) === "0,0.4,0,0", "...with shape 0, its softness and no invert");
  ok(String(uniforms["rA[1]"]) === "0.5,0.5,0.6,0.75", "a radial region packs to (cx, cy, rx, ry)");
  ok(String(uniforms["rB[1]"]) === "1,0.7,0,0", "...with shape 1 and its softness");

  // Hold-to-compare must drop the layers too, or the "before" is half-graded.
  $("frame").dispatch("mousedown", { preventDefault() {} });
  ok(uniforms.nLayers === 0 && uniforms.useGrade === 0, "holding drops the layers as well");
  $("frame").dispatch("mouseup");
  ok(uniforms.nLayers === 2 && uniforms.useGrade === 1, "and they come back on release");

  // A layer cube that will not parse is the same failure as a base cube that
  // will not: the canvas must come down, not stay up missing a layer.
  forget();
  const badLayer = reset({ layers: [layer(REGION_TOP)], version: 21 });
  routes["/media/cube?layer=0&v=21"] = "LUT_3D_SIZE 2\n0 0 0\n";   // truncated
  render(badLayer);
  await settle();
  ok(liveOn === false && frameAsked(), "an unusable layer cube falls back to the server");

  // --- gate: a spatial effect is not in the cube, so the server renders -----
  forget();
  render(reset({ spec: { ...SPEC, effects: { ...EFFECTS, grain: 0.3 } }, version: 5 }));
  await settle();
  ok(liveOn === false, "grain sends the frame back to ffmpeg");
  // A mask that does not come from geometry (roadmap B2's segmentation model)
  // has nothing for regionMask to evaluate. Allow-list, so an unknown shape
  // falls back rather than being drawn as a `linear`.
  render(reset({ layers: [layer({ ...REGION_TOP, shape: "semantic" })], version: 51 }));
  await settle();
  ok(liveOn === false, "a shape the shader cannot compute goes back to the server");
  ok(frameAsked(), "and the server-rendered still appears");
  render(reset({ layers: [{ region: "sky" }], version: 52 }));
  await settle();
  ok(liveOn === false, "a region that is not even an object falls back too");
  // Six slots is every stack compile_stack can build; a look.json can say more.
  render(reset({ layers: Array(7).fill(layer(REGION_TOP)), version: 53 }));
  await settle();
  ok(liveOn === false, "more layers than the shader has slots falls back");
  render(reset({ layers: Array(6).fill(layer(REGION_TOP)), version: 54 }));
  await settle();
  ok(liveOn === true && uniforms.nLayers === 6, "and exactly six is still live");

  render(reset({ spec: { ...SPEC, effects: { ...EFFECTS, grain: 0.3 } }, version: 5 }));
  await settle();
  ok($("gl").hidden === true, "the canvas is hidden, not left stale");
  ok($("playBtn").hidden === true && $("clipBtn").hidden === true,
     "and its two controls go with it");
  ok(frameAsked(), "the server-rendered still is what appears");

  // --- fallback: the codec the browser cannot decode -------------------------
  render(reset({ version: 6 }));
  await settle();
  ok(liveOn === true, "back to live on a clean spec");
  $("imgGraded").src = "";
  $("vid").dispatch("error");
  ok(liveOn === false && $("gl").hidden === true, "a decode failure drops the canvas");
  ok(frameAsked(), "and the server-rendered image still appears");
  render(reset({ version: 7 }));
  await settle();
  ok(liveOn === false, "a clip the browser cannot decode stays on the still path");

  // --- fallback: no WebGL2 at all -------------------------------------------
  forget();
  glAvailable = false;
  render(reset({ version: 8 }));
  await settle();
  ok(liveOn === false && $("gl").hidden === true, "no WebGL2 means no canvas");
  ok($("playBtn").hidden === true, "and no play button to press");
  ok(frameAsked(), "the server-rendered image still appears");

  // --- fallback: WebGL2, but the shader will not compile --------------------
  forget();
  glAvailable = true; shadersCompile = false;
  render(reset({ version: 9 }));
  await settle();
  ok(liveOn === false && frameAsked(), "a shader that will not compile falls back too");

  // --- fallback: a cube the parser refuses ----------------------------------
  forget();
  shadersCompile = true;
  const broken = reset({ version: 10 });
  routes["/media/cube"] = "LUT_3D_SIZE 2\n0 0 0\n";     // truncated
  render(broken);
  await settle();
  ok(liveOn === false, "an unusable cube must not leave an ungraded canvas up");
  ok(frameAsked(), "it falls back to the server too");

  console.log(`page harness ok (${checked} checks)`);
  process.exit(0);
})();
