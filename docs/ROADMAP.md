# Roadmap — dumb UI, smart backend

The design constraint that decides everything below: **the interface stays
simple.** A person opens a clip, types a sentence, sees a frame. No node graph,
no timeline, no scopes panel, no window-drawing tools, no 40-slider inspector.
Everything "advanced" happens where they cannot see it — in decomposing what
they said into numbers.

That is not a smaller version of competing with Resolve. It is a different
competition, and it has one hard problem at its centre:

> An LLM authoring 43 numbers already strains a 20B model — which is exactly why
> `ragvid/looks.py` exists. Every expressive feature on the wish list (regions,
> curves, keyframes, per-region grades) multiplies that number. At ~200 numbers,
> asking a model to author them directly does not degrade, it collapses.

So the headline of this roadmap is not a feature. It is an architecture change
that makes every feature after it affordable.

## The architecture change: intent → compiler → spec

**Today:** `prompt → LLM → GradeSpec (43 floats) → LUT`. The model authors every
number, holds ~40 at identity, and moves only the right three.

**Proposed:** `prompt → LLM → Intent → compiler → GradeSpec → LUT`

The model stops emitting numbers and starts emitting *intent*: a short list of
typed operations, each with a target, a direction and a coarse magnitude.

```
"warm it up, darken the sky, keep her skin natural, half strength"

  →  [ {op: "temperature", dir: "warm", amount: "moderate"},
       {op: "exposure",    dir: "down", amount: "moderate", region: "sky"},
       {op: "protect",     target: "skin"},
       {op: "strength",    value: 0.5} ]

  →  compiler: measured ClipStats + closed-form solvers + region resolver
  →  GradeSpec
```

Why this is load-bearing:

- **The model's job stops growing.** New expressiveness adds compiler passes and
  verbs, not floats the model must author. Intent stays ~5 items no matter how
  capable the backend becomes.
- **Numbers come from measurement, not invention.** `ragvid/match.py` already
  proves the pattern: it solves slope/offset in closed form from `ClipStats`
  with no model in the loop. "Moderate warm" becomes a number by consulting the
  clip's measured colour, not by the model guessing 2000 Kelvin.
- **It is testable without an LLM.** Compiler passes are pure functions over
  `ClipStats`. Today most correctness claims need a mocked provider.
- **`GradeSpec` stays the currency.** `Intent` sits *upstream* of the spec; the
  spec is still the only thing session/history/undo/server/LUT ever see, so the
  core rule in `docs/ARCHITECTURE.md` survives.
- **The simple UI gets its one honest upgrade free.** Intent is human-readable
  by construction, so "what it did" can be shown as a few sentences with a
  strength slider each — a *simpler* control surface than the current 43-slider
  panel, not a more complex one.

## The layers, sorted by UI cost

Marks are against the code as of 2026-08-23: **[have] / [partial] / [none]**.
"Invisible" means the person types the same sentence and gets a better answer.

### Tier A — invisible, and the reason the tool is trustworthy

| # | Item | Status | UI cost | Note |
|---|---|---|---|---|
| A1 | Intent → compiler architecture | **[have]** | invisible (plus the C3 list) | `intent.py` + `compiler.py`. Measured against the direct path: 23/23 checks vs 21/23, 10/10 prompts vs 8/10, 2.8× cheaper. Default wherever the endpoint can constrain decoding to a schema. |
| A2 | Float pipeline end to end | **[have]** | invisible | `probe.py` samples raw `rgb48le`. Negligible on ordinary footage; on log through a conversion LUT `crushed_low` was over-reporting 2.0× and `p1` snapped to 1/255 — the two fields a shadow verb reads. |
| A3 | 10-bit output | **[have]** | invisible | Output matches the source. On an S-Log3 ramp: 635 distinct levels from a 10-bit source against 160 from 8, and 128 against 67 in the bottom third. |
| A4 | Built-in log transforms (S-Log3, V-Log, C-Log, Log-C, N-Log) | **[have]** | **removed** UI | `logspace.py` generates the conversion. All five land 18% grey on 0.408–0.412 against Rec.709's 0.409, from constant sets spanning 0.343–0.423. Two spans, three links and a file hunt became one `<select>`. |
| A5 | Richer measurement feeding the compiler | **[have]** | invisible | The compiler reads them all. `hue_strength` added: HSV saturation cannot fall when a frame holds two opposite hues, so it could not carry hue confidence. |
| A6 | Auto-balance before the creative grade | **[have]** | invisible (first row of the C3 list) | Green-cast and magenta-cast shots given the same "warmer" end up 1500× closer (0.0602 → 0.00004). A sodium street and a blue night are left bit-for-bit alone. |
| A7 | Scene-cut awareness | **[partial]** | invisible | `cuts` counts luma-histogram jumps — a whip pan measures 1.00 by frame difference and 0.22 by histogram. Measured only; nothing acts on it yet. |
| A8 | ASC CDL export + `look.json` sidecar | **[have]** | none — they just appear | Closed the data-loss bug: a `.cube` cannot carry `EffectSpec`. `exposure` folds exactly into CDL slope; what CDL cannot carry is named in its `<Description>`, diffed against identity so a new field joins the list automatically. |
| A9 | LUT precision | **[have]** | — | `lut.py` escalates 33³→65³ when hue qualifiers are on, with measured banding error behind the choice. |

### Tier B — the expressive jump, which only A1 makes affordable

Each is a compiler pass plus a verb or two. **None adds a control to the screen.**

| # | Item | Status | The sentence it unlocks |
|---|---|---|---|
| B1 | Regions — a grade that applies to part of the frame | **[none]** | "darken the top of the frame" |
| B2 | Semantic masks (sky / skin / faces / foliage) | **[none]** | "keep her skin natural", "make the sky moody" — highest-value item here |
| B3 | Curves (master, per-channel, hue-vs-hue, hue-vs-sat, lum-vs-sat) | **[none]** | "crush the blacks but keep the highlights soft" |
| B4 | Real HSL qualifier (arbitrary volume, matte finesse, despill) | **[partial]** | "kill the sodium streetlights" — six fixed bands cannot isolate one colour |
| B5 | Keyframes / change over time | **[none]** | "get darker as he walks into the tunnel" |
| B6 | Protect / exclude verbs | **[none]** | "…but don't touch skin" — needs B2; the most-repeated real note in grading |
| B7 | Relative edits against the measured clip | **[partial]** | "a stop brighter", "twice as warm as that" |

**B1 + B2 is the moat.** It is what Resolve users pay for (Magic Mask), and the
one case where a sentence is genuinely a *better* interface than a tracker and a
bezier — no UI beats "the sky" as a way to say the sky.

Two consequences to accept deliberately when B1 lands:

1. A per-region grade means `GradeSpec` can no longer be flat. Keep `GradeSpec`
   as the currency of *one correction* and add an outer container holding a list
   of (region, spec). Revise `docs/ARCHITECTURE.md` in the same commit rather
   than quietly contradicting it.
2. A region is spatial, so it **cannot bake into a `.cube`** — the same category
   as `EffectSpec` today. Export grows a per-region LUT + mask chain in
   `render.py`, and A8's `look.json` becomes the only lossless round-trip
   format. That is why A8 ships first.

### Tier C — small, visible, high feel-per-line

The only things allowed to touch the UI, one small element each.

| # | Item | Status | UI cost |
|---|---|---|---|
| C1 | WebGL preview instead of ffmpeg-per-still | **[none]** | none visually; scrubbing becomes instant |
| C2 | Real-time playback | **[none]** | a play button; follows from C1 |
| C3 | "What it did", as sentences | **[have]** | one list — replaced `#said` |
| C4 | Per-item strength sliders | **[have]** | 43-slider panel deleted; `index.html` net −47 lines |
| C5 | Clipping / exposure warning on the frame | **[none]** | one toggle; `ClipStats` already measures it |
| C6 | Before/after compare | **[have]** | hold-to-see-original |
| C7 | Undo / start over / history | **[have]** | — |

Explicitly **not** doing, per the simple-UI constraint: scopes panels, node
graph, timeline, media pool, window-drawing tools, curve editors, versions
manager, group grades, multi-user database.

### Tier D — deliberately out of scope

Camera raw decode (BRAW / R3D / ARRIRAW), full colour management (ACES / OCIO),
timeline conform (OTIO / EDL / AAF), HDR delivery (PQ / HLG / Dolby Vision),
mezzanine codecs, broadcast-safe limiting, tracking, collaborative databases.

Interop replaces most of these: A8 means a colourist takes ragvid's CDL and LUT
*into* Resolve and does the pipeline work there. The positioning is not "replace
Resolve" — it is **the fastest way from a sentence to a look, and it hands you a
CDL the rest of your pipeline already understands.**

## Build order

1. **A8** — CDL + `look.json` sidecar. Small, closes a real data-loss bug, and
   is a prerequisite for B1's export path.
2. **A3 + A4** — 10-bit output and built-in log transforms.
3. **A1** — intent → compiler, first as a standalone library with its own tests,
   then wired into `vibe.py` once it beats the direct path on the same prompts.
4. **A2, A5–A7** — measurement and correctness, all invisible, now feeding the
   compiler rather than the model.
5. **C1** — WebGL preview. The grade is already a 3D LUT, which a fragment
   shader samples directly, so previews stop round-tripping through ffmpeg.
6. **B1 + B2** — regions and semantic masks; the point at which the flat-spec
   rule is deliberately revised.
7. **B3–B7** — one compiler pass at a time, no new UI for any of them.

## The decision to make before B2

B2 needs a segmentation model looking at actual pixels, which contradicts the
standing rule *"pixels are never sent to a model"* (`spec.py` docstring,
`docs/ARCHITECTURE.md`). The rule's real intent is privacy and cost — a *local*
segmentation model honours both; a hosted vision API does not. Restate the rule
as **"no pixels leave the machine"** and run segmentation locally. Make that an
explicit documented decision, not a side effect of shipping a feature.

## How A1 gets proven

The whole roadmap rests on A1, so it does not ship on taste:

- **It must be more accurate.** Take ~10 real prompts, run them through the
  direct-spec path and the intent+compiler path against the same clip, and
  compare measured results against what the sentence asked for ("warmer" must
  move measured `dominant_hue` / `mean` in the right direction and nothing
  else). Keep the direct path until intent wins.
- **Token cost must go down, not up.** Measure real request size both ways.
  Intent is a smaller schema than 43 fields; if it is not measurably cheaper on
  Groq's free tier, the design is wrong.
