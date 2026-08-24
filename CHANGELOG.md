# Changelog

What changed, described by what it does for you rather than by which files
moved. Versions follow [semantic versioning](https://semver.org): while ragvid
is on `0.x`, a middle-number bump can change the shape of a saved grade.

## Unreleased

### Saying what you want

- **"Crush the blacks but keep the highlights soft" now works.** It used to be
  two requests ragvid could only half-hear. Deepening the blacks meant sliding
  the dark part of the picture downwards until some of it hit pure black and
  stayed there — detail gone, and gone for good once the look is rendered. It
  now bends the bottom of the picture instead of sliding it, so the darkest
  tones get closer together without landing on top of each other. Counted in
  the 0–255 levels a finished file is made of, the old way flattened levels
  1, 2, 3 and 4 all onto 0; the new one leaves every one of them distinct, and
  reaches a deeper black doing it. Say "really crush them" and it will still
  spend the black point, because that is what those words mean.
- **You can ask for the highlights' shape, not just their brightness.** "Keep
  the highlights soft", "don't let it clip", "let the whites go hard" —
  previously there was no way to say any of it, and those words moved nothing
  at all. Now they bend the top end of the picture without making it brighter
  or darker.
- **Deepening the blacks no longer brightens the whites behind your back.**
  Contrast was one knob doing both ends at once, so asking for one gave you the
  other. On a test ramp, the old way pulled a black at 0.08 down to 0.054 but
  dragged a white at 0.95 up to 0.969; it now reaches 0.033 and leaves the
  white at 0.952.

A grade is now 44 numbers rather than 43. Old projects, saved looks and
`look.json` files all still open — the new one reads as "leave it alone".

## 0.2.0 — the first one you can install

Until now ragvid existed only as a git checkout. This is the first release on
PyPI: `uv tool install ragvid`, then `ragvid serve`.

Version `0.1.0` was never published. If something on your machine reports it,
that is an editable install from a checkout, not a release, and the code behind
it can be anything from the last three days of development.

### Saying what you want

- **You describe, ragvid decides the numbers.** The AI service no longer invents
  colour values. It picks from a fixed list of 16 things a colourist does —
  warmer, lift the shadows, deepen the blues — with a direction and a coarse
  amount, and ragvid works out the actual magnitudes from what it measured off
  your footage. On the same ten prompts this got 10 right where the old way got
  8, for 2.8× less money per prompt, judged by measuring the graded pixels
  rather than by reading the plan.
- **A sentence can say which part of the picture.** By colour ("deepen the
  blues, leave skin alone"), by place ("darken the top of the frame") or by
  thing ("make the sky moody"). Ten colour families, six places, and five
  things — sky, foliage, person, water, buildings.
- **Naming a thing needs `ragvid[masks]`.** About 13 MB of machinery plus a
  15 MB model that downloads once, on first use, after asking. It runs on your
  own computer; no frame is sent anywhere. Without it, naming a thing tells you
  what to install instead of ignoring the word.
- **A second sentence no longer destroys the first.** "Less blue" now edits the
  list of things ragvid decided to do, so the regions and effects you asked for
  earlier survive the next adjustment.
- **It balances the clip first.** Exposure and white balance are corrected
  against the measured footage before your look goes on top, so "moody" means
  the same thing on an underexposed clip as on a good one.

### Your footage

- **Shot in log?** Pick your camera from a list — S-Log3, V-Log, C-Log3, LogC3,
  N-Log — and ragvid converts before measuring and before grading, instead of
  guessing at un-flattening a flat grey picture. No hunting for a LUT file.
- **10-bit stays 10-bit.** Exports keep the source's bit depth where the encoder
  can carry it, so a 10-bit clip no longer comes back banded.
- **Measurement in 16 bits.** Measuring 8-bit was lying about exactly the
  numbers a shadow adjustment reads: crushed blacks were over-reported by 2×.

### Seeing it

- **The preview is now drawn in your browser**, by the graphics card, from the
  same LUT the export uses — so it keeps up while you type. When a look includes
  something a LUT cannot do (grain, glow, vignette, a soft edge), the preview
  declines the shortcut and shows the real rendered frame rather than a
  confident approximation of the wrong thing.
- **It tells you what it did**, in sentences, each with its own strength — and
  those sentences are the same list the adjustment step edits.
- **The clip plays** in the browser, graded.

### Getting the look out

- **Every export leaves a `.cdl` and a `look.json` beside it.** The CDL is the
  interchange format the rest of a post pipeline already reads; `look.json` is
  the complete look, including the spatial effects a `.cube` cannot carry.
- **Hardware encoding** where the machine has it — NVENC, Quick Sync, VAAPI,
  AMF, VideoToolbox — with a real trial encode deciding, not a guess. 10-bit
  H.264 stays on the software encoder, because none of the hardware ones can do
  it.

### Keys and services

- **Keys are entered in the app and nowhere else.** There is no command-line
  flag for a key, deliberately: arguments are visible in your shell history and
  to other users on the machine. The key is stored in a file only you can read
  and never shown again beyond its last four characters.
- **Groq, Ollama, OpenAI, Anthropic, xAI, Mistral, DeepSeek, Moonshot,
  OpenRouter, Together**, and anything OpenAI-compatible. Services that cannot
  guarantee a complete answer take an older, simpler route rather than being
  coaxed into one and quietly getting it wrong.

### Matching a photo

Hand it a reference image instead of a description and it matches your footage
to that photo's colours. No account, no service, no network — it is closed-form
arithmetic on your own machine.

---

Not yet: curves ("crush the blacks but keep the highlights soft"), a real HSL
qualifier, keyframes, and protect/exclude verbs. One clip gets one look; if
your video spans a cut, split it first.
