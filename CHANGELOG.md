# Changelog

What changed, described by what it does for you rather than by which files
moved. Versions follow [semantic versioning](https://semver.org): while ragvid
is on `0.x`, a middle-number bump can change the shape of a saved grade.

## 0.3.0 — which part of the picture you mean

A sentence could already say what to do and how much. It can now say **which
pixels** — by colour, by place, or by thing — **which end of the brightness
range**, and **which way a colour should lean**. None of that added a control to
the screen; the sentence carries all of it.

This is also the first release that is actually on PyPI. `0.2.0` was tagged and
built, but the upload never completed, so nothing was ever installable.


### Saying what you want

- **"Warm up the greens" now warms the greens.** It used to warm the whole
  picture — a white-balance move, aimed at everything, while the list of what it
  did said only "warmed it up". A colour named on "warmer" or "cooler" now turns
  that colour and leaves the rest of the frame untouched, down to the last
  digit: "push the greens toward yellow", "make the skin less orange". The turn
  cannot change how bright a colour is, which is not free — turning a colour
  around the axis that looks obvious would have shifted brightness by up to 42
  of the 256 levels a finished file is made of.
- **It costs nothing to say.** No new word, no new control: the sentence could
  already name a colour, and there is now somewhere for it to land.
- **"Drain the shadows but keep the sky rich" now works.** Colour used to be one
  knob for the whole picture: asking for less of it took the same amount out of
  the brightest part of the frame as out of the darkest. You can now say which
  end you mean — grey shadows with the highlights untouched, or rich highlights
  over drained shadows — and "a lot" still means exactly what it meant before,
  measured at the end you named rather than twice as far. What it cannot do is
  put the colour in the middle and take it off both ends; that shape is not
  expressible here, and asking for it will get you one end or the other.

- **"Drain the shadows but keep the highlights rich" no longer comes out flat.**
  Naming both ends of the picture in one sentence used to leave you with almost
  nothing: the two halves of the request cancelled each other and the result was
  weaker than either half on its own. They now pull together. This was found by
  asking a real model the sentence rather than by reading the code, and it took
  a second, quieter bug with it — "drain the shadows and make it punchier" was
  putting the punch into the shadows it had just drained.
- **A look you made on one clip can be carried to another.** Saving a grade
  writes down the sentence behind it, not only the numbers it produced, so
  opening a different clip and applying that look asks the same question of the
  new footage instead of pasting the old answer onto it. On a clip lit the
  opposite way, the sentence lands ten times closer to what it meant than the
  copied numbers do — those numbers overshot the warmth being asked for by more
  than five times. Looks saved by the previous version still open; they simply
  have no sentence behind them, and say so.

### Your work stays put

- **Opening a clip you have graded before brings its grades back.** Every clip
  now keeps its own folder, so opening a second clip cannot touch the first
  one's history, and restarting ragvid no longer loses what you had. Before
  this, every clip shared one folder and opening anything replaced whatever was
  in it — and there was no way to get an old session back at all, because the
  app never read one.
- **Grading the same clip twice from the terminal adds to its history** instead
  of replacing it.
- **A half-finished save can no longer damage a session.** The file is written
  beside the old one and swapped in only once it is complete, so a crash or a
  full disk leaves the last good version rather than a truncated one.
- **A damaged session says so plainly** instead of failing as an internal error.
- **A damaged or unrelated look file says so plainly** too, instead of failing
  as an internal error — including one written by some other tool that happens
  to end in `.json`, and one from a newer version of ragvid than you are
  running.

If you had a project open in a previous version, it lived in a shared folder
that is no longer read. Open the clip again and grade it; nothing else moved.

## 0.2.0 — packaged, but never uploaded

Until now ragvid existed only as a git checkout. This was meant to be the first
release on PyPI, and everything for it landed — the packaging, the wheel, the
release workflow, the tag. The upload itself never ran, so this version was
never installable from anywhere. `0.3.0` is the one that is.

Version `0.1.0` was never published. If something on your machine reports it,
that is an editable install from a checkout, not a release, and the code behind
it can be anything from the last three days of development.

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

### Looking back over what you tried

- **Clicking an old prompt no longer throws away the newer ones.** It used to
  jump straight back and delete everything after it, with no way to return. Now
  a click just *shows* you that step — what it did, and the settings it used —
  in the column beside the list. Going back to it is a button you press on
  purpose, and it keeps everything else.
- **One row per thing you asked for.** Nudging a slider or switching a move off
  still counts as a step you can undo, but those adjustments now fold into the
  prompt they belong to, instead of filling the list with "adjusted, adjusted,
  adjusted".
- **A delete button** — the only thing in that list that removes anything. It
  asks first, and it takes a prompt together with the adjustments you made to it.

### Fixed

- A broken-image icon sat in the corner of the preview whenever the live preview
  was running.

### How it decides

- **You describe, ragvid decides the numbers.** The AI service no longer invents
  colour values. It picks from a fixed list of 18 things a colourist does —
  warmer, lift the shadows, deepen the blues — with a direction and a coarse
  amount, and ragvid works out the actual magnitudes from what it measured off
  your footage. On eleven test sentences this got all eleven right where the old
  way got five, for 2.0× less money per prompt, judged by measuring the graded
  pixels rather than by reading the plan.
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
