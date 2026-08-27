"""End-to-end pass through the running app, measured on the exported FILE.

Deliberately no model: every grade below is posted as an explicit Intent to
/api/intent, so the run is reproducible, spends no tokens, and tests the thing
that has never been tested -- the whole chain from HTTP through compile, bake,
mask, ffmpeg and back out to a file on disk.

The one check that matters is decoding the EXPORT and comparing it against the
server's own preview of the same frame. Every serious bug in this project's
history was a preview that disagreed with the file, and that comparison had
never been run on a full clip through the whole chain.

WHAT IT MEASURED, on test_files/test.mp4, in 8-bit code values:

    a targeted hue turn        mean 1.47  max 39   grade 0.08s  frame 1.93s  export 37.8s
    a tonal saturation split   mean 1.46  max 39   grade 0.08s  frame 1.04s  export 37.8s
    a geometric region         mean 1.44  max 44   grade 0.08s  frame 1.36s  export 59.4s
    a semantic mask            mean 1.45  max 32   grade 0.08s  frame 3.20s  export 53.8s
    a protect                  mean 1.24  max 33   grade 0.09s  frame 2.53s  export 60.9s
    a spatial effect           mean 2.76  max 38   grade 0.08s  frame 1.22s  export 50.3s

THOSE NUMBERS MEAN NOTHING WITHOUT THE CONTROL, because a constant that barely
moves between a hue rotation and a semantic mask is the signature of a floor
rather than of a grade that disagrees. The floor is the codec: an UNGRADED
preview against an ungraded H.264 re-encode at the same settings measures
mean 1.02, max 35.00. So the grade itself contributes under half a code value,
and the preview matches the export.

Grain is the exception at 2.76 and it is not a disagreement: grain is per-frame
noise, so the encoder mangles it and the two draws differ regardless.

The 3.20s frame is the segmentation model, once, as segment.py documents.

`bake_layers`' n^2 (server.py) did NOT need its cache: the deepest stack this
pass built was two layers. Measured rather than assumed, and nothing written.

THE FIRST VERSION OF THIS SCRIPT WAS WRONG AND REPORTED A NUMBER ANYWAY. It
asked for "?at=" where the route reads "?t=", compared the export's frame 8
against the preview's frame 0, and printed a mean of 19.4 code values as though
it meant something. The assert on shape is there because of it, and the lesson
is the file-level one: a harness that produces a plausible number is not a
harness that measured anything.
"""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
REPO = Path(__file__).resolve().parent.parent
SP = REPO / "out" / "pass"
SP.mkdir(parents=True, exist_ok=True)
PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"


def call(path, body=None, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch(path, out, timeout=600):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        Path(out).write_bytes(r.read())
    return out


def decode(path, at=None):
    """A frame (or a whole PNG) as float 0..1, via ffmpeg to raw rgb24."""
    args = ["ffmpeg", "-v", "error"]
    if at is not None:
        args += ["-ss", f"{at:.3f}"]
    args += ["-i", str(path)]
    if at is not None:
        args += ["-frames:v", "1"]
    args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(args, capture_output=True).stdout
    meta = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height", "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(meta[0]), int(meta[1])
    return np.frombuffer(raw[: w * h * 3], np.uint8).reshape(h, w, 3).astype(np.float64) / 255.0


GRADES = [
    ("a targeted hue turn", [dict(op="warmth", dir="up", amount="strong", target="red")]),
    ("a tonal saturation split", [dict(op="shadow_saturation", dir="down", amount="strong"),
                                  dict(op="highlight_saturation", dir="up", amount="moderate")]),
    ("a geometric region", [dict(op="exposure", dir="down", amount="strong", target="top")]),
    ("a semantic mask", [dict(op="exposure", dir="down", amount="strong", target="sky")]),
    ("a protect", [dict(op="exposure", dir="down", amount="strong"),
                   dict(op="protect", dir="up", target="person")]),
    ("a spatial effect", [dict(op="vignette", dir="up", amount="strong"),
                          dict(op="grain", dir="up", amount="moderate")]),
]


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "test_files" / "test.mp4")
    root = SP / "root"
    root.mkdir(exist_ok=True)
    env = dict(os.environ, XDG_DATA_HOME=str(root / "data"))
    # Isolated, but not so isolated that the segmentation model has to be
    # downloaded again: link the one already on this machine into the scratch
    # data dir. Without this a semantic mask answers 428 SegmentUnavailable and
    # the leg that exercises it silently does not run.
    from ragvid.platform import data_dir as real_data_dir
    real_model = real_data_dir() / "models" / "segformer-b0-ade-512.onnx"
    fake_models = root / "data" / "ragvid" / "models"
    fake_models.mkdir(parents=True, exist_ok=True)
    link = fake_models / real_model.name
    if real_model.exists() and not link.exists():
        link.symlink_to(real_model)
    proc = subprocess.Popen(
        ["uv", "run", "ragvid", "serve", "--no-browser", "--port", str(PORT), "--root", str(root)],
        cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(80):
            try:
                call("/api/state"); break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("the server never answered")
        print(f"server up on 127.0.0.1:{PORT}, isolated root {root}")

        t0 = time.time()
        call("/api/project", {"path": clip})
        print(f"opened {Path(clip).name} in {time.time()-t0:.1f}s")

        rows = []
        for label, ops in GRADES:
            call("/api/reset") if False else None
            t0 = time.time()
            state = call("/api/intent", {"intent": {"ops": ops, "strength": "full"}})
            t_grade = time.time() - t0

            # the server's own preview of one frame -- what the person is looking at
            t0 = time.time()
            prev = fetch("/media/frame?t=8.0", SP / "preview.png")
            t_frame = time.time() - t0

            t0 = time.time()
            dest = SP / f"export_{len(rows)}.mp4"
            job = call("/api/export", {"out": str(dest), "gpu": False})["job"]
            out = None
            for _ in range(1200):
                st = call(f"/api/export/{job}")
                if st["state"] == "done" or st.get("path"):
                    out = st["path"]; break
                if st.get("error"):
                    raise SystemExit(f"export failed: {st['error']}")
                time.sleep(0.5)
            t_export = time.time() - t0
            if not out:
                print(f"  {label:26s} export never finished")
                continue

            a = decode(prev)
            b = decode(out, at=8.0)
            # A silent crop would compare the wrong pixels and report a number
            # anyway. The first run of this script did exactly that -- it asked
            # for "?at=" where the route reads "?t=", got frame 0 against the
            # export's frame 8, and reported a mean error of 19.4 code values as
            # though it meant something.
            assert a.shape == b.shape, f"preview {a.shape} vs export {b.shape}"
            err = np.abs(a - b)
            rows.append((label, err.mean() * 255, err.max() * 255, t_grade, t_frame, t_export,
                         len(state.get("layers", []) or []), Path(out).stat().st_size / 1e6))
            print(f"  {label:26s} preview vs EXPORT: mean {err.mean()*255:5.2f}  "
                  f"max {err.max()*255:6.2f} code values   "
                  f"grade {t_grade:5.2f}s  frame {t_frame:5.2f}s  export {t_export:6.2f}s")
            call("/api/undo", {})

        print("\nsummary, in 8-bit code values:")
        for r in rows:
            print(f"  {r[0]:26s} mean {r[1]:5.2f}  max {r[2]:6.2f}   export {r[7]:5.1f} MB")
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)
        print("server stopped")


main()
