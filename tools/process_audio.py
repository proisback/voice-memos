"""Turn new Speechify recordings into the site's memo audio and transcript timings.

Usage (run from anywhere; paths are absolute or relative to this repo):
  python tools/process_audio.py --dir  "C:/path/to/folder"      # nine files: memo-01.mp3 ... memo-09.mp3
  python tools/process_audio.py --single "C:/path/to/all.mp3"   # one file with all nine memos, in order

Each memo must be read as: product name, "The problem.", "What I did.", "What happened.",
and optionally "What stayed with me." (see docs/copy/voice-memos-speechify.md in my-portfolio).

What it does:
  1. Transcribes with faster-whisper (word timestamps).
  2. Finds the spoken section labels in each memo. Stops with an error if any memo is missing a label.
  3. Writes audio/memo-NN.mp3 (mono, 64 kbps), cutting the single file when needed.
  4. Rewrites DUR and TAPES in index.html: per memo, [words start, section ends] for each spoken section.
  5. Prints each memo's transcript so numbers can be checked by ear and eye.
Requires: ffmpeg on PATH, `pip install faster-whisper`.
"""
import argparse, json, os, re, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = [["the", "problem"], ["what", "i", "did"], ["what", "happened"], ["what", "stayed", "with", "me"]]


def pcm(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(model, path):
    segs, _ = model.transcribe(pcm(path), word_timestamps=True, beam_size=5)
    return [{"w": w.word.strip(), "s": float(w.start), "e": float(w.end)} for s in segs for w in s.words]


norm = lambda w: re.sub(r"[^a-z]", "", w.lower())
norm_keep = lambda w: re.sub(r"[^a-z0-9-]", "", w.lower())


def find_labels(words, start=0, stop=None, strict=True):
    """Indices where each label sequence starts, in order, within words[start:stop]. Missing labels are None.
    strict: the label must start a sentence (so "...to address the problem." is not a label). Inside a memo
    already bounded by its spoken heading, strict is off, because Speechify sometimes runs a label into the
    previous phrase ("Signal, the problem.")."""
    stop = len(words) if stop is None else stop
    found, i = [], start
    for lab in LABELS:
        hit = None
        for j in range(i, stop - len(lab) + 1):
            ok = not strict or j == 0 or re.search(r"[.?!]$", words[j - 1]["w"])
            if ok and [norm(words[j + k]["w"]) for k in range(len(lab))] == lab:
                hit = j
                break
        found.append(hit)
        if hit is not None:
            i = hit + len(lab)
    return found


def sections(words, idx, t0, end):
    """[words start, section ends] relative to t0, for each label found."""
    present = [(k, j) for k, j in enumerate(idx) if j is not None]
    out = []
    for n, (k, j) in enumerate(present):
        body = j + len(LABELS[k])
        stop = words[present[n + 1][1]]["s"] if n + 1 < len(present) else end
        out.append([round(words[body]["s"] - t0, 2), round(stop - t0, 2)])
    return out


def duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True, check=True).stdout)


def encode(src, dst, ss=None, t=None):
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.2f}", "-t", f"{t:.2f}"]
    cmd += ["-i", src]
    if t is not None:
        cmd += ["-af", f"afade=t=in:d=0.03,afade=t=out:st={max(0, t - 0.25):.2f}:d=0.25"]
    cmd += ["-ac", "1", "-b:a", "64k", "-f", "mp3", dst]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir")
    g.add_argument("--single")
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    out_dir = os.path.join(ROOT, "audio")
    tapes, problems, texts = [], [], []

    if a.dir:
        for n in range(1, 10):
            src = os.path.join(a.dir, f"memo-{n:02d}.mp3")
            if not os.path.exists(src):
                sys.exit(f"Missing {src}")
            dst = os.path.join(out_dir, f"memo-{n:02d}.mp3.new")
            encode(src, dst)
            words = transcribe(model, dst)
            idx = find_labels(words)
            if None in idx[:3]:
                problems.append(f"memo-{n:02d}: could not find labels {[' '.join(LABELS[k]) for k, j in enumerate(idx) if j is None]}")
            tapes.append(sections(words, idx, 0, words[-1]["e"]))
            texts.append(" ".join(w["w"] for w in words))
    else:
        dst_all = a.single
        words = transcribe(model, dst_all)
        total = duration(dst_all)
        is_head = lambda w: re.search(r"memo|mp3", w, re.I)
        # Preferred: Speechify read the "memo-NN.mp3" headings aloud, which mark each memo exactly.
        num = lambda w: (m := re.match(r"^memos?-?0?([1-9])$", norm_keep(w))) and int(m.group(1))
        heads, pos = [], 0
        for n in range(1, 10):
            hit = next((j for j in range(pos, len(words)) if num(words[j]["w"]) == n), None)
            if hit is None:
                heads = None
                break
            heads.append(hit)
            pos = hit + 1
        if heads:
            # The instructions mention "memo01 ... memo09" before the scripts; use the last memo01 before memo02.
            heads[0] = max(j for j in range(0, heads[1]) if num(words[j]["w"]) == 1)
            bounds = []
            for n in range(9):
                lo = heads[n]
                while lo < len(words) and is_head(words[lo]["w"]):
                    lo += 1
                hi = heads[n + 1] if n < 8 else len(words)
                bounds.append((lo, hi))
            memos = [find_labels(words, lo, hi, strict=False) for lo, hi in bounds]
            starts = [lo for lo, _ in bounds]
            lasts = [hi - 1 for _, hi in bounds]
        else:
            # Fallback: no headings. Find labels as sentence starts, and start each memo at the sentence before "The problem".
            memos, pos = [], 0
            for n in range(9):
                idx = find_labels(words, pos)
                if idx[0] is None:
                    sys.exit(f"Could not find 'The problem' for memo {n + 1}")
                memos.append(idx)
                pos = max(j for j in idx if j is not None) + 1
            acronym = re.compile(r"^([A-Za-z]\.)+$")
            starts = []
            for n, idx in enumerate(memos):
                p = idx[0]
                lo = max(j for j in memos[n - 1] if j is not None) + 1 if n else 0
                start = lo
                for j in range(p - 2, lo - 1, -1):  # also skips any title or instructions read before memo 01
                    if re.search(r"[.?!]$", words[j]["w"]) and not acronym.match(words[j]["w"]):
                        start = j + 1
                        break
                starts.append(start)
            lasts = [(starts[n + 1] - 1) if n < 8 else len(words) - 1 for n in range(9)]
        for n, idx in enumerate(memos):
            # Start in the pause before the product name: Whisper's word starts can run a little late,
            # so back off up to 0.45 s, but never into the previous word (a spoken heading or memo).
            prev_end = words[starts[n] - 1]["e"] + 0.05 if starts[n] > 0 else 0
            t0 = max(0, prev_end, words[starts[n]]["s"] - 0.45)
            last = lasts[n]
            while last > starts[n] and is_head(words[last]["w"]):  # a file heading read aloud
                last -= 1
            end = min(total, words[last]["e"] + 0.45)
            if None in idx[:3]:
                problems.append(f"memo-{n + 1:02d}: could not find labels {[' '.join(LABELS[k]) for k, j in enumerate(idx) if j is None]}")
                continue
            tapes.append(sections(words, idx, t0, words[last]["e"]))
            encode(dst_all, os.path.join(out_dir, f"memo-{n + 1:02d}.mp3.new"), t0, end - t0)
            texts.append(" ".join(w["w"] for w in words[starts[n]:last + 1]))

    if problems:
        for f in os.listdir(out_dir):
            if f.endswith(".new"):
                os.remove(os.path.join(out_dir, f))
        sys.exit("Stopped, nothing replaced:\n  " + "\n  ".join(problems))

    for n in range(1, 10):
        new = os.path.join(out_dir, f"memo-{n:02d}.mp3.new")
        os.replace(new, os.path.join(out_dir, f"memo-{n:02d}.mp3"))
    durs = [round(duration(os.path.join(out_dir, f"memo-{n:02d}.mp3")), 2) for n in range(1, 10)]

    page = os.path.join(ROOT, "index.html")
    html = open(page, encoding="utf-8").read()
    html = re.sub(r"let DUR = \[[^\]]*\];", "let DUR = " + json.dumps(durs) + ";", html, count=1)
    html = re.sub(r"const TAPES = \[.*?\];\n", "const TAPES = " + json.dumps(tapes, separators=(",", ":")) + ";\n", html, count=1, flags=re.S)
    open(page, "w", encoding="utf-8").write(html)

    for n in range(9):
        print(f"\nmemo-{n + 1:02d}  {durs[n]}s  sections: {len(tapes[n])}  {tapes[n]}")
        print("  heard:", texts[n])
    print("\nDone. index.html updated; review the transcripts above before publishing.")


if __name__ == "__main__":
    main()
