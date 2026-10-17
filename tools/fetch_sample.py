"""
A larger text sample for training SAEs: WikiText-103 (Merity et al. 2016, CC BY-SA 3.0, Wikipedia's good and featured
articles), from Hugging Face (Salesforce/wikitext, wikitext-103-raw-v1, the training split's first part, ~150 MB).
Written as a JSON list of paragraphs, the format the app's neuron corpus uses, so Features can train on it.

  python tools/fetch_sample.py --out <cache>/datasets/wikitext-103.json [--chars 60000000]

Article titles and section headings are dropped, WikiText's spaced punctuation (" @-@ ", " , ") is joined back, and
paragraphs under 200 characters are left out. Progress lines: “PSI-PROGRESS {json}”; the last is PSI-DONE or PSI-ERROR.
Runs only when the user presses “Get a larger text sample” in Features (it downloads from the internet).
"""
import argparse, json, os, re, sys, time

URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet"


def say(kind, **o): print(f"PSI-{kind} {json.dumps(o)}", flush=True)


def clean(t):
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    t = re.sub(r" ([,.;:!?)\]'])", r"\1", t); t = re.sub(r"([(\[]) ", r"\1", t)
    return t.replace(" 's", "'s").replace(" n't", "n't").strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--chars", type=int, default=60_000_000)
    a = ap.parse_args(); t0 = time.time()
    import requests, pyarrow.parquet as pq, io
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".parquet.part"
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status(); total = int(r.headers.get("Content-Length") or 0); got = 0; last = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); got += len(chunk)
                if got - last > 8 << 20: last = got; say("PROGRESS", stage="download", done=got, of=total, note=f"{got / 2**20:.0f} of {total / 2**20:.0f} MB")
    say("PROGRESS", stage="convert", note="joining paragraphs")
    lines = pq.read_table(tmp, columns=["text"]).column("text").to_pylist(); os.remove(tmp)
    paras, n = [], 0
    for ln in lines:
        s = ln.strip()
        if not s or re.fullmatch(r"=+ .* =+", s): continue
        s = clean(s)
        if len(s) < 200: continue
        paras.append(s); n += len(s)
        if n >= a.chars: break
    json.dump(paras, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    say("DONE", out=os.path.abspath(a.out), paragraphs=len(paras), chars=n, approx_pieces=n // 4, seconds=round(time.time() - t0), source=URL, licence="CC BY-SA 3.0 (WikiText-103)")


if __name__ == "__main__":
    try: main()
    except Exception as e: print(f"PSI-ERROR {type(e).__name__}: {e}", flush=True); sys.exit(1)
