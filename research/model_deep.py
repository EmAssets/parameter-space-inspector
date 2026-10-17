"""
Level 2 of the storage ladder: per-layer profiles and run metadata, for a model and (with --init) its untrained twin.

The ladder (model_ladder.py) gives one verdict per level. This adds the detail that explains a verdict:
  provenance     versions, device, precision, parameter count, peak memory, seconds per section, code commit
  tokens         how each language's test sentences are cut into tokens (cross-language results depend on it)
  weights        per layer and matrix: spectral norm, stable rank, heavy-tail alpha (no forward pass needed). Two alphas:
                 `alpha` = Hill on the top 10% of W^T W's eigenvalues (a fixed cut: comparable only within the same matrix
                 shape and precision); `alpha_ks` = power-law MLE with xmin chosen by the smallest KS distance
                 (Clauset, Shalizi & Newman 2009), with xmin_ks, ks_D and n_tail. gpt-oss: 8 experts per layer sampled.
  residual       per layer: norm, anisotropy, effective dimension, massive activations and where they sit
  senses         the sense confusion matrix at the best layer: which meanings the model keeps apart, which it merges
  lens           per layer: where the answer to each fact first becomes the top guess (logit lens)
  heads          per layer: attention heads acting as sinks, previous-token heads and induction heads (random tokens
                 drawn with --seed). Attention patterns only, unless the OV copy check runs (dense float models with
                 separable v/o weights): per head, the eigenvalue copy score of W_U W_OV E over 500 sampled tokens
                 (E = effective embedding, norms ignored; Elhage et al. 2021); induction heads that also copy (score >= 0.5)
                 are counted as 'attention+OV'.
  gram           per layer: the sentences' centred Gram matrix (in <tag>.npz), for layer-by-layer comparison of models
  routing        mixture-of-experts only: expert use per layer, and how much the routed expert tells the sense

  python research/model_deep.py --model <folder> --tag <name> [--init] [--q8] [--cpu] [--seed N]
Writes results/model_deep/<tag>.json (and the untrained twin's as <tag>_init.json).
"""
import argparse, json, os, platform, subprocess, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model_ladder as L
import model_ladder_data as D

OUT = os.environ.get("PSI_DEEP_DIR") or os.path.join(HERE, "results", "model_deep"); os.makedirs(OUT, exist_ok=True)
r3 = lambda x: round(float(x), 3)


def provenance(M, path, init, q8):
    import transformers
    try: commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except Exception: commit = None
    n_params = sum(p.numel() for p in M.m.parameters())
    if M.moe: n_params += sum(ex.E * ex.I * ex.H * 3 for ex in M.experts)            # MXFP4 experts are buffers, not parameters
    return {"path": path, "untrained_twin": init, "precision": "MXFP4 experts + int8 attention" if M.moe else "int8 (LLM.int8)" if q8 else str(M.m.dtype).replace("torch.", ""),
            "params": int(n_params), "layers": M.L, "hidden": int(M.WU.shape[1]), "vocab": int(M.WU.shape[0]), "units_per_layer": int(M.I),
            "model_type": getattr(M.cfg, "model_type", None), "tied_embeddings": bool(M.WU.data_ptr() == M.WE.data_ptr()),
            "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0) if L.DEV == "cuda" else platform.processor() or "cpu", "code_commit": commit,
            "seed": L.SEED}


def tokens(M):
    """How the test sentences are cut: tokens per sentence, tokens for the marked word, and how often it is one token."""
    def stats(sents):
        n_sent, n_span, one = [], [], 0
        for s in sents:
            text, (a, b) = L.unmark(s); enc = M.tok(text, return_offsets_mapping=True)
            k = [i for i, (x, y) in enumerate(enc["offset_mapping"]) if y > x and x <= b and y > a]
            n_sent.append(len(enc["input_ids"])); n_span.append(len(k)); one += len(k) == 1
        return {"tokens_per_sentence": r3(np.mean(n_sent)), "tokens_per_marked_word": r3(np.mean(n_span)), "marked_word_one_token": r3(one / len(sents)), "n": len(sents)}
    ids = lambda ss: [set(M.tok(L.unmark(s)[0])["input_ids"]) for s in ss]
    E = set().union(*ids([s for k in L.SENSE_NAMES for s in D.SENSES[k]]))
    out = {"en": stats([s for k in L.SENSE_NAMES for s in D.SENSES[k]])}
    for lang, T in D.TRANS.items():
        xs = [s for k in L.SENSE_NAMES for s in T[k]]
        # Share of a translated sentence's tokens that also occur in the English sentences: a random network can match these.
        out[lang] = stats(xs) | {"tokens_shared_with_english": r3(np.mean([len(x & E) / len(x) for x in ids(xs)]))}
    out["foreign_red_tokens"] = {w: len(M.tok(" " + w, add_special_tokens=False)["input_ids"]) for w in D.FOREIGN}
    return out


def dense_w(mod):
    if isinstance(mod, L.Int8Linear): return mod.weight.float() * (mod.weight.SCB.float()[:, None] / 127)
    return mod.weight.float()


ALPHA_NOTE = ("alpha: Hill on the top 10% of eigenvalues, a fixed cut: comparable only within the same shape and precision; "
              "alpha_ks: xmin chosen by KS distance (Clauset, Shalizi & Newman 2009)")


def hill_ks(lam, max_cand=200, min_tail=10):
    """Power-law fit to eigenvalues lam (any order): for each candidate xmin (up to max_cand log-spaced ranks among the top
    half), the continuous MLE alpha = 1 + n / sum(ln(x / xmin)) on the tail x >= xmin, and the KS distance between the
    tail's empirical CDF and the fitted one; the xmin with the smallest distance wins (Clauset, Shalizi & Newman 2009)."""
    none = {"alpha_ks": None, "xmin_ks": None, "ks_D": None, "n_tail": None}
    x = np.sort(np.asarray(lam, dtype=np.float64))[::-1]; x = x[x > 0]
    top = len(x) // 2
    if top < min_tail: return none
    ranks = np.unique(np.round(np.geomspace(min_tail, top, min(max_cand, top - min_tail + 1))).astype(int))
    best = None; lx = np.log(x)
    for n in ranks:                                                              # tail = the n largest; xmin = the n-th largest
        xmin = x[n - 1]; sl = float((lx[:n] - np.log(xmin)).sum())
        if sl <= 0: continue
        a = 1 + n / sl; t = x[:n][::-1]; P = 1 - (t / xmin) ** (1 - a); i = np.arange(1, n + 1)
        Dk = float(max(np.abs(i / n - P).max(), np.abs((i - 1) / n - P).max()))
        if best is None or Dk < best[0]: best = (Dk, a, xmin, n)
    if best is None: return none
    return {"alpha_ks": r3(best[1]), "xmin_ks": float(f"{best[2]:.4g}"), "ks_D": r3(best[0]), "n_tail": int(best[3])}


@torch.no_grad()
def spectrum(W):
    s = torch.linalg.svdvals(W.to(L.DEV).float()); lam = (s ** 2).double()
    k = max(10, len(lam) // 10); tail = lam[:k]                                   # Hill estimator on the top 10% eigenvalues of W^T W
    alpha = 1 + k / float(torch.log(tail / tail[-1]).sum().clamp_min(1e-9))
    return {"spectral_norm": r3(s[0]), "stable_rank": r3(lam.sum() / lam[0]), "alpha": r3(alpha)} | hill_ks(lam.cpu().numpy())


@torch.no_grad()
def weights(M, experts_per_layer=8):
    """Per layer: every 2-D projection (attention and MLP); for experts, the first few experts' matrices."""
    out = []
    for i, ly in enumerate(M.layers):
        row = {}
        for name, mod in ly.named_modules():
            if isinstance(mod, (torch.nn.Linear, L.Int8Linear)) and min(mod.weight.shape) >= 16:
                row[name] = spectrum(dense_w(mod))
            elif type(mod).__name__ == "Conv1D":                                     # GPT-2 style
                row[name] = spectrum(mod.weight.T)
        if M.moe:
            ex = M.experts[i]
            for proj in ("gate_up_proj", "down_proj"):
                sp = [spectrum(ex.w(e, proj)) for e in range(min(experts_per_layer, ex.E))]
                row[f"experts.{proj}"] = {k: (r3(np.mean([x[k] for x in sp if x[k] is not None])) if any(x[k] is not None for x in sp) else None)
                                          for k in sp[0]} | {"experts_sampled": len(sp)}
        out.append(row)
    return out


@torch.no_grad()
def residual(R):
    """R [N, L+1, D]: one marked token per sentence at every layer."""
    out = []
    for l in range(R.shape[1]):
        X = R[:, l].to(L.DEV).float(); Xn = torch.nn.functional.normalize(X, dim=-1); n = X.shape[0]
        cos = (Xn @ Xn.T); aniso = (cos.sum() - n) / (n * (n - 1))
        ev = torch.linalg.eigvalsh(torch.cov(X.T) if X.shape[1] <= 4096 else (X - X.mean(0)) @ (X - X.mean(0)).T / (n - 1)).clamp_min(0)
        a = X.abs(); top = a.max(1)
        dims, cnt = torch.unique(top.indices, return_counts=True)
        Z = (X - X.mean(0)) / (X.std(0) + 1e-6)                                     # every dimension weighted alike
        ez = torch.linalg.eigvalsh(Z @ Z.T / (n - 1)).clamp_min(0)
        out.append({"norm": r3(X.norm(dim=-1).mean()), "anisotropy": r3(aniso), "effective_dim": r3(ev.sum() ** 2 / (ev ** 2).sum()),
                    "effective_dim_standardised": r3(ez.sum() ** 2 / (ez ** 2).sum()),
                    "massive_ratio": r3((top.values / a.median(1).values.clamp_min(1e-6)).median()), "massive_dim": int(dims[cnt.argmax()]),
                    "massive_dim_share": r3(cnt.max() / n)})
    return out


@torch.no_grad()
def senses(R, y, layer):
    """Leave-one-out nearest-centroid at one layer: the sense confusion matrix (rows: true sense)."""
    X = R[:, layer].to(L.DEV).float(); y = torch.tensor(y, device=L.DEV); n_s = len(L.SENSE_NAMES)
    C = np.zeros((n_s, n_s), dtype=int)
    for i in range(len(y)):
        m = torch.ones(len(y), dtype=torch.bool, device=L.DEV); m[i] = False
        p = int(L.centroid_acc(X[m], y[m], X[i:i + 1], y[i:i + 1])[0]); C[int(y[i]), p] += 1
    acc = C.diagonal() / C.sum(1)
    pairs = sorted(((int(C[a, b] + C[b, a]), L.SENSE_NAMES[a], L.SENSE_NAMES[b]) for a in range(n_s) for b in range(a + 1, n_s)), reverse=True)
    return {"layer": layer, "per_sense_acc": {k: r3(v) for k, v in zip(L.SENSE_NAMES, acc)}, "confusion": C.tolist(),
            "most_confused": [{"senses": [a, b], "count": c} for c, a, b in pairs[:5] if c > 0]}


@torch.no_grad()
def lens(M):
    """Logit lens on the fact prompts: per layer, the answer's rank and the guess entropy; first layer it leads."""
    facts = [(p, "red") for p in D.FACTS_RED] + list(D.FACTS_OTHER)
    _, R = M.batch_last([p for p, _ in facts])
    ans = torch.tensor([M.tok(" " + a, add_special_tokens=False)["input_ids"][0] for _, a in facts], device=L.DEV)
    # Forced choice, as M6 asks it: the right answer against the other answers in the set (the colour words). Top-1
    # over the whole vocabulary is rarely the bare answer ("is bright red"), so it alone almost never reaches half.
    cand = torch.unique(ans); col = {int(c): i for i, c in enumerate(cand.tolist())}; ai = torch.tensor([col[int(a)] for a in ans], device=L.DEV)
    per = []
    for l in range(R.shape[1]):
        lg = torch.log_softmax(M.final_logits(R[:, l].to(L.DEV)).float(), -1)
        rank = (lg > lg.gather(1, ans[:, None])).sum(1)
        fc = (lg[:, cand].argmax(1) == ai).float().mean()
        per.append({"median_rank": int(rank.median()), "top1": r3((rank == 0).float().mean()), "forced_choice": r3(fc), "entropy": r3(-(lg.exp() * lg).sum(-1).mean())})
    first = next((l for l, x in enumerate(per) if x["top1"] >= 0.5), None)
    chance = 1 / len(cand)
    first_fc = next((l for l in range(len(per)) if all(x["forced_choice"] >= 0.5 for x in per[l:])), None)   # from here on, right among the candidates for half the facts
    return {"n_facts": len(facts), "n_candidates": len(cand), "forced_choice_chance": r3(chance), "per_layer": per,
            "first_layer_top1_half": first, "first_layer_forced_choice_half": first_fc}


@torch.no_grad()
def heads(M, T=48, reps=4, seed=0):
    """Attention-head roles on random tokens repeated twice: attention to the first token (sink), to the previous
    token, and to the token after the earlier copy of the current one (induction). Needs eager attention."""
    try: M.m.set_attn_implementation("eager")
    except Exception as e: return {"skipped": f"eager attention not available: {e}"}
    g = torch.Generator().manual_seed(seed); V = min(M.tok.vocab_size, int(M.WU.shape[0]))
    bos = [M.tok.bos_token_id] if M.tok.bos_token_id is not None else []
    agg = None
    for _ in range(reps):
        r = torch.randint(1000, V - 1000, (T,), generator=g).tolist(); ids = torch.tensor([bos + r + r], device=L.DEV); o = len(bos)
        att = M.m(input_ids=ids, output_attentions=True).attentions                    # L x [1, H, S, S]
        q = torch.arange(o + T + 1, o + 2 * T, device=L.DEV)                           # second copy, from its 2nd token
        rows = []
        for A in att:
            A = A[0].float()
            rows.append(torch.stack([A[:, q, 0].mean(-1), A[:, q, q - 1].mean(-1), A[:, q, q - T + 1].mean(-1),
                                     -(A[:, q].clamp_min(1e-9).log() * A[:, q]).sum(-1).mean(-1)], -1))   # [H, 4]
        S = torch.stack(rows)                                                          # [L, H, 4]
        agg = S if agg is None else agg + S
    try: M.m.set_attn_implementation("sdpa")
    except Exception: pass
    S = (agg / reps).cpu().numpy()
    per = [{"sink_heads": int((h[:, 0] > 0.5).sum()), "previous_token_heads": int((h[:, 1] > 0.4).sum()), "induction_heads": int((h[:, 2] > 0.3).sum()),
            "max_induction": r3(h[:, 2].max()), "mean_entropy": r3(h[:, 3].mean())} for h in S]
    out = {"per_layer": per, "heads_per_layer": int(S.shape[1]), "random_tokens": T, "repeats": reps, "seed": int(seed),
           "totals": {k: int(sum(x[k] for x in per)) for k in ("sink_heads", "previous_token_heads", "induction_heads")},
           "thresholds": {"sink": 0.5, "previous_token": 0.4, "induction": 0.3}, "scores": [[[r3(v) for v in hh[:3]] for hh in h] for h in S]}
    try: ov, why = ov_copy(M, seed=seed)
    except Exception as e: ov, why = None, f"error: {e!r}"
    if ov is None:
        out["ov_check"] = f"attention-only (OV not checked: {why})"; out["induction_check"] = "attention-only"
    else:
        out["ov_check"] = "attention+OV"; out["induction_check"] = "attention+OV"; out["ov_copy"] = ov; thr = OV_COPY_MIN
        for l, x in enumerate(per):
            cp = ov["per_layer"][l] if l < len(ov["per_layer"]) else []
            x["copying_heads"] = int(sum(1 for v in cp if v >= thr))
            x["induction_heads_copying"] = int(sum(1 for hd, v in enumerate(cp) if hd < S.shape[1] and S[l, hd, 2] > 0.3 and v >= thr))
        for k in ("copying_heads", "induction_heads_copying"): out["totals"][k] = int(sum(x[k] for x in per))
        out["thresholds"]["ov_copy"] = thr
    return out


OV_COPY_MIN = 0.5         # a head 'copies' when its OV eigenvalue copy score is >= 0.5 (untrained pythia-70m heads: |score| <= 0.1)


def ov_heads(M, ly):
    """Per-head value and output weights of one layer, float [H, dh, D] and [H, D, dh]; or (None, reason)."""
    cfg = M.cfg; H = getattr(cfg, "num_attention_heads", None)
    if H is None: return None, "no num_attention_heads in the config"
    att = getattr(ly, "attention", None)
    if att is not None and hasattr(att, "query_key_value") and hasattr(att, "dense"):            # GPT-NeoX: qkv rows [H, 3, dh]
        qkv, o = att.query_key_value, att.dense
        if type(qkv) is not torch.nn.Linear or type(o) is not torch.nn.Linear: return None, "quantised attention weights"
        dh = qkv.weight.shape[0] // (3 * H)
        Wv = qkv.weight.float().reshape(H, 3, dh, -1)[:, 2]                                      # [H, dh, D]
        Wo = o.weight.float().reshape(o.weight.shape[0], H, dh).permute(1, 0, 2)                  # [H, D, dh]
        return (Wv, Wo), None
    sa = getattr(ly, "self_attn", None)
    if sa is not None and hasattr(sa, "v_proj") and hasattr(sa, "o_proj"):                       # Llama / Qwen / Mistral / OLMo (GQA)
        v, o = sa.v_proj, sa.o_proj
        if type(v) is not torch.nn.Linear or type(o) is not torch.nn.Linear: return None, "quantised attention weights"
        dh = getattr(cfg, "head_dim", None) or o.weight.shape[1] // H
        Hkv = v.weight.shape[0] // dh
        Wv = v.weight.float().reshape(Hkv, dh, -1).repeat_interleave(H // max(Hkv, 1), 0)         # [H, dh, D]
        Wo = o.weight.float().reshape(o.weight.shape[0], H, dh).permute(1, 0, 2)
        return (Wv, Wo), None
    return None, f"attention layout not handled ({type(att if att is not None else sa).__name__})"


@torch.no_grad()
def ov_copy(M, n_tokens=500, top=10, seed=0):
    """OV copy check (Elhage et al. 2021): per head, the full OV circuit C = W_U W_O W_V E on n_tokens sampled tokens
    ([n, n]: logit of token j when the head reads token i), with E the effective embedding (W_E plus layer 0's MLP on it,
    as in Nanda's induction work; raw W_E where the layout is not handled); norms otherwise ignored.
    copy_score = sum Re(eigenvalues) / sum |eigenvalues| (+1: copies, 0: none, -1: suppresses). Also top10_share: the
    share of sampled tokens that C ranks first-10 among the sampled tokens. ({per_layer, ...}, None) or (None, why)."""
    if M.moe: return None, "mixture-of-experts model (int8 attention, MXFP4 experts)"
    if M.WE.dtype not in (torch.float32, torch.bfloat16, torch.float16): return None, "embedding not a float matrix"
    V = min(int(M.WU.shape[0]), int(M.WE.shape[0])); g = torch.Generator().manual_seed(int(seed) + 17)
    lo = 1000 if V > 3000 else 0
    tok = torch.randint(lo, V - lo, (n_tokens,), generator=g).to(M.WE.device)
    E = M.WE[tok].float(); WU = M.WU[tok].float(); ly0 = M.layers[0]; eff = "raw W_E"
    try:
        ln = getattr(ly0, "post_attention_layernorm", None)
        if ln is not None and hasattr(ly0, "mlp"):
            x = M.WE[tok]; E = (x + ly0.mlp(ln(x))).float(); eff = "W_E + MLP0(norm(W_E))"
    except Exception as e: eff = f"raw W_E (MLP0 failed: {type(e).__name__})"
    per, top_share = [], []
    for ly in M.layers:
        w, why = ov_heads(M, ly)
        if w is None: return None, why
        Wv, Wo = (x.to(E.device) for x in w); row, row2 = [], []
        for h in range(Wv.shape[0]):
            C = ((E @ Wv[h].T) @ Wo[h].T) @ WU.T                                                   # [n, n]
            lam = torch.linalg.eigvals(C.cpu().double())
            row.append(r3(float(lam.real.sum() / lam.abs().sum().clamp_min(1e-30))))
            row2.append(r3(float(((C > C.diagonal()[:, None]).sum(1) < top).float().mean())))
        per.append(row); top_share.append(row2)
    return {"per_layer": per, "top10_share": top_share, "n_tokens": n_tokens, "top": top, "seed": int(seed), "embedding": eff,
            "top10_chance": r3(top / n_tokens), "threshold": OV_COPY_MIN,
            "note": "copy_score = sum Re(eig) / sum |eig| of W_U W_OV E over sampled tokens; layer norms ignored except inside MLP0"}, None


@torch.no_grad()
def routing(M, sents, y, perms=200):
    """For each layer: how evenly experts are used at the marked tokens, and the normalised mutual information
    between the sense and the top routed expert, against shuffled senses."""
    seen = {}
    orig = L.MoEExperts.forward
    def fwd(self, hs, router_indices=None, routing_weights=None):
        seen[id(self)] = router_indices.detach().cpu() if router_indices is not None else None
        return orig(self, hs, router_indices, routing_weights)
    L.MoEExperts.forward = fwd
    top = []
    try:
        for s in sents:
            text, span = L.unmark(s); ids, p = M.span_pos(text, span)
            M.forward_capture(torch.tensor([ids], device=L.DEV), torch.tensor([p], device=L.DEV))
            top.append([int(seen[id(ex)].reshape(len(ids), -1)[p, 0]) for ex in M.experts])
    finally:
        L.MoEExperts.forward = orig
    T = np.array(top); y = np.array(y); E = M.experts[0].E; rng = L.draw(5)
    def nmi(a, b):
        J = np.zeros((a.max() + 1, E)); np.add.at(J, (a, b), 1); J /= J.sum()
        pa, pb = J.sum(1, keepdims=True), J.sum(0, keepdims=True); nz = J > 0
        mi = (J[nz] * np.log(J[nz] / (pa @ pb)[nz])).sum()
        h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
        return mi / max(1e-9, np.sqrt(h(pa) * h(pb)))
    out = []
    for l in range(T.shape[1]):
        u = np.bincount(T[:, l], minlength=E) / len(T); ent = -(u[u > 0] * np.log(u[u > 0])).sum() / np.log(E)
        v = nmi(y, T[:, l]); null = [nmi(rng.permutation(y), T[:, l]) for _ in range(perms)]
        out.append({"experts_used": int((u > 0).sum()), "use_evenness": r3(ent), "sense_nmi": r3(v), "null_mean": r3(np.mean(null)), "p": r3(np.mean(np.array(null) >= v))})
    return {"per_layer": out, "experts": E, "n_sentences": len(T)}


@torch.no_grad()
def run(path, tag, init, q8):
    t0 = time.time(); secs = {}
    def timed(name, f, *a):
        t = time.time(); v = f(*a); secs[name] = round(time.time() - t, 1); L.log(f"   {name} {secs[name]}s"); return v
    L.log(f"== deep {tag} ({path}{'  untrained twin' if init else ''})")
    if L.DEV == "cuda": torch.cuda.reset_peak_memory_stats()
    M = L.Model(path, init=init, seed=L.SEED, q8=q8)
    res = {"tag": tag, "provenance": provenance(M, path, init, q8)}
    res["tokens"] = timed("tokens", tokens, M)
    res["weights"] = timed("weights", weights, M); res["weights_note"] = ALPHA_NOTE
    en = [s for k in L.SENSE_NAMES for s in D.SENSES[k]]; y = [i for i, k in enumerate(L.SENSE_NAMES) for _ in D.SENSES[k]]
    R, _ = timed("read", M.read_marked, en, False)
    res["residual"] = timed("residual", residual, R)
    ladder = os.path.join(L.OUT, tag.replace("@", "_") + ".json")
    best = json.load(open(ladder, encoding="utf-8"))["M2_features"]["best_layer"] if os.path.exists(ladder) else int(np.argmax([x["effective_dim"] for x in res["residual"]]))
    res["senses"] = timed("senses", senses, R, y, best)
    # Per-layer centred Gram matrices of the same 140 sentences: any two models can be compared layer by layer (CKA)
    # (raw, and with every dimension standardised so a few massive dimensions cannot dominate the comparison)
    G, Gz = [], []
    for l in range(R.shape[1]):
        X = R[:, l].float(); X = X - X.mean(0); G.append((X @ X.T).numpy())
        Z = X / (X.std(0) + 1e-6); Gz.append((Z @ Z.T).numpy())
    np.savez_compressed(os.path.join(OUT, tag.replace("@", "_") + ".npz"), gram=np.stack(G).astype(np.float32), gram_z=np.stack(Gz).astype(np.float32))
    del R
    res["lens"] = timed("lens", lens, M)
    res["heads"] = timed("heads", heads, M, 48, 4, L.SEED)
    if M.moe: res["routing"] = timed("routing", routing, M, en, y)
    res["provenance"]["peak_gpu_gb"] = r3(torch.cuda.max_memory_allocated() / 2 ** 30) if L.DEV == "cuda" else None
    res["provenance"]["seconds"] = secs | {"total": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, tag.replace("@", "_") + ".json"), "w", encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False, indent=1)
    L.log(f"   deep done in {res['provenance']['seconds']['total']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--init", action="store_true"); ap.add_argument("--q8", action="store_true"); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="the untrained twin's init and the head test's random tokens (default 0)")
    # arguments stripped: arguments read from a file written on Windows can end in CR, which argparse would reject
    a = ap.parse_args([x.strip() for x in sys.argv[1:] if x.strip()]); L.SEED = a.seed
    if a.cpu: L.DEV = "cpu"
    L.LOG = os.path.join(OUT, "deep.log")
    run(a.model, a.tag, a.init, a.q8)
