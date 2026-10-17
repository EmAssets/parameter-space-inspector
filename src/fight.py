"""Live fight-class classifier: the stored ridge probe (research/results/fightprobe-qwen.npz), applied to the model's
mean-pooled residual stream at the probe's layer. Six classes by what a competitor models
(emergencemachine.com/what-kind-of-fight-is-this/).

What comes back, and what each number is:
  raw      the ridge probe's output per class (fitted to one-hot targets): the measurement.
  share    softmax(raw * TEMPERATURE), TEMPERATURE = 4. A display choice so six numbers read as
           percentages; it is not a probability and says nothing against chance by itself.
  baseline the same sentence read from the word embeddings alone (layer 0), when the probe file
           stores that probe too (W0/mu0/sd0 in the .npz); otherwise "not stored". The stored study's
           layer-0 accuracy is the vocabulary floor the live reading should be put beside.
  tokens   each token's pull on the top class: ((h_t - mu) / sd) @ W[:, top] / T, which sums to the
           pooled raw score (mean pooling is linear), so the words that carried the reading are visible.
"""
import os
import numpy as np
import torch

NAMES = {"A": "the Field", "B": "Gates", "C": "the Autopilot", "D": "the Strategist", "E": "the Director", "F": "the Worldmaker"}
MODELS = {"A": "nothing: physics resolving", "B": "thresholds: if X then Y", "C": "the world, not itself", "D": "itself modeling the world, and its opponent",
          "E": "how others model, and shapes it", "F": "what exists: the categories themselves"}
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "results", "fightprobe-qwen.npz")
TEMPERATURE = 4.0


class Fight:
    def __init__(self, tracer):
        self.t = tracer; z = np.load(PATH, allow_pickle=False)
        self.W = torch.tensor(z["W"]); self.mu = torch.tensor(z["mu"]); self.sd = torch.tensor(z["sd"]); self.classes = [str(c) for c in z["classes"]]; self.layer = int(z["layer"])
        # the word-embedding probe, when the research script stored it (older files have none)
        self.W0 = torch.tensor(z["W0"]) if "W0" in z.files else None
        self.mu0 = torch.tensor(z["mu0"]) if "mu0" in z.files else None; self.sd0 = torch.tensor(z["sd0"]) if "sd0" in z.files else None
        self.trained_on = str(z["trained_on"]) if "trained_on" in z.files else "not recorded"

    def _name(self):
        p = str(getattr(self.t.model.config, "_name_or_path", "") or "").replace("\\", "/").rstrip("/").split("/")[-1]
        p = p[:-3] if p.lower().endswith("-hf") else p
        return "Qwen3-4B" if p.lower().startswith("qwen3-4b") else (p or "the loaded model")

    def _fits(self):
        """The fight-class probe was learned from Qwen3-4B's insides; another model's numbers mean something else."""
        cfg = self.t.model.config; d = int(self.mu.shape[0])
        if cfg.hidden_size != d or self.layer > cfg.num_hidden_layers:
            raise ValueError(f"The fight-class probe was learned inside Qwen3-4B ({d:,} numbers at layer {self.layer}); the inside engine now holds a different model "
                             f"({cfg.hidden_size:,} numbers, {cfg.num_hidden_layers} layers), so it cannot be read. Load Qwen3-4B: the probe is stored for that model only.")

    @torch.no_grad()
    def classify(self, text):
        self._fits()
        tok = self.t.tok; ids = tok(text, return_tensors="pt").input_ids.to(self.t.device)
        hs = self.t.model(input_ids=ids, output_hidden_states=True).hidden_states
        H = hs[self.layer][0].float().cpu()                     # tokens x d
        Z = (H - self.mu) / self.sd; s = Z.mean(0) @ self.W      # raw ridge outputs, one per class
        p = torch.softmax(s * TEMPERATURE, -1)
        order = torch.argsort(-p).tolist(); top = order[0]
        pull = (Z @ self.W[:, top]) / max(1, Z.shape[0])         # per-token pull on the top class; sums to s[top]
        pieces = [tok.decode([int(i)]) for i in ids[0].tolist()]
        tokens = [{"piece": pc, "pull": round(float(v), 3)} for pc, v in zip(pieces, pull.tolist())]
        base = {"state": "not stored", "how": "the bundled probe file has no layer-0 probe beside the best layer, so there is no words-alone reading"}
        if self.W0 is not None:
            s0 = ((hs[0][0].float().cpu().mean(0) - self.mu0) / self.sd0) @ self.W0; p0 = torch.softmax(s0 * TEMPERATURE, -1); o0 = int(torch.argmax(p0))
            base = {"state": "read", "layer": 0, "top": self.classes[o0], "name": NAMES[self.classes[o0]], "share": round(float(p0[o0]), 3), "raw": round(float(s0[o0]), 3),
                    "agrees": self.classes[o0] == self.classes[top]}
        return {"layer": self.layer, "top": self.classes[top], "name": NAMES[self.classes[top]], "models": MODELS[self.classes[top]],
                "scores": [{"class": self.classes[i], "name": NAMES[self.classes[i]], "share": round(float(p[i]), 3), "raw": round(float(s[i]), 3)} for i in order],
                "temperature": TEMPERATURE, "share_note": f"share = softmax(raw × {TEMPERATURE:g}): a display choice, not a probability; compare raw scores or the stored study's accuracies against chance",
                "baseline": base, "tokens": tokens, "model": self._name(), "probe_trained_on": self.trained_on, "n_tokens": len(pieces)}
