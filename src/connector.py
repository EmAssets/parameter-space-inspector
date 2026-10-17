"""Live connector meter: where a sentence falls on the model's shared "connection" axis
(learned from the stored connector study, research/results/connector-qwen.npz). 0 = the average sentence about a thing on its own, 1 = the average
sentence about a connection, at Qwen3-4B's best layer past the word embeddings (the deep layer).

Also returned: each token's pull along the axis (((h_t - mu) / sd) @ shared / T, in axis units; the
pulls sum to the sentence's position), and, when the .npz carries a layer-0 axis (shared0, mu0, sd0,
ent0, scale0; the bundled file does not carry one), the same sentence read from the word
embeddings alone (layer 0: mostly word choice) as a baseline; otherwise baseline.state = "not stored"."""
import os
import numpy as np
import torch

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "results", "connector-qwen.npz")


class Connector:
    def __init__(self, tracer):
        self.t = tracer; z = np.load(PATH, allow_pickle=False)
        self.shared = torch.tensor(z["shared"]); self.mu = torch.tensor(z["mu"]); self.sd = torch.tensor(z["sd"])
        self.ent = float(z["ent"]); self.scale = float(z["scale"]); self.layer = int(z["layer"])
        self.levels = [str(x) for x in z["levels"]]; self.per_level = [float(x) for x in z["per_level"]]
        self.language = float(z["language"]); self.meta = float(z["meta"])
        self.base = None
        if all(k in z.files for k in ("shared0", "mu0", "sd0", "ent0", "scale0")):
            self.base = {"shared": torch.tensor(z["shared0"]), "mu": torch.tensor(z["mu0"]), "sd": torch.tensor(z["sd0"]), "ent": float(z["ent0"]), "scale": float(z["scale0"])}

    def _name(self):
        p = str(getattr(self.t.model.config, "_name_or_path", "") or "").replace("\\", "/").rstrip("/").split("/")[-1]
        p = p[:-3] if p.lower().endswith("-hf") else p
        return "Qwen3-4B" if p.lower().startswith("qwen3-4b") else (p or "the loaded model")

    def _fits(self):
        """The connection axis was learned from Qwen3-4B's insides; another model's numbers mean something else."""
        cfg = self.t.model.config; d = int(self.mu.shape[0])
        if cfg.hidden_size != d or self.layer > cfg.num_hidden_layers:
            raise ValueError(f"The connection axis was learned inside Qwen3-4B ({d:,} numbers at layer {self.layer}); the inside engine now holds a different model "
                             f"({cfg.hidden_size:,} numbers, {cfg.num_hidden_layers} layers), so it cannot be read. Load Qwen3-4B: the axis is stored for that model only.")

    @torch.no_grad()
    def score(self, text):
        self._fits()
        tok = self.t.tok; ids = tok(text, return_tensors="pt").input_ids.to(self.t.device)
        hs = self.t.model(input_ids=ids, output_hidden_states=True).hidden_states
        H = hs[self.layer][0].float().cpu(); T = max(1, H.shape[0])
        pull = (((H - self.mu) / self.sd) @ self.shared) / T            # raw units; sums to the pooled projection
        x = float(pull.sum()); pos = (x - self.ent) / (self.scale + 1e-9)
        pieces = [tok.decode([int(i)]) for i in ids[0].tolist()]
        # per-token pull in axis units: the token's share of (x - ent) / scale, so the pulls sum to the position
        tokens = [{"piece": pc, "pull": round(float((v - self.ent / T) / (self.scale + 1e-9)), 3)} for pc, v in zip(pieces, pull.tolist())]
        base = {"state": "not stored", "how": "research/results/connector-qwen.npz holds only the deep layer's axis; a words-alone reading would need the layer-0 axis stored beside it (shared0, mu0, sd0, ent0, scale0)"}
        if self.base:
            b = self.base; x0 = float(((hs[0][0].float().cpu().mean(0) - b["mu"]) / b["sd"]) @ b["shared"])
            base = {"state": "read", "layer": 0, "position": round((x0 - b["ent"]) / (b["scale"] + 1e-9), 3)}
        return {"text": text, "position": round(pos, 3), "layer": self.layer, "model": self._name(), "unit": "connection-axis units (0 = thing on its own, 1 = average connection)",
                "reference": {"levels": dict(zip(self.levels, self.per_level)), "language": self.language, "meta": self.meta}, "tokens": tokens, "baseline": base, "n_tokens": len(pieces)}
