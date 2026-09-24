"""A trained OSC policy: one observer per checkpoint + the certified thresholds
(+ optionally the certified VERIFY setting: tau model, kappa, ell)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .certify import Shape, thresholds
from .features import candidate_features, checkpoint_order
from .observer import Observer


class OSCPolicy:
    def __init__(self, task: str, max_rounds: int, mid_round: bool, observers: dict[str, Observer],
                 a: float | None, shape: Shape, meta: dict | None = None,
                 verify: dict | None = None):
        self.task = task
        self.max_rounds = max_rounds
        self.mid_round = mid_round
        self.checkpoints = checkpoint_order(max_rounds, mid_round)
        self.observers = observers
        self.a = a
        self.shape = shape
        self.meta = meta or {}
        # a=None means nothing could be certified: never stop before the last checkpoint
        a_eff = a if a is not None else np.inf
        self.thr = dict(zip(self.checkpoints, thresholds(self.checkpoints, a_eff, shape)))
        # verify: {"tau": TauModel dict, "kappa": float, "ell": float, "mode": str}; None = never
        self.verify = verify
        self._tau = None
        if verify is not None:
            from .verify import TauModel
            self._tau = TauModel.from_dict(verify["tau"])

    def has_checkpoint(self, cp: str) -> bool:
        return cp in self.thr

    def assess(self, prefix: list[dict], cp: str, force_stop: bool = False,
               boost: tuple | None = None) -> dict:
        """Score the candidates visible at checkpoint cp and decide whether to stop.
        boost = (answer, log likelihood ratio) from an earlier verification of that answer."""
        cands, F = candidate_features(prefix)
        if not cands:
            return dict(checkpoint=cp, top=None, second=None, top_pct=0.0, lead=0.0,
                        threshold=float(self.thr.get(cp, np.inf)), stop=force_stop, probs={})
        obs = self.observers[cp]
        z = obs.raw_scores(F) / obs.T
        if boost is not None and boost[0] in cands:
            z = z + boost[1] * np.array([c == boost[0] for c in cands], float)
        p = np.exp(z - z.max())
        p = p / p.sum()
        order = np.argsort(-p)
        top_pct = float(p[order[0]])
        second_pct = float(p[order[1]]) if len(p) > 1 else 0.0
        lead = top_pct - second_pct
        thr = float(self.thr.get(cp, np.inf))
        dec = dict(
            checkpoint=cp, top=cands[order[0]],
            second=cands[order[1]] if len(p) > 1 else None,
            top_pct=round(top_pct, 4), lead=round(lead, 4),
            threshold=None if not np.isfinite(thr) else round(thr, 4),
            stop=bool(force_stop or lead >= thr),
            probs={c: round(float(x), 4) for c, x in zip(cands, p)},
        )
        if self.verify is not None:
            from .verify import agree_latest, state_vector
            ci = self.checkpoints.index(cp)
            s = state_vector(top_pct, lead, F[order[0]], agree_latest(prefix, cands[order[0]]),
                             ci, len(self.checkpoints))
            dec["state"] = [round(float(x), 4) for x in s]
        return dec

    def verify_now(self, dec: dict) -> dict | None:
        """Proposition 1: verify the leading answer iff ell <= p_top <= u(s).
        Returns {"t1", "t0", "u"} when the policy says verify, else None."""
        if self.verify is None or dec.get("top") is None or "state" not in dec:
            return None
        from .verify import verify_band_upper
        t1, t0 = self._tau.taus(np.array(dec["state"]))
        t1, t0 = float(t1[0]), float(t0[0])
        u = float(verify_band_upper(t1, t0, self.verify["kappa"]))
        if self.verify["ell"] <= dec["top_pct"] <= u:
            return dict(t1=round(t1, 4), t0=round(t0, 4), u=round(u, 4))
        return None

    @staticmethod
    def log_lr(verdict: str, t1: float, t0: float) -> float:
        if verdict == "pass":
            return math.log(t1 / t0)
        if verdict == "fail":
            return math.log((1 - t1) / (1 - t0))
        return 0.0

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return dict(task=self.task, max_rounds=self.max_rounds, mid_round=self.mid_round,
                    a=self.a, b=self.shape.b, mid_extra=self.shape.mid_extra,
                    observers={cp: o.to_dict() for cp, o in self.observers.items()},
                    verify=self.verify, meta=self.meta)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "OSCPolicy":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["task"], d["max_rounds"], d["mid_round"],
                   {cp: Observer.from_dict(o) for cp, o in d["observers"].items()},
                   d["a"], Shape(d["b"], d["mid_extra"]), d.get("meta"), d.get("verify"))
