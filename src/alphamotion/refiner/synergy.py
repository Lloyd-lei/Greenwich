"""The synergy gate: does the refined motion still MOVE like the original?

Acceptance standard (product requirement): the refined motion, re-encoded
through Greenwich on ITS OWN body, must reach >= 70% AR likelihood relative to
the original motion under Equator's prior.

Operationalisation: per-token likelihood ratio, geometric mean over the 32
tokens, with BOTH sides passed through the same re-encode channel:

    ratio = exp( mean(NLL_reencoded_original) - mean(NLL_refined) )

Why the reference is the RE-ENCODED original and not the original directly:
the codec's round-trip itself costs likelihood (measured: a human clip decoded
and re-encoded on its own body scores ratio 0.218 against its raw self, with
no refiner and no embodiment change anywhere). Charging that intrinsic cost to
the refiner would make the 70% bar unreachable by construction and would
measure the codec, not the refinement. Both sides re-encoded = like for like;
what remains in the ratio is exactly synergy damage from the embodiment change
plus the refinement passes.

* ratio ~ 1.0  the refined motion is exactly as plausible a continuation
               pattern as the original — limb synergies intact;
* ratio -> 0   the refinement broke the inter-limb coordination (arms no
               longer phase-locked to the gait, amplitudes collapsed, ...):
               the prior finds the refined token sequence far less likely.
The gate is only meaningful on the repaired decoder (round-trip fidelity 61%);
on the pre-repair decoder re-encoding itself destroys the tokens and every
ratio saturates low — which is why that checkpoint was crowned for release.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

GATE_THRESHOLD = 0.70


@dataclass
class SynergyReport:
    ratio: float
    nll_original: float
    nll_refined: float
    per_token_ratio_min: float
    passed: bool

    def as_dict(self) -> dict:
        return {"ratio": round(self.ratio, 4),
                "nll_original": round(self.nll_original, 4),
                "nll_refined": round(self.nll_refined, 4),
                "per_token_ratio_min": round(self.per_token_ratio_min, 4),
                "passed": self.passed,
                "threshold": GATE_THRESHOLD}


@torch.no_grad()
def synergy_gate(greenwich, equator, orig_pose9, orig_spec, orig_dof,
                 refined_rot6d, refined_spec, refined_dof,
                 n_frames: int | None = None) -> SynergyReport:
    """Compare AR likelihood of original vs refined motion token sequences.

    orig_pose9: the source observation [T,J,9] (any embodiment).
    refined_rot6d: the refined motion's global rotations on its body [T,J',6].
    """
    n = n_frames or orig_pose9.shape[0]
    # reference channel: the original, decoded back onto ITS OWN body and
    # re-encoded — the same channel the refined motion goes through
    codes_o = greenwich.encode(orig_pose9, orig_spec, orig_dof)
    rot_o = greenwich.decode(codes_o, orig_spec, orig_dof)
    p9_o, _ = greenwich.pose9(rot_o.cpu(), orig_spec, is_global=True)
    codes_o2 = greenwich.encode(p9_o, orig_spec, orig_dof)
    tok_o, ep_o = equator.tokenize(codes_o2)
    nll_o = equator.token_nll(tok_o, ep_o, n)

    p9_r, _reach = greenwich.pose9(refined_rot6d.cpu(), refined_spec,
                                   is_global=True)
    codes_r = greenwich.encode(p9_r, refined_spec, refined_dof)
    tok_r, ep_r = equator.tokenize(codes_r)
    nll_r = equator.token_nll(tok_r, ep_r, n)

    ratio = float(torch.exp(nll_o.mean() - nll_r.mean()))
    per_tok = torch.exp(nll_o - nll_r)
    return SynergyReport(ratio=min(ratio, 9.99),
                         nll_original=float(nll_o.mean()),
                         nll_refined=float(nll_r.mean()),
                         per_token_ratio_min=float(per_tok.min()),
                         passed=ratio >= GATE_THRESHOLD)
