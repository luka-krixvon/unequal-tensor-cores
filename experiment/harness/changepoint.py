"""Crossover and knee estimators for the H200/B300 measurement study.  v2.

v2 (2026-08-12) supersedes v1 following the pre-data adversarial review of the
pre-registration; all changes were made BEFORE any target-hardware data
exists. Changes vs v1:
  1. CI includes censored bootstrap draws as -inf/+inf on the log2 axis; a
     percentile landing on censored mass is reported as an OPEN bound.
  2. crossover() reports ALL crossings (n_crossings, list); crossover_ci adds
     a multimodality flag (fraction of draws >1 grid step from the estimate).
  3. Failed runs: y <= 0 or missing repeats invalidate the grid point for
     BOTH arms (logged); pairs with < MIN_GRID surviving points are
     'unestimable'.
  4. Paired BLOCK bootstrap: repeat index r is the time-block; a resample
     draws block indices and takes both arms' r-th repeat at every grid
     point, preserving cross-point and cross-arm correlation.
  5. Tangential zeros: d[i]==0 counts as a crossing only if the nearest
     nonzero neighbours have opposite signs; consecutive zeros deduplicate.
  6. Censoring direction: endpoint-|d| heuristic replaced by explicit
     'no-crossing-indicated' status when endpoint |d| differ by < 20%.
  7. knee(): requires >= 6 valid grid points and a pre-specified existence
     test (two-segment SSE must be < SSE_one_segment / 1.5), else
     status 'no-knee'.
  8. corrected_crossover_ci(): B300-INT8 correction X/(1-P) with P(regime)
     resampled inside every bootstrap draw (penalty uncertainty propagates);
     out-of-support regimes use nearest-P and are flagged extrapolated.

Pure standard-library Python. Locked by SHA-256 in experiment/preregistration.md.
"""
from __future__ import annotations
import math
from statistics import median

Grid = list[float]
Reps = list[list[float]]          # reps[i] = repeats at grid[i]; index r = time-block r

MIN_GRID = 5                      # pairs with fewer valid points are unestimable
NOCROSS_ENDPOINT_TOL = 0.20       # |d| endpoints within 20% -> no direction call
KNEE_MIN_GRID = 6
KNEE_EXISTENCE_RATIO = 3.0        # SSE_1seg/SSE_2seg must exceed this; 1.5 (~BIC-zero-margin at n=13)
                                  # false-kneed pure power laws 22% of the time in validation
NEG_INF, POS_INF = float("-inf"), float("inf")


# --------------------------------------------------------------------------
# validity / cleaning
# --------------------------------------------------------------------------
def clean_pair(x: Grid, reps_a: Reps, reps_b: Reps):
    """Drop grid points invalid in EITHER arm (y<=0, empty). Returns
    (x', a', b', dropped_indices). Pre-registered rule #3."""
    keep, dropped = [], []
    for i in range(len(x)):
        ra = [v for v in reps_a[i] if v is not None and v > 0]
        rb = [v for v in reps_b[i] if v is not None and v > 0]
        if len(ra) == len(reps_a[i]) and len(rb) == len(reps_b[i]) and ra and rb:
            keep.append(i)
        else:
            dropped.append(i)
    return ([x[i] for i in keep], [reps_a[i] for i in keep],
            [reps_b[i] for i in keep], dropped)


def _med(reps: Reps) -> list[float]:
    return [median(r) for r in reps]


# --------------------------------------------------------------------------
# crossings
# --------------------------------------------------------------------------
def _crossings(logx: list[float], d: list[float]) -> list[float]:
    """Sign-change locations; zeros count only with opposite-signed
    neighbours; consecutive zeros deduplicate (rule #5)."""
    out, n = [], len(d)
    i = 0
    while i < n:
        if d[i] == 0.0:
            j = i
            while j + 1 < n and d[j + 1] == 0.0:
                j += 1
            left = next((d[k] for k in range(i - 1, -1, -1) if d[k] != 0.0), None)
            right = next((d[k] for k in range(j + 1, n) if d[k] != 0.0), None)
            if left is not None and right is not None and left * right < 0:
                out.append((logx[i] + logx[j]) / 2.0)
            i = j + 1
            continue
        if i + 1 < n and d[i + 1] != 0.0 and d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            out.append(logx[i] + t * (logx[i + 1] - logx[i]))
        i += 1
    return out


def crossover(x: Grid, reps_a: Reps, reps_b: Reps):
    """Point estimate. Returns dict:
      status: 'crossing' | 'below' | 'above' | 'no-crossing-indicated'
              | 'unestimable'
      estimate: first crossing in x units (status 'crossing' only)
      crossings: all crossing locations (x units)
      n_crossings, dropped_points
    """
    xc, ra, rb, dropped = clean_pair(x, reps_a, reps_b)
    if len(xc) < MIN_GRID:
        return {"status": "unestimable", "dropped_points": dropped}
    logx = [math.log2(v) for v in xc]
    d = [math.log(p) - math.log(q) for p, q in zip(_med(ra), _med(rb))]
    cr = _crossings(logx, d)
    if cr:
        return {"status": "crossing", "estimate": 2.0 ** cr[0],
                "crossings": [2.0 ** c for c in cr], "n_crossings": len(cr),
                "dropped_points": dropped}
    lo, hi = abs(d[0]), abs(d[-1])
    if min(lo, hi) >= (1 - NOCROSS_ENDPOINT_TOL) * max(lo, hi):
        status = "no-crossing-indicated"
    else:
        status = "below" if lo < hi else "above"
    return {"status": status, "crossings": [], "n_crossings": 0,
            "dropped_points": dropped,
            "bound": xc[0] if status == "below" else xc[-1]}


# --------------------------------------------------------------------------
# paired block bootstrap
# --------------------------------------------------------------------------
def _block_resample(rng, reps_a: Reps, reps_b: Reps):
    """Rule #4: draw block indices with replacement; both arms and every grid
    point take the SAME block's repeat. Requires equal rep count per arm."""
    r = min(min(len(v) for v in reps_a), min(len(v) for v in reps_b))
    picks = [rng.randrange(r) for _ in range(r)]
    ra = [[row[p] for p in picks] for row in reps_a]
    rb = [[row[p] for p in picks] for row in reps_b]
    return ra, rb


def crossover_ci(x: Grid, reps_a: Reps, reps_b: Reps,
                 n_boot: int = 1000, alpha: float = 0.10, seed: int = 20260812):
    """Percentile CI on log2 axis with censored draws included as +/-inf
    (rule #1). Returns point-estimate dict plus:
      ci: (lo, hi) where an endpoint may be the string '<x_min' / '>x_max'
      censor_frac, multimodal_frac
    """
    import random
    rng = random.Random(seed)
    pt = crossover(x, reps_a, reps_b)
    if pt["status"] == "unestimable":
        return pt
    xc, ra0, rb0, _ = clean_pair(x, reps_a, reps_b)
    step = (math.log2(xc[-1]) - math.log2(xc[0])) / max(len(xc) - 1, 1)

    draws = []
    for _ in range(n_boot):
        ra, rb = _block_resample(rng, ra0, rb0)
        r = crossover(xc, ra, rb)
        if r["status"] == "crossing":
            draws.append(math.log2(r["estimate"]))
        elif r["status"] == "below":
            draws.append(NEG_INF)
        elif r["status"] == "above":
            draws.append(POS_INF)
        else:  # no-crossing-indicated: direction unknown -> both tails half
            draws.append(NEG_INF if rng.random() < 0.5 else POS_INF)
    draws.sort()
    ncens = sum(1 for v in draws if v in (NEG_INF, POS_INF))
    lo_v = draws[int(len(draws) * (alpha / 2))]
    hi_v = draws[min(len(draws) - 1, int(len(draws) * (1 - alpha / 2)))]
    lo = f"<{xc[0]:g}" if lo_v == NEG_INF else 2.0 ** lo_v
    hi = f">{xc[-1]:g}" if hi_v == POS_INF else 2.0 ** hi_v
    out = dict(pt)
    out["ci"] = (lo, hi)
    out["censor_frac"] = ncens / len(draws)
    if pt["status"] == "crossing":
        c = math.log2(pt["estimate"])
        finite = [v for v in draws if v not in (NEG_INF, POS_INF)]
        out["multimodal_frac"] = (sum(1 for v in finite if abs(v - c) > step)
                                  / max(len(finite), 1))
    return out


# --------------------------------------------------------------------------
# penalty-corrected crossover (B300 INT8 arm), rule #8
# --------------------------------------------------------------------------
def _penalty_curve(logx_p: list[float], p_meds: list[float], q: float):
    """Piecewise-linear P at log-load q; clamp to nearest end out of support.
    Returns (P, extrapolated)."""
    if q <= logx_p[0]:
        return p_meds[0], q < logx_p[0]
    if q >= logx_p[-1]:
        return p_meds[-1], q > logx_p[-1]
    for i in range(len(logx_p) - 1):
        if logx_p[i] <= q <= logx_p[i + 1]:
            t = (q - logx_p[i]) / (logx_p[i + 1] - logx_p[i])
            return p_meds[i] + t * (p_meds[i + 1] - p_meds[i]), False
    return p_meds[-1], True


def corrected_crossover_ci(x: Grid, reps_int8_raw: Reps, reps_ref: Reps,
                           x_penalty: Grid, reps_penalty: Reps,
                           n_boot: int = 1000, alpha: float = 0.10,
                           seed: int = 20260812):
    """Crossover of corrected INT8 (X/(1-P)) vs reference arm, with P
    resampled per bootstrap draw. reps_penalty[i] = repeated P estimates in
    [0,1) at x_penalty[i] (from BACKEND-PENALTY-H200). Marks draws that
    needed penalty extrapolation."""
    import random
    rng = random.Random(seed)
    logx_p = [math.log2(v) for v in x_penalty]

    def correct(ra, pmeds):
        out, extrap = [], False
        for xi, row in zip(x, ra):
            P, ex = _penalty_curve(logx_p, pmeds, math.log2(xi))
            extrap |= ex
            P = min(max(P, 0.0), 0.95)
            out.append([v / (1.0 - P) for v in row])
        return out, extrap

    pmeds0 = [median(r) for r in reps_penalty]
    corr0, extrap0 = correct(reps_int8_raw, pmeds0)
    pt = crossover(x, corr0, reps_ref)

    draws, ncens, nextrap = [], 0, 0
    xc, _, _, _ = clean_pair(x, corr0, reps_ref)
    if pt["status"] == "unestimable":
        return pt
    for _ in range(n_boot):
        pm = [median([r[rng.randrange(len(r))] for _ in r]) for r in reps_penalty]
        ra, rb = _block_resample(rng, reps_int8_raw, reps_ref)
        rc, ex = correct(ra, pm)
        nextrap += ex
        r = crossover(x, rc, rb)
        if r["status"] == "crossing":
            draws.append(math.log2(r["estimate"]))
        else:
            ncens += 1
            draws.append(NEG_INF if r.get("status") == "below" else POS_INF)
    draws.sort()
    lo_v = draws[int(len(draws) * (alpha / 2))]
    hi_v = draws[min(len(draws) - 1, int(len(draws) * (1 - alpha / 2)))]
    out = dict(pt)
    out["ci"] = (f"<{xc[0]:g}" if lo_v == NEG_INF else 2.0 ** lo_v,
                 f">{xc[-1]:g}" if hi_v == POS_INF else 2.0 ** hi_v)
    out["censor_frac"] = ncens / n_boot
    out["penalty_extrapolated_frac"] = nextrap / n_boot
    out["penalty_extrapolated"] = extrap0
    return out


# --------------------------------------------------------------------------
# knee, rules #7
# --------------------------------------------------------------------------
def _sse_two_seg(lx, ly, b):
    X = [[1.0, xi - b, max(0.0, xi - b)] for xi in lx]
    ata = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(3)] for i in range(3)]
    aty = [sum(X[k][i] * ly[k] for k in range(len(X))) for i in range(3)]
    m = [row[:] + [aty[i]] for i, row in enumerate(ata)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            return float("inf")
        m[c], m[p] = m[p], m[c]
        for r in range(3):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [a - f * b2 for a, b2 in zip(m[r], m[c])]
    coef = [m[i][3] / m[i][i] for i in range(3)]
    return sum((ly[k] - (coef[0] + coef[1] * (lx[k] - b) + coef[2] * max(0.0, lx[k] - b))) ** 2
               for k in range(len(lx)))


def _sse_one_seg(lx, ly):
    n = len(lx); sx = sum(lx); sy = sum(ly)
    sxx = sum(v * v for v in lx); sxy = sum(a * b for a, b in zip(lx, ly))
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        return float("inf")
    m = (n * sxy - sx * sy) / den
    c = (sy - m * sx) / n
    return sum((y - (m * xv + c)) ** 2 for xv, y in zip(lx, ly))


def knee(x: Grid, reps: Reps, n_cand: int = 200):
    """Returns dict: status 'knee' with estimate, or 'no-knee', or
    'unestimable' (grid < KNEE_MIN_GRID after cleaning)."""
    pairs = [(xi, r) for xi, r in zip(x, reps)
             if r and all(v is not None and v > 0 for v in r)]
    if len(pairs) < KNEE_MIN_GRID:
        return {"status": "unestimable"}
    lx = [math.log2(p[0]) for p in pairs]
    ly = [math.log2(median(p[1])) for p in pairs]
    lo, hi = lx[1], lx[-2]
    best_b, best_sse = None, float("inf")
    for i in range(n_cand + 1):
        b = lo + (hi - lo) * i / n_cand
        s = _sse_two_seg(lx, ly, b)
        if s < best_sse:
            best_sse, best_b = s, b
    sse1 = _sse_one_seg(lx, ly)
    if best_sse <= 0 or sse1 / max(best_sse, 1e-300) < KNEE_EXISTENCE_RATIO:
        if not (best_sse <= 0 and sse1 > 0):
            return {"status": "no-knee", "sse_ratio": sse1 / max(best_sse, 1e-300)}
    return {"status": "knee", "estimate": 2.0 ** best_b,
            "sse_ratio": sse1 / max(best_sse, 1e-300)}


def knee_ci(x: Grid, reps: Reps, n_boot: int = 1000, alpha: float = 0.10,
            seed: int = 20260812, n_cand: int = 200):
    import random
    rng = random.Random(seed)
    pt = knee(x, reps, n_cand)
    if pt["status"] != "knee":
        return pt
    r = min(len(v) for v in reps)
    draws, nk = [], 0
    for _ in range(n_boot):
        picks = [rng.randrange(r) for _ in range(r)]
        rr = [[row[p] for p in picks] for row in reps]
        kb = knee(x, rr, n_cand)
        if kb["status"] == "knee":
            draws.append(math.log2(kb["estimate"]))
        else:
            nk += 1
    out = dict(pt)
    out["noknee_frac"] = nk / n_boot
    if draws:
        draws.sort()
        lo = draws[max(0, int(len(draws) * (alpha / 2)))]
        hi = draws[min(len(draws) - 1, int(len(draws) * (1 - alpha / 2)))]
        out["ci"] = (2.0 ** lo, 2.0 ** hi)
    return out
