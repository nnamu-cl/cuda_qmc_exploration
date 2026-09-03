#!/usr/bin/env python3
"""R_nl and P_l^m goldens. Closed forms via mpmath; grid via scipy."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.special import assoc_laguerre, lpmv

mp.mp.dps = 50

OUT = Path(__file__).resolve().parents[1] / "goldens" / "special_values.json"


def radial_norm(n: int, l: int, a0: float = 1.0) -> float:
    return math.sqrt(
        (2.0 / (n * a0)) ** 3
        * math.factorial(n - l - 1)
        / (2.0 * n * math.factorial(n + l))
    )


def radial_rnl_scipy(n: int, l: int, r: float, a0: float = 1.0) -> float:
    if r == 0.0 and l > 0:
        return 0.0
    rho = 2.0 * r / (n * a0)
    rho_l = 1.0 if l == 0 else rho**l
    laguerre = float(assoc_laguerre(rho, n - l - 1, 2 * l + 1))
    return radial_norm(n, l, a0) * math.exp(-rho / 2.0) * rho_l * laguerre


def radial_rnl_mp(n: int, l: int, r: float, a0: float = 1.0) -> mp.mpf:
    rho = 2 * mp.mpf(r) / (n * a0)
    norm = mp.sqrt(
        (2 / (n * a0)) ** 3
        * mp.factorial(n - l - 1)
        / (2 * n * mp.factorial(n + l))
    )
    if n - l - 1 <= 0:
        laguerre = mp.mpf(1) if n - l - 1 == 0 else mp.mpf(1)
        if n - l - 1 < 0:
            raise ValueError("invalid n,l")
    else:
        laguerre = mp.laguerre(n - l - 1, 2 * l + 1, rho)
    return norm * mp.exp(-rho / 2) * (mp.mpf(1) if l == 0 else rho**l) * laguerre


def main() -> None:
    r10_at_1 = float(radial_rnl_mp(1, 0, 1.0))
    expected = 2.0 / math.e
    if abs(r10_at_1 - expected) > 1e-12:
        raise SystemExit(f"R_10(1) = {r10_at_1}, expected {expected}")
    if abs(radial_rnl_scipy(1, 0, 1.0) - expected) > 1e-12:
        raise SystemExit("scipy R_10(1) mismatch")
    p11_at_0 = float(lpmv(1, 1, 0.0))
    if abs(p11_at_0 - (-1.0)) > 1e-12:
        raise SystemExit(f"P_1^1(0) = {p11_at_0}, expected -1")

    radial = []
    r_grid = np.concatenate(([0.0], np.geomspace(0.05, 12.0, 49)))
    for n in range(1, 7):
        for l in range(n):
            for r in r_grid:
                value = radial_rnl_scipy(n, l, float(r))
                if 0.2 <= float(r) <= 3.0 and n <= 3:
                    mp_value = float(radial_rnl_mp(n, l, float(r)))
                    if abs(mp_value) > 1e-14 and abs(value - mp_value) / abs(
                        mp_value
                    ) > 1e-9:
                        raise SystemExit(
                            f"scipy/mpmath R mismatch n={n} l={l} r={r}: "
                            f"{value} vs {mp_value}"
                        )
                radial.append({"n": n, "l": l, "r": float(r), "R": value})

    legendre = []
    x_grid = np.linspace(-0.99, 0.99, 50)
    for l in range(0, 7):
        for m in range(-l, l + 1):
            for x in x_grid:
                value = float(lpmv(m, l, float(x)))
                if abs(x) < 0.5 and l <= 3:
                    mp_value = float(mp.legenp(l, m, float(x)))
                    if abs(value) > 1e-14 and abs(value - mp_value) / abs(
                        value
                    ) > 1e-8:
                        raise SystemExit(
                            f"scipy/mpmath P mismatch l={l} m={m} x={x}: "
                            f"{value} vs {mp_value}"
                        )
                legendre.append(
                    {"l": l, "m": m, "x": float(x), "P": value}
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"radial": radial, "legendre": legendre}, indent=2) + "\n"
    )
    print(f"wrote {OUT} ({len(radial)} radial, {len(legendre)} legendre)")


if __name__ == "__main__":
    main()
