#!/usr/bin/env python3
"""
tools/analyze.py  --  minimal plotter for the CSV outputs of lbm_pro.

Usage:
    python3 tools/analyze.py series ts.csv         # time series + FFT
    python3 tools/analyze.py cp cp.csv             # Cp(theta) curve
    python3 tools/analyze.py both ts.csv cp.csv    # both

Reads what lbm_pro's selftest writes with:
    LB_CSV=ts.csv LB_CP=cp.csv ./lbm_pro

Writes PNGs next to the inputs (or stdout for the summary numbers).
Dependencies: numpy + matplotlib. If matplotlib is missing, prints the
numbers only (useful on headless servers where you only want a St estimate
or a Cp(theta=0) value).
"""
import sys, os, math

try:
    import numpy as np
except ImportError:
    print("numpy required", file=sys.stderr); sys.exit(2)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False


def read_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    header = open(path).readline().strip().split(",")
    return header, data


def analyze_series(path):
    hdr, d = read_csv(path)
    step = d[:, 0]
    uy   = d[:, hdr.index("uy_probe")]
    Cd   = d[:, hdr.index("Cd")]
    Cl   = d[:, hdr.index("Cl")]

    # Welch-ish PSD on Cl second half: FFT with a Hann window.
    h0 = len(Cl) // 2
    sig = Cl[h0:] - Cl[h0:].mean()
    N = len(sig)
    if N < 32:
        print(f"  too few samples (N={N}); run more steps")
        return
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    sw = sig * win
    spec = np.abs(np.fft.rfft(sw))
    freq = np.fft.rfftfreq(N, d=1.0)       # cycles / step
    # dominant freq (skip DC)
    k = 1 + int(np.argmax(spec[1:]))
    f = freq[k]
    print(f"  samples    : {len(d)}  (second half: {N})")
    print(f"  Cd mean    : {Cd[h0:].mean():+.4f}  (std {Cd[h0:].std():.4f})")
    print(f"  Cl mean    : {Cl[h0:].mean():+.4f}  (amp {0.5*(Cl[h0:].max()-Cl[h0:].min()):.4f})")
    print(f"  Cl peak f  : {f:.5f} /step  (period {1/f:.1f} steps)")

    if HAVE_PLT:
        fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 7))
        a1.plot(step, uy); a1.set_ylabel("uy probe"); a1.set_title(os.path.basename(path))
        a2.plot(step, Cd, label="Cd"); a2.plot(step, Cl, label="Cl"); a2.legend(); a2.set_ylabel("force coef")
        a3.loglog(freq[1:], spec[1:]); a3.axvline(f, color="r", lw=0.5)
        a3.set_xlabel("freq [1/step]"); a3.set_ylabel("|FFT(Cl)|"); a3.set_title(f"peak {f:.4f}")
        fig.tight_layout()
        out = os.path.splitext(path)[0] + ".png"
        fig.savefig(out, dpi=100)
        print(f"  wrote {out}")


def analyze_cp(path):
    hdr, d = read_csv(path)
    th = d[:, 0]
    Cp = d[:, 1]
    print(f"  theta=0 (upstream stagn): Cp = {Cp[0]:+.3f}  (textbook: ~+1)")
    k90 = int(np.argmin(np.abs(th - 90)))
    print(f"  theta=90 (side):          Cp = {Cp[k90]:+.3f}  (textbook: ~-1 at Re~100)")
    k180 = int(np.argmin(np.abs(th - 180)))
    print(f"  theta=180 (base/wake):    Cp = {Cp[k180]:+.3f}  (viscous: typ. ~-1 to 0)")
    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(th, Cp)
        ax.axhline(0, color="k", lw=0.3)
        ax.set_xlabel("theta [deg]  (0 = upstream stagnation)")
        ax.set_ylabel("Cp")
        ax.set_title(f"Cp(theta) around cylinder -- {os.path.basename(path)}")
        ax.grid(alpha=0.3)
        out = os.path.splitext(path)[0] + ".png"
        fig.tight_layout(); fig.savefig(out, dpi=100)
        print(f"  wrote {out}")


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "series":
        analyze_series(sys.argv[2])
    elif cmd == "cp":
        analyze_cp(sys.argv[2])
    elif cmd == "both":
        print("=== time series ===")
        analyze_series(sys.argv[2])
        print("\n=== Cp(theta) ===")
        analyze_cp(sys.argv[3])
    else:
        print(__doc__); sys.exit(1)


if __name__ == "__main__":
    main()
