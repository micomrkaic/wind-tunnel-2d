/* ==========================================================================
 *  lbm_common.h  --  D2Q9 primitives shared by lbm_tunnel / lbm_bench /
 *                    lbm_pro. Previously each file carried its own
 *                    identical copy; a bug fixed in one is now a bug
 *                    fixed everywhere. Include-once header; all symbols
 *                    `static inline` so there is no link-time collision
 *                    if two translation units ever share the header.
 *
 *  Scope: just the math primitives (lattice velocities, weights, opposite-
 *  direction index, c_s^2, Maxwell-Boltzmann equilibrium feq). Collision
 *  operators, boundary conditions and per-solver state stay in each
 *  solver -- they have genuinely different signatures (BGK uses a single
 *  omega; TRT uses omega_p/omega_m; lbm_pro adds a q-array). Pulling
 *  those in here would be premature abstraction, not consolidation.
 *  ========================================================================
 *  Reduced lattice units everywhere: dx = dt = 1, c_s^2 = 1/3.
 * ======================================================================== */
#ifndef LBM_COMMON_H
#define LBM_COMMON_H

/* D2Q9 lattice velocity vectors.
 *
 *      6   2   5           ^ y
 *       \  |  /            |
 *        \ | /             |
 *      3 - 0 - 1           +---> x
 *        / | \
 *       /  |  \
 *      7   4   8
 *
 * Indices 1..4 are the axial neighbours (weight 1/9), 5..8 the diagonals
 * (weight 1/36), 0 the rest population (weight 4/9). OPP[k] is the index
 * of the direction opposite to k, used for bounce-back and TRT. */
static const int    LBM_CX [9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
static const int    LBM_CY [9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
static const double LBM_W  [9] = { 4.0/9.0,
                                   1.0/9.0,  1.0/9.0,  1.0/9.0,  1.0/9.0,
                                   1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };
static const int    LBM_OPP[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
static const double LBM_CS2    = 1.0 / 3.0;

/* The Maxwell-Boltzmann equilibrium truncated to O(u^2) -- the standard
 * weakly-compressible LBM equilibrium. All three solvers (BGK, TRT in
 * the bench, TRT in lbm_pro) use the SAME equilibrium; only the
 * relaxation operator differs. */
static inline double lbm_feq(int k, double rho, double ux, double uy)
{
    double cu = LBM_CX[k] * ux + LBM_CY[k] * uy;
    double u2 = ux * ux + uy * uy;
    return LBM_W[k] * rho * (1.0 + cu / LBM_CS2
                                 + 0.5 * cu * cu / (LBM_CS2 * LBM_CS2)
                                 - 0.5 * u2 / LBM_CS2);
}

/* Compatibility aliases so existing per-file names (CX, CY, W, OPP, CS2,
 * feq) keep working inside each translation unit. Only the one solver
 * that includes this header "sees" the aliases; nothing leaks. */
#define CX   LBM_CX
#define CY   LBM_CY
#define W    LBM_W
#define OPP  LBM_OPP
#define CS2  LBM_CS2
#define feq  lbm_feq

#endif /* LBM_COMMON_H */
