/* ==========================================================================
 *  lbm_tunnel.c  --  D2Q9-BGK lattice Boltzmann wind tunnel
 *
 *  Companion to wind_tunnel.c (molecular dynamics). SAME tunnel geometry,
 *  SAME obstacle shapes, SAME SDL shell and colour map -- a deliberately
 *  different SOLVER so the two can be put side by side on the low-Mach
 *  cylinder, where they must converge to the same Navier-Stokes wake.
 *
 *  This is NOT particle dynamics. The state is the D2Q9 distribution
 *  f_i(x,t) on a regular lattice; hydrodynamics emerges via Chapman-
 *  Enskog, viscosity is SET by the relaxation time tau (not measured),
 *  and walls are bounce-back. Standard BGK is weakly compressible: this
 *  solver is honest only in the subsonic regime. Push the inlet Mach up
 *  and it will destabilise -- that failure, exactly where the MD keeps
 *  working, IS the point of the comparison.
 *
 *  Reduced lattice units: dx = dt = 1, c_s^2 = 1/3.
 * ======================================================================== */

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __EMSCRIPTEN__
/* WASM build: no ffmpeg piping (there is no pipe); the main loop uses
 * ASYNCIFY so the native while(!quit) structure still yields back to
 * the browser once per frame via emscripten_sleep. OpenMP pragmas are
 * no-ops here (we build serial-only). */
#include <emscripten.h>
#endif

typedef double real;

#define PI 3.14159265358979323846
#define WIN_W 1280
#define WIN_H 720

/* World extent (sigma), shared with the MD code so geometry is identical */
#define LX 320.0
#define LY 100.0

/* Lattice resolution: nodes per sigma. NX x NY total. 3 nodes/sigma over
 * 320x100 -> 960x300 ~ 288k nodes: fine enough for a clean wake, cheap
 * enough to stay interactive. */
#define NPS 3.0
#define NX  ((int)(LX * NPS))
#define NY  ((int)(LY * NPS))

#define MAX_POLY_V 64
#define MAX_OBSTACLES 16

/* ==========================================================================
 *  Geometry + obstacles  (ported verbatim from wind_tunnel.c so the two
 *  programs see byte-identical walls and obstacle outlines)
 * ======================================================================== */

typedef enum {
    GEOM_STRAIGHT = 0,
    GEOM_NOZZLE,
    GEOM_VENTURI,
    GEOM_DIFFUSER,
    GEOM_BUMP,
    GEOM_COUNT
} Geometry;

static const char *geom_name(Geometry g)
{
    switch (g) {
        case GEOM_STRAIGHT: return "STRAIGHT";
        case GEOM_NOZZLE:   return "DE LAVAL NOZZLE";
        case GEOM_VENTURI:  return "VENTURI";
        case GEOM_DIFFUSER: return "DIFFUSER";
        case GEOM_BUMP:     return "TRANSONIC BUMP";
        default:            return "?";
    }
}

static inline real smoothstep(real a, real b, real x)
{
    if (x <= a) return 0.0;
    if (x >= b) return 1.0;
    real t = (x - a) / (b - a);
    return t * t * (3.0 - 2.0 * t);
}

static real geom_halfheight(Geometry g, real x)
{
    real hwide = 0.40 * LY;
    real hthr  = 0.17 * LY;
    switch (g) {
        case GEOM_STRAIGHT:
            return hwide;
        case GEOM_NOZZLE: {
            real xt = 0.45 * LX;
            if (x < xt) {
                real s = smoothstep(0.10 * LX, xt, x);
                return hwide + (hthr - hwide) * s;
            } else {
                real s = smoothstep(xt, 0.95 * LX, x);
                return hthr + (hwide - hthr) * s;
            }
        }
        case GEOM_VENTURI: {
            real c  = 0.5 * LX;
            real w  = 0.16 * LX;
            real dip = exp(-((x - c) * (x - c)) / (2.0 * w * w));
            return hwide - (hwide - hthr) * dip;
        }
        case GEOM_DIFFUSER: {
            real s = smoothstep(0.30 * LX, 0.85 * LX, x);
            return (0.55 * hthr + 0.45 * hwide)
                 + (hwide - (0.55 * hthr + 0.45 * hwide)) * s;
        }
        case GEOM_BUMP:
        default:
            return hwide;
    }
}

static void geom_bounds(Geometry g, real x, real *yb, real *yt)
{
    real mid = 0.5 * LY;
    if (g == GEOM_BUMP) {
        real h  = 0.40 * LY;
        real c  = 0.45 * LX;
        real w  = 0.10 * LX;
        real bump = 0.42 * h * exp(-((x - c) * (x - c)) / (2.0 * w * w));
        *yb = (mid - h) + bump;
        *yt = (mid + h);
        return;
    }
    real h = geom_halfheight(g, x);
    *yb = mid - h;
    *yt = mid + h;
}

typedef enum { OBS_CIRCLE = 0, OBS_POLY } ObsType;

typedef struct {
    ObsType type;
    real cx, cy, r;
    int  nv;
    real px[MAX_POLY_V], py[MAX_POLY_V];
    real nx[MAX_POLY_V], ny[MAX_POLY_V];
    real bbminx, bbminy, bbmaxx, bbmaxy;
} Obstacle;

static void poly_finalize(Obstacle *o)
{
    o->type = OBS_POLY;
    o->bbminx = o->bbminy =  1e30;
    o->bbmaxx = o->bbmaxy = -1e30;
    real area2 = 0.0;
    for (int i = 0; i < o->nv; i++) {
        int j = (i + 1) % o->nv;
        area2 += o->px[i] * o->py[j] - o->px[j] * o->py[i];
    }
    real sgn = (area2 >= 0.0) ? 1.0 : -1.0;
    for (int i = 0; i < o->nv; i++) {
        int j = (i + 1) % o->nv;
        real dx = o->px[j] - o->px[i];
        real dy = o->py[j] - o->py[i];
        real L  = sqrt(dx * dx + dy * dy);
        if (L < 1e-12) L = 1e-12;
        o->nx[i] =  sgn * dy / L;
        o->ny[i] = -sgn * dx / L;
        if (o->px[i] < o->bbminx) o->bbminx = o->px[i];
        if (o->py[i] < o->bbminy) o->bbminy = o->py[i];
        if (o->px[i] > o->bbmaxx) o->bbmaxx = o->px[i];
        if (o->py[i] > o->bbmaxy) o->bbmaxy = o->py[i];
    }
}

static Obstacle make_circle(real cx, real cy, real r)
{
    Obstacle o; memset(&o, 0, sizeof o);
    o.type = OBS_CIRCLE; o.cx = cx; o.cy = cy; o.r = r;
    o.bbminx = cx - r; o.bbmaxx = cx + r;
    o.bbminy = cy - r; o.bbmaxy = cy + r;
    return o;
}

static Obstacle make_rect(real cx, real cy, real w, real h)
{
    Obstacle o; memset(&o, 0, sizeof o);
    o.nv = 4;
    real hw = 0.5 * w, hh = 0.5 * h;
    o.px[0] = cx - hw; o.py[0] = cy - hh;
    o.px[1] = cx + hw; o.py[1] = cy - hh;
    o.px[2] = cx + hw; o.py[2] = cy + hh;
    o.px[3] = cx - hw; o.py[3] = cy + hh;
    poly_finalize(&o);
    return o;
}

static Obstacle make_tri(real cx, real cy, real len, real half)
{
    Obstacle o; memset(&o, 0, sizeof o);
    o.nv = 3;
    o.px[0] = cx - 0.6 * len; o.py[0] = cy;
    o.px[1] = cx + 0.4 * len; o.py[1] = cy - half;
    o.px[2] = cx + 0.4 * len; o.py[2] = cy + half;
    poly_finalize(&o);
    return o;
}

static Obstacle make_wing(real cx, real cy, real c, real tt, real aoa)
{
    Obstacle o; memset(&o, 0, sizeof o);
    const int np = 24;
    real ca = cos(-aoa), sa = sin(-aoa);
    int k = 0;
    for (int i = 0; i <= np && k < MAX_POLY_V; i++) {
        real xc = 0.5 * (1.0 - cos(PI * (real)i / (real)np));
        real yt = 5.0 * tt * (0.2969 * sqrt(xc) - 0.1260 * xc
                  - 0.3516 * xc * xc + 0.2843 * xc * xc * xc
                  - 0.1015 * xc * xc * xc * xc);
        real X = (xc - 0.30) * c, Y = yt * c;
        o.px[k] = cx + (X * ca - Y * sa);
        o.py[k] = cy + (X * sa + Y * ca);
        k++;
    }
    for (int i = np - 1; i >= 1 && k < MAX_POLY_V; i--) {
        real xc = 0.5 * (1.0 - cos(PI * (real)i / (real)np));
        real yt = 5.0 * tt * (0.2969 * sqrt(xc) - 0.1260 * xc
                  - 0.3516 * xc * xc + 0.2843 * xc * xc * xc
                  - 0.1015 * xc * xc * xc * xc);
        real X = (xc - 0.30) * c, Y = -yt * c;
        o.px[k] = cx + (X * ca - Y * sa);
        o.py[k] = cy + (X * sa + Y * ca);
        k++;
    }
    o.nv = k;
    poly_finalize(&o);
    return o;
}

static bool poly_inside(const Obstacle *o, real x, real y)
{
    if (x < o->bbminx || x > o->bbmaxx || y < o->bbminy || y > o->bbmaxy)
        return false;
    bool in = false;
    for (int i = 0, j = o->nv - 1; i < o->nv; j = i++) {
        real yi = o->py[i], yj = o->py[j];
        if (((yi > y) != (yj > y))) {
            real xint = o->px[i] + (y - yi) * (o->px[j] - o->px[i]) / (yj - yi);
            if (x < xint) in = !in;
        }
    }
    return in;
}

/* ==========================================================================
 *  D2Q9 lattice
 *
 *      6   2   5
 *        \ | /
 *      3 - 0 - 1
 *        / | \
 *      7   4   8
 * ======================================================================== */

#include "lbm_common.h"       /* D2Q9 CX/CY/W/OPP/CS2 + feq */

typedef struct {
    /* distributions: f[k][node], plus a scratch buffer for streaming */
    double *f, *ftmp;
    /* macroscopic fields (diagnostics / rendering) */
    double *rho, *ux, *uy;
    /* node classification: 1 = fluid, 0 = solid (wall or obstacle) */
    uint8_t *solid;

    Geometry geom;
    Obstacle obs[MAX_OBSTACLES];
    int nobs, obs_preset;
    double obs_scale;

    /* control parameters */
    double mach;     /* inlet speed / lattice sound speed   */
    double tau;      /* BGK relaxation time -> viscosity     */
    double rho0;     /* reference density                    */

    /* Symmetry-breaking seed for the Karman instability. A perfectly
     * symmetric, noise-free LBM sits on the unstable-but-symmetric wake
     * solution forever (a pencil balanced on its tip). Real shedding
     * needs a perturbation to grow from. This is a TRANSIENT one-sided
     * nudge applied only for perturb_window steps after it is armed; once
     * the instability has grown it sustains itself, so what you observe
     * afterwards is the natural shedding, not forced oscillation. Only
     * meaningful for the cylinder (a clean length scale + Hopf onset). */
    int  perturb;            /* armed by the user (P key)            */
    long perturb_start;      /* step at which the window opened       */
    long perturb_window;     /* how many steps the nudge stays active */

    long step;
} Lbm;

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM (%zu bytes)\n", n); exit(1); }
    return p;
}

/* world <-> lattice index helpers */
static inline double node_wx(int i) { return (i + 0.5) / NPS; }
static inline double node_wy(int j) { return (j + 0.5) / NPS; }
static inline int    NID(int i, int j) { return j * NX + i; }

/* Exact field/solid test against the true analytic geometry + obstacles,
 * identical predicate to the MD code's field_visible (so a given obstacle
 * masks the SAME region in both programs). */
static int point_solid(const Lbm *s, double x, double y)
{
    if (x < 0.0 || x > LX) return 1;
    double yb, yt; geom_bounds(s->geom, x, &yb, &yt);
    if (y < yb || y > yt) return 1;
    for (int o = 0; o < s->nobs; o++) {
        const Obstacle *ob = &s->obs[o];
        if (ob->type == OBS_CIRCLE) {
            double dx = x - ob->cx, dy = y - ob->cy;
            if (dx * dx + dy * dy < ob->r * ob->r) return 1;
        } else if (poly_inside(ob, x, y)) {
            return 1;
        }
    }
    return 0;
}

static void rebuild_obstacles(Lbm *s)
{
    s->nobs = 0;
    real cx = 0.42 * LX, cy = 0.5 * LY;
    double k = s->obs_scale;
    switch (s->obs_preset) {
        case 0: break;
        case 1: {
            double r = 0.10 * LY * k;
            if (r > 0.46 * LY) r = 0.46 * LY;
            s->obs[s->nobs++] = make_circle(cx, cy, r);
            break;
        }
        case 2: {
            double w = 0.05 * LX * k, h = 0.34 * LY * k;
            if (h > 0.92 * LY) h = 0.92 * LY;
            if (w > 0.30 * LX) w = 0.30 * LX;
            s->obs[s->nobs++] = make_rect(cx, cy, w, h);
            break;
        }
        case 3: {
            double len = 0.16 * LX * k, half = 0.16 * LY * k;
            if (half > 0.46 * LY) half = 0.46 * LY;
            if (len  > 0.40 * LX) len  = 0.40 * LX;
            s->obs[s->nobs++] = make_tri(cx, cy, len, half);
            break;
        }
        case 4: {
            double chord = 0.22 * LX * k;
            if (chord > 0.55 * LX) chord = 0.55 * LX;
            s->obs[s->nobs++] =
                make_wing(cx, cy, chord, 0.15, 12.0 * PI / 180.0);
            break;
        }
        default: break;
    }
}

#define OBS_PRESET_COUNT 5

static const char *obs_name(int p)
{
    switch (p) {
        case 0: return "NONE";
        case 1: return "CYLINDER";
        case 2: return "PLATE";
        case 3: return "WEDGE";
        case 4: return "NACA0015 WING";
        default: return "?";
    }
}

/* Rebuild the solid mask from the current geometry + obstacles. Called at
 * start and whenever geometry/obstacle/scale changes. */
static void rebuild_mask(Lbm *s)
{
    rebuild_obstacles(s);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < NY; j++)
        for (int i = 0; i < NX; i++)
            s->solid[NID(i, j)] =
                (uint8_t)point_solid(s, node_wx(i), node_wy(j));
}

/* feq provided by lbm_common.h */

static double inlet_speed(const Lbm *s)
{
    /* lattice sound speed is sqrt(CS2); "Mach" is relative to it */
    return s->mach * sqrt(CS2);
}

/* Reynolds number Re = U L / nu, in lattice units. The honest length
 * scale is the obstacle's: the cylinder diameter when present (the case
 * with a textbook Re and a well-defined shedding onset ~47). With no
 * obstacle, or a non-circular one, there is no single unambiguous L, so
 * we fall back to the duct half-height and the caller labels it as a
 * duct Reynolds number rather than pretending it is the body Re. */
static double reynolds(const Lbm *s, int *have_body_D)
{
    double U  = inlet_speed(s);
    double nu = CS2 * (s->tau - 0.5);
    if (nu < 1e-9) nu = 1e-9;
    double L;
    if (s->nobs > 0 && s->obs[0].type == OBS_CIRCLE) {
        L = 2.0 * s->obs[0].r * NPS;       /* cylinder diameter, nodes */
        if (have_body_D) *have_body_D = 1;
    } else {
        L = geom_halfheight(s->geom, 0.5 * LX) * NPS;
        if (have_body_D) *have_body_D = 0;
    }
    return U * L / nu;
}

/* Initialise every fluid node to the uniform inlet flow at equilibrium. */
static void lbm_init_field(Lbm *s)
{
    double U = inlet_speed(s);
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < NX * NY; n++) {
        s->rho[n] = s->rho0;
        s->ux[n]  = s->solid[n] ? 0.0 : U;
        s->uy[n]  = 0.0;
        for (int k = 0; k < 9; k++)
            s->f[k * NX * NY + n] = feq(k, s->rho0, s->ux[n], s->uy[n]);
    }
    s->step = 0;
}

static void lbm_alloc(Lbm *s)
{
    size_t N = (size_t)NX * NY;
    s->f     = xmalloc(sizeof(double) * 9 * N);
    s->ftmp  = xmalloc(sizeof(double) * 9 * N);
    s->rho   = xmalloc(sizeof(double) * N);
    s->ux    = xmalloc(sizeof(double) * N);
    s->uy    = xmalloc(sizeof(double) * N);
    s->solid = xmalloc(sizeof(uint8_t) * N);

    s->geom       = GEOM_STRAIGHT;   /* cylinder-wake comparison default */
    s->obs_preset = 1;               /* cylinder */
    s->obs_scale  = 1.0;
    s->mach       = 0.10;            /* firmly subsonic: BGK's honest band */
    s->rho0       = 1.0;
    /* tau from a target kinematic viscosity. nu = cs^2 (tau - 1/2).
     * tau = 0.6 -> nu = 1/30 in lattice units: a clean, stable wake with
     * vortex shedding at this resolution and inlet speed. */
    s->tau        = 0.60;

    s->perturb        = 0;
    s->perturb_start  = 0;
    s->perturb_window = 400;    /* brief one-shot kick, then natural */

    rebuild_mask(s);
    lbm_init_field(s);
}

static void lbm_free(Lbm *s)
{
    free(s->f); free(s->ftmp); free(s->rho);
    free(s->ux); free(s->uy); free(s->solid);
}

/* --------------------------------------------------------------------------
 *  One LBM step: collide -> stream -> boundary conditions -> macroscopics
 * ------------------------------------------------------------------------ */

static void lbm_step(Lbm *s)
{
    const int N = NX * NY;
    const double omega = 1.0 / s->tau;
    const double U = inlet_speed(s);

    /* (1) COLLIDE in place: relax toward local equilibrium. Pure node-
     * local -> embarrassingly parallel, no neighbour information at all.
     * This is the structural opposite of the MD force loop. */
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        if (s->solid[n]) continue;
        double rho = 0.0, mx = 0.0, my = 0.0;
        for (int k = 0; k < 9; k++) {
            double fk = s->f[k * N + n];
            rho += fk;
            mx  += CX[k] * fk;
            my  += CY[k] * fk;
        }
        double ux = mx / rho, uy = my / rho;
        s->rho[n] = rho; s->ux[n] = ux; s->uy[n] = uy;
        for (int k = 0; k < 9; k++) {
            double fe = feq(k, rho, ux, uy);
            s->f[k * N + n] += omega * (fe - s->f[k * N + n]);
        }
    }

    /* (2) STREAM, pull scheme into ftmp: each fluid node GATHERS, for
     * every direction k, the post-collision population from the upwind
     * neighbour (the node at -c_k). Every (k,node) slot is written
     * exactly once by construction -- no stale/uninitialised slots, the
     * failure mode of a naive push. If that upwind neighbour is solid (or
     * off-grid), halfway bounce-back applies: this node instead receives
     * its OWN opposite population, f_k(n) <- f_{opp(k)}(n), which is the
     * standard no-slip wall at the link midpoint. */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            int n = NID(i, j);
            if (s->solid[n]) continue;
            for (int k = 0; k < 9; k++) {
                int si = i - CX[k];      /* upwind source for dir k */
                int sj = j - CY[k];
                if (si < 0 || si >= NX || sj < 0 || sj >= NY
                    || s->solid[NID(si, sj)]) {
                    s->ftmp[k * N + n] = s->f[OPP[k] * N + n];
                } else {
                    s->ftmp[k * N + n] = s->f[k * N + NID(si, sj)];
                }
            }
        }
    }
    double *sw = s->f; s->f = s->ftmp; s->ftmp = sw;

    /* (3) INLET at i=0: equilibrium velocity boundary, impose (rho0,U,0).
     * Set after streaming so it is the state the next collide sees. The
     * MD code needed particle injection plus a closed upstream wall to
     * stop back-diffusion; here we just fix the moments and nothing can
     * leak upstream because there are no particles -- that whole class of
     * bug does not exist. */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < NY; j++) {
        int n = NID(0, j);
        if (s->solid[n]) continue;
        for (int k = 0; k < 9; k++)
            s->f[k * N + n] = feq(k, s->rho0, U, 0.0);
    }

    /* (4) OUTLET at i=NX-1: convective / zero-gradient. Copy the FULL
     * post-stream distribution from the interior neighbour column. Doing
     * this after streaming (not on the post-collision field) makes it a
     * stable Neumann condition: the wake convects out without reflecting
     * a pressure wave, and no spurious mass is created (the earlier bug
     * was applying this to the post-collision f, which injected mass). */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < NY; j++) {
        int n  = NID(NX - 1, j);
        int nm = NID(NX - 2, j);
        if (s->solid[n] || s->solid[nm]) continue;
        for (int k = 0; k < 9; k++)
            s->f[k * N + n] = s->f[k * N + nm];
    }

    /* (5) Refresh macroscopic fields on the boundary columns so the
     * diagnostics / renderer see the imposed inlet and convected outlet
     * (interior was updated during collide). */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < NY; j++) {
        for (int ii = 0; ii < 2; ii++) {
            int col = ii ? NX - 1 : 0;
            int n = NID(col, j);
            if (s->solid[n]) { s->rho[n] = s->ux[n] = s->uy[n] = 0.0; continue; }
            double rho = 0.0, mx = 0.0, my = 0.0;
            for (int k = 0; k < 9; k++) {
                double fk = s->f[k * N + n];
                rho += fk; mx += CX[k] * fk; my += CY[k] * fk;
            }
            s->rho[n] = rho;
            s->ux[n]  = rho > 1e-12 ? mx / rho : 0.0;
            s->uy[n]  = rho > 1e-12 ? my / rho : 0.0;
        }
    }

    /* (6) TRANSIENT SYMMETRY-BREAKING SEED. While armed and inside its
     * window, add a small one-sided transverse velocity bias to a patch
     * just downstream of the cylinder (lower half only -> breaks the
     * up/down mirror symmetry that otherwise keeps the wake pinned on the
     * unstable steady branch). Implemented by nudging uy in the patch and
     * re-equilibrating those nodes. It is deliberately TRANSIENT: once the
     * Karman mode has grown it sustains itself, and a permanent forcing
     * would contaminate the frequency/amplitude with the forcing itself.
     * Cylinder only -- it is the case with a clean Hopf onset. */
    if (s->perturb && s->nobs > 0 && s->obs[0].type == OBS_CIRCLE
        && s->step >= s->perturb_start
        && s->step <  s->perturb_start + s->perturb_window) {
        double cx = s->obs[0].cx, cy = s->obs[0].cy, rr = s->obs[0].r;
        /* patch ~1 diameter behind the cylinder, lower half only */
        double x0 = cx + 1.0 * rr, x1 = cx + 3.0 * rr;
        double y0 = cy - 1.2 * rr, y1 = cy;
        double bias = 0.02 * U;                    /* small kick */
        int i0 = (int)(x0 * NPS), i1 = (int)(x1 * NPS);
        int j0 = (int)(y0 * NPS), j1 = (int)(y1 * NPS);
        if (i0 < 1)        i0 = 1;
        if (i1 > NX - 2)   i1 = NX - 2;
        if (j0 < 1)        j0 = 1;
        if (j1 > NY - 2)   j1 = NY - 2;
        #pragma omp parallel for schedule(static)
        for (int j = j0; j <= j1; j++)
            for (int i = i0; i <= i1; i++) {
                int n = NID(i, j);
                if (s->solid[n]) continue;
                double rho = s->rho[n];
                double ux = s->ux[n];
                /* HOLD the patch at a small fixed transverse velocity
                 * (a bounded one-sided "blow"), not uy+bias. Adding bias
                 * to the current uy every step compounds geometrically
                 * and drives |u| past the BGK stability limit in a few
                 * hundred steps -- the failure seen in testing. A fixed
                 * small target is a proper bounded asymmetric forcing
                 * that seeds the Hopf mode without runaway. */
                double uy = bias;
                s->uy[n] = uy;
                for (int k = 0; k < 9; k++)
                    s->f[k * N + n] = feq(k, rho, ux, uy);
            }
    }

    s->step++;
}
static double lbm_umax(const Lbm *s)
{
    double m = 0.0;
    for (int n = 0; n < NX * NY; n++) {
        if (s->solid[n]) continue;
        double sp = sqrt(s->ux[n] * s->ux[n] + s->uy[n] * s->uy[n]);
        if (sp > m) m = sp;
    }
    return m;
}

/* ==========================================================================
 *  Headless self-test  (no SDL): used for validation / regression
 * ======================================================================== */

static int run_selftest(void)
{
    Lbm s; lbm_alloc(&s);

    const char *eg = getenv("LB_GEOM");
    const char *eo = getenv("LB_OBS");
    const char *em = getenv("LB_MACH");
    const char *et = getenv("LB_TAU");
    const char *en = getenv("LB_STEPS");
    const char *ep = getenv("LB_PERTURB");
    if (eg) s.geom       = (Geometry)(atoi(eg) % GEOM_COUNT);
    if (eo) s.obs_preset = atoi(eo) % OBS_PRESET_COUNT;
    if (em) s.mach       = atof(em);
    if (et) s.tau        = atof(et);
    int steps = en ? atoi(en) : 2000;
    if (s.tau <= 0.5) s.tau = 0.51;   /* tau>1/2 required for nu>0 */
    if (ep && atoi(ep) && s.obs_preset == 1) {
        s.perturb = 1;
        s.perturb_start = 200;        /* let the steady wake form first */
    }

    rebuild_mask(&s);
    lbm_init_field(&s);

    int nfluid = 0;
    for (int n = 0; n < NX * NY; n++) nfluid += !s.solid[n];

    /* Wake probe for vortex-shedding frequency. Sample transverse
     * velocity uy at a point a few diameters downstream of the obstacle
     * centre, on the centreline. The Karman street makes uy oscillate;
     * its frequency f gives the Strouhal number St = f D / U, a
     * geometry-nondimensional observable (~0.2 for a circular cylinder
     * at moderate Re) that BOTH solvers and the literature can be checked
     * against -- the honest cross-validation, not just "looks similar". */
    double obs_D = 0.0, obs_cx = 0.42 * LX, obs_cy = 0.5 * LY;
    if (s.nobs > 0 && s.obs[0].type == OBS_CIRCLE)
        obs_D = 2.0 * s.obs[0].r;
    int probe_i = (int)((obs_cx + 2.5 * (obs_D > 0 ? obs_D : 20.0)) * NPS);
    int probe_j = (int)(obs_cy * NPS);
    if (probe_i >= NX - 2) probe_i = NX - 2;
    if (probe_i < 1) probe_i = 1;
    double *uy_ts = xmalloc(sizeof(double) * (steps + 1));
    int ts_n = 0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int blew = 0;
    for (int k = 0; k < steps; k++) {
        lbm_step(&s);
        uy_ts[ts_n++] = s.uy[NID(probe_i, probe_j)];
        if (k % 200 == 199) {
            double um = lbm_umax(&s);
            if (getenv("LB_TRACE"))
                printf("    [step %5d] |u|max=%.4f\n", k + 1, um);
            if (!isfinite(um) || um > 0.7) {
                printf("    [DIVERGED at step %d, |u|max=%.3g]\n",
                       k + 1, um);
                blew = 1; break;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);

    /* Strouhal from zero-crossings over the *second half* of the run
     * (first half is startup transient: discard it). Two zero-crossings
     * per shedding period. Dependency-free and robust for a clean
     * single-frequency wake. */
    double St = -1.0, shed_f = -1.0;
    if (!blew && obs_D > 0.0 && ts_n > 200) {
        int h0 = ts_n / 2;
        /* subtract the local mean to centre the signal */
        double mean = 0.0;
        for (int i = h0; i < ts_n; i++) mean += uy_ts[i];
        mean /= (ts_n - h0);
        int crossings = 0; int first = -1, last = -1;
        for (int i = h0 + 1; i < ts_n; i++) {
            double a = uy_ts[i - 1] - mean, b = uy_ts[i] - mean;
            if ((a <= 0.0 && b > 0.0) || (a >= 0.0 && b < 0.0)) {
                if (first < 0) first = i;
                last = i;
                crossings++;
            }
        }
        if (crossings >= 4 && last > first) {
            /* crossings-1 half-periods span (last-first) steps */
            double half_period = (double)(last - first) / (crossings - 1);
            double period = 2.0 * half_period;     /* steps per cycle */
            shed_f = 1.0 / period;                  /* cycles / step  */
            St = shed_f * obs_D / inlet_speed(&s);
        }
    }
    free(uy_ts);

    /* mass + simple field diagnostics */
    double mass = 0.0, umax = 0.0, uxbar = 0.0;
    int cnt = 0;
    for (int n = 0; n < NX * NY; n++) {
        if (s.solid[n]) continue;
        mass += s.rho[n];
        double sp = sqrt(s.ux[n] * s.ux[n] + s.uy[n] * s.uy[n]);
        if (sp > umax) umax = sp;
        uxbar += s.ux[n]; cnt++;
    }
    uxbar /= (cnt ? cnt : 1);

#ifdef _OPENMP
    int th = omp_get_max_threads();
#else
    int th = 1;
#endif
    printf("lbm self-test: geom=%s obstacle=%s\n",
           geom_name(s.geom), obs_name(s.obs_preset));
    printf("  lattice         : %d x %d  (%d fluid nodes)\n",
           NX, NY, nfluid);
    printf("  threads         : %d\n", th);
    printf("  mach (inlet)    : %.3f   tau=%.3f  nu=%.5f\n",
           s.mach, s.tau, CS2 * (s.tau - 0.5));
    printf("  steps           : %d\n", steps);
    printf("  wall time       : %.3f s  (%.0f steps/s, %.2e node-steps/s)\n",
           dt, steps / dt, (double)nfluid * steps / dt);
    printf("  mean ux         : %.5f   (inlet U=%.5f)\n",
           uxbar, inlet_speed(&s));
    printf("  |u|max          : %.5f   (lattice c_s=%.4f)\n",
           umax, sqrt(CS2));
    printf("  total mass      : %.3f   (rho0 * nfluid = %.3f)\n",
           mass, s.rho0 * nfluid);
    if (St > 0.0)
        printf("  wake probe f    : %.3e /step   (zero-cross estimate; "
               "St=%.3f -- NOT noise-robust, see README)\n", shed_f, St);
    else
        printf("  wake probe f    : n/a  (no oscillation detected)\n");
    printf("  status          : %s\n", blew ? "DIVERGED" : "OK");

    lbm_free(&s);
    return blew ? 1 : 0;
}

/* ==========================================================================
 *  Rendering shell  (font / colour map / camera shared with wind_tunnel.c)
 * ======================================================================== */

static const uint8_t FONT[64][5] = {
    {0x00,0x00,0x00,0x00,0x00}, /*   */ {0x00,0x00,0x5F,0x00,0x00}, /* ! */
    {0x00,0x07,0x00,0x07,0x00}, /* " */ {0x14,0x7F,0x14,0x7F,0x14}, /* # */
    {0x24,0x2A,0x7F,0x2A,0x12}, /* $ */ {0x23,0x13,0x08,0x64,0x62}, /* % */
    {0x36,0x49,0x55,0x22,0x50}, /* & */ {0x00,0x05,0x03,0x00,0x00}, /* ' */
    {0x00,0x1C,0x22,0x41,0x00}, /* ( */ {0x00,0x41,0x22,0x1C,0x00}, /* ) */
    {0x14,0x08,0x3E,0x08,0x14}, /* * */ {0x08,0x08,0x3E,0x08,0x08}, /* + */
    {0x00,0x50,0x30,0x00,0x00}, /* , */ {0x08,0x08,0x08,0x08,0x08}, /* - */
    {0x00,0x60,0x60,0x00,0x00}, /* . */ {0x20,0x10,0x08,0x04,0x02}, /* / */
    {0x3E,0x51,0x49,0x45,0x3E}, /* 0 */ {0x00,0x42,0x7F,0x40,0x00}, /* 1 */
    {0x42,0x61,0x51,0x49,0x46}, /* 2 */ {0x21,0x41,0x45,0x4B,0x31}, /* 3 */
    {0x18,0x14,0x12,0x7F,0x10}, /* 4 */ {0x27,0x45,0x45,0x45,0x39}, /* 5 */
    {0x3C,0x4A,0x49,0x49,0x30}, /* 6 */ {0x01,0x71,0x09,0x05,0x03}, /* 7 */
    {0x36,0x49,0x49,0x49,0x36}, /* 8 */ {0x06,0x49,0x49,0x29,0x1E}, /* 9 */
    {0x00,0x36,0x36,0x00,0x00}, /* : */ {0x00,0x56,0x36,0x00,0x00}, /* ; */
    {0x08,0x14,0x22,0x41,0x00}, /* < */ {0x14,0x14,0x14,0x14,0x14}, /* = */
    {0x00,0x41,0x22,0x14,0x08}, /* > */ {0x02,0x01,0x51,0x09,0x06}, /* ? */
    {0x32,0x49,0x79,0x41,0x3E}, /* @ */ {0x7E,0x11,0x11,0x11,0x7E}, /* A */
    {0x7F,0x49,0x49,0x49,0x36}, /* B */ {0x3E,0x41,0x41,0x41,0x22}, /* C */
    {0x7F,0x41,0x41,0x22,0x1C}, /* D */ {0x7F,0x49,0x49,0x49,0x41}, /* E */
    {0x7F,0x09,0x09,0x09,0x01}, /* F */ {0x3E,0x41,0x49,0x49,0x7A}, /* G */
    {0x7F,0x08,0x08,0x08,0x7F}, /* H */ {0x00,0x41,0x7F,0x41,0x00}, /* I */
    {0x20,0x40,0x41,0x3F,0x01}, /* J */ {0x7F,0x08,0x14,0x22,0x41}, /* K */
    {0x7F,0x40,0x40,0x40,0x40}, /* L */ {0x7F,0x02,0x0C,0x02,0x7F}, /* M */
    {0x7F,0x04,0x08,0x10,0x7F}, /* N */ {0x3E,0x41,0x41,0x41,0x3E}, /* O */
    {0x7F,0x09,0x09,0x09,0x06}, /* P */ {0x3E,0x41,0x51,0x21,0x5E}, /* Q */
    {0x7F,0x09,0x19,0x29,0x46}, /* R */ {0x46,0x49,0x49,0x49,0x31}, /* S */
    {0x01,0x01,0x7F,0x01,0x01}, /* T */ {0x3F,0x40,0x40,0x40,0x3F}, /* U */
    {0x1F,0x20,0x40,0x20,0x1F}, /* V */ {0x7F,0x20,0x18,0x20,0x7F}, /* W */
    {0x63,0x14,0x08,0x14,0x63}, /* X */ {0x07,0x08,0x70,0x08,0x07}, /* Y */
    {0x61,0x51,0x49,0x45,0x43}, /* Z */ {0x00,0x7F,0x41,0x41,0x00}, /* [ */
    {0x02,0x04,0x08,0x10,0x20}, /* \ */ {0x00,0x41,0x41,0x7F,0x00}, /* ] */
    {0x04,0x02,0x01,0x02,0x04}, /* ^ */ {0x40,0x40,0x40,0x40,0x40}, /* _ */
};

typedef struct {
    SDL_Window   *win;
    SDL_Renderer *ren;
    int W, H;
    double scale, cam_x, cam_y;
    int show_hud;
    int show_legend;
    int color_by;            /* 0 speed, 1 vorticity */
    SDL_Texture *ftex;
    uint8_t     *ftpx;
    int          ftw, fth;
    /* video recording (pipe raw RGB24 to ffmpeg), same as the MD shell */
    FILE   *ff;
    uint8_t *frame;
    int      rec;
} View;

/* --- video recording: pipe raw RGB24 frames to ffmpeg --------------------
 * Verbatim port of the MD shell's recorder, only the output filename
 * prefix differs (lbm_tunnel_*) so a side-by-side capture session does
 * not have the two programs overwrite each other's clips. */
#ifdef __EMSCRIPTEN__
/* WASM: ffmpeg piping is impossible in the browser. The V key becomes
 * a no-op; a future revision could emit WebM via MediaRecorder from a
 * JS shim, but that is non-trivial and this port is "run in browser",
 * not "record in browser". */
static void rec_start(View *v) { (void)v; }
static void rec_stop (View *v) { (void)v; }
static void rec_frame(View *v) { (void)v; }
#else
static void rec_start(View *v)
{
    if (v->rec) return;
    time_t tt = time(NULL);
    struct tm *lt = localtime(&tt);
    char fname[128];
    strftime(fname, sizeof fname, "lbm_tunnel_%Y%m%d_%H%M%S.mp4", lt);
    char cmd[512];
    snprintf(cmd, sizeof cmd,
        "ffmpeg -hide_banner -loglevel error -y -f rawvideo -pixel_format rgb24 "
        "-video_size %dx%d -framerate 60 -i pipe:0 "
        "-an -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \"%s\"",
        v->W, v->H, fname);
    v->ff = popen(cmd, "w");
    if (!v->ff) { fprintf(stderr, "could not start ffmpeg\n"); return; }
    if (!v->frame) v->frame = xmalloc((size_t)v->W * v->H * 3);
    v->rec = 1;
    fprintf(stderr, "recording -> %s\n", fname);
}

static void rec_stop(View *v)
{
    if (!v->rec) return;
    v->rec = 0;
    if (v->ff) { pclose(v->ff); v->ff = NULL; }
    fprintf(stderr, "recording stopped\n");
}

static void rec_frame(View *v)
{
    if (!v->rec || !v->ff) return;
    if (SDL_RenderReadPixels(v->ren, NULL, SDL_PIXELFORMAT_RGB24,
                             v->frame, v->W * 3) != 0) return;
    fwrite(v->frame, 1, (size_t)v->W * v->H * 3, v->ff);
}
#endif

/* On-screen draggable sliders, identical look/behaviour to the MD code so
 * the two windows feel the same. For LBM the live knobs are the inlet
 * MACH and the relaxation time TAU (= viscosity = Reynolds number). */
typedef struct {
    int x, y, w, h;
    double mn, mx;
    double *val;
    const char *label;
    int integer;
} Slider;

static double w2sx(const View *v, double x) { return (x - v->cam_x) * v->scale; }
static double w2sy(const View *v, double y) { return v->H - (y - v->cam_y) * v->scale; }

static void draw_char(View *v, int sx, int sy, char ch, int px,
                      Uint8 r, Uint8 g, Uint8 b)
{
    if (ch >= 'a' && ch <= 'z') ch = (char)(ch - 32);
    if (ch < 0x20 || ch > 0x5F) ch = 0x20;
    const uint8_t *gl = FONT[(int)ch - 0x20];
    SDL_SetRenderDrawColor(v->ren, r, g, b, 255);
    for (int col = 0; col < 5; col++)
        for (int row = 0; row < 7; row++)
            if (gl[col] & (1 << row)) {
                SDL_Rect q = { sx + col * px, sy + row * px, px, px };
                SDL_RenderFillRect(v->ren, &q);
            }
}

static void draw_text(View *v, int sx, int sy, const char *t, int px,
                       Uint8 r, Uint8 g, Uint8 b)
{
    int cx = sx;
    for (const char *p = t; *p; p++) {
        if (*p == '\n') { sy += 8 * px; cx = sx; continue; }
        draw_char(v, cx, sy, *p, px, r, g, b);
        cx += 6 * px;
    }
}

static void draw_slider(View *v, const Slider *sl)
{
    SDL_SetRenderDrawColor(v->ren, 40, 44, 54, 255);
    SDL_Rect tr = { sl->x, sl->y, sl->w, sl->h };
    SDL_RenderFillRect(v->ren, &tr);
    double f = (*sl->val - sl->mn) / (sl->mx - sl->mn);
    if (f < 0) f = 0;
    if (f > 1) f = 1;
    SDL_SetRenderDrawColor(v->ren, 70, 130, 190, 255);
    SDL_Rect fr = { sl->x, sl->y, (int)(sl->w * f), sl->h };
    SDL_RenderFillRect(v->ren, &fr);
    SDL_SetRenderDrawColor(v->ren, 200, 210, 230, 255);
    SDL_Rect hr = { sl->x + (int)(sl->w * f) - 2, sl->y - 2, 4, sl->h + 4 };
    SDL_RenderFillRect(v->ren, &hr);
    char buf[96];
    if (sl->integer)
        snprintf(buf, sizeof buf, "%s %d", sl->label, (int)(*sl->val));
    else
        snprintf(buf, sizeof buf, "%s %.3f", sl->label, *sl->val);
    draw_text(v, sl->x, sl->y - 14, buf, 1, 210, 215, 225);
}

/* Returns 1 if a slider was actually grabbed/moved (so the caller can
 * re-initialise the lattice -- an LBM distribution is consistent with
 * only one parameter set, unlike the MD inlet which can change live). */
static int slider_drag(Slider *sl, int n, int mx, int my)
{
    int hit = 0;
    for (int i = 0; i < n; i++) {
        Slider *s = &sl[i];
        if (mx >= s->x - 4 && mx <= s->x + s->w + 4 &&
            my >= s->y - 6 && my <= s->y + s->h + 6) {
            double f = (double)(mx - s->x) / s->w;
            if (f < 0) f = 0;
            if (f > 1) f = 1;
            double val = s->mn + f * (s->mx - s->mn);
            if (s->integer) val = floor(val + 0.5);
            *s->val = val;
            hit = 1;
        }
    }
    return hit;
}

/* identical blue->cyan->green->yellow->red map as the MD code */
static void speed_color(double t, Uint8 *r, Uint8 *g, Uint8 *b)
{
    if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
    double x = t * 4.0;
    double R, G, B;
    if      (x < 1.0) { R = 0;     G = x;     B = 1; }
    else if (x < 2.0) { R = 0;     G = 1;     B = 2 - x; }
    else if (x < 3.0) { R = x - 2; G = 1;     B = 0; }
    else              { R = 1;     G = 4 - x; B = 0; }
    *r = (Uint8)(R * 255); *g = (Uint8)(G * 255); *b = (Uint8)(B * 255);
}

/* diverging blue-white-red map for signed vorticity, t in [-1,1] */
static void vort_color(double t, Uint8 *r, Uint8 *g, Uint8 *b)
{
    if (t < -1.0) t = -1.0; else if (t > 1.0) t = 1.0;
    if (t >= 0.0) { *r = 255; *g = (Uint8)(255 * (1 - t)); *b = (Uint8)(255 * (1 - t)); }
    else          { *b = 255; *g = (Uint8)(255 * (1 + t)); *r = (Uint8)(255 * (1 + t)); }
}

static void field_init(View *v)
{
    v->ftw = NX;
    v->fth = NY;
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
    v->ftex = SDL_CreateTexture(v->ren, SDL_PIXELFORMAT_RGBA8888,
                                SDL_TEXTUREACCESS_STREAMING, NX, NY);
    if (v->ftex) SDL_SetTextureBlendMode(v->ftex, SDL_BLENDMODE_BLEND);
    v->ftpx = xmalloc((size_t)NX * NY * 4);
}

/* Build the field texture. LBM's macroscopic fields are sharp at every
 * node every step -- no time-averaging, no warm-up, zero statistical
 * noise (the MD code had to reconstruct this by binning + EMA). */
static void draw_field(View *v, const Lbm *s)
{
    double U = inlet_speed(s);
    double inv = 1.0 / (1.8 * (U > 1e-6 ? U : 1e-6));
    /* vorticity scale: a few * U / dx is a reasonable full-scale */
    double wref = 1.0 / (3.0 * U / (1.0 / NPS) + 1e-9);

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            int n = NID(i, j);
            /* texture row 0 = top of screen; world y increases upward */
            int trow = NY - 1 - j;
            uint32_t *px = (uint32_t *)&v->ftpx[((size_t)trow * NX + i) * 4];
            if (s->solid[n]) { *px = 0; continue; }
            Uint8 r, g, b;
            if (v->color_by == 0) {
                double sp = sqrt(s->ux[n]*s->ux[n] + s->uy[n]*s->uy[n]);
                speed_color(sp * inv, &r, &g, &b);
            } else {
                /* central-difference vorticity w = dv/dx - du/dy */
                int im = i>0?i-1:i, ip = i<NX-1?i+1:i;
                int jm = j>0?j-1:j, jp = j<NY-1?j+1:j;
                double dvdx = (s->uy[NID(ip,j)] - s->uy[NID(im,j)]);
                double dudy = (s->ux[NID(i,jp)] - s->ux[NID(i,jm)]);
                double w = (dvdx - dudy) * 0.5 * NPS;
                vort_color(w * wref, &r, &g, &b);
            }
            *px = ((uint32_t)r << 24) | ((uint32_t)g << 16)
                | ((uint32_t)b << 8)  | 235u;
        }
    }
    SDL_UpdateTexture(v->ftex, NULL, v->ftpx, NX * 4);

    SDL_Rect dst;
    dst.x = (int)w2sx(v, 0.0);
    dst.y = (int)w2sy(v, LY);
    dst.w = (int)w2sx(v, LX) - dst.x;
    dst.h = (int)w2sy(v, 0.0) - dst.y;
    SDL_RenderCopy(v->ren, v->ftex, NULL, &dst);
}

/* Colour legend. The two modes need different legends: SPEED is a one-
 * sided blue->red ramp (0 .. 1.8 U), VORTICITY is a diverging blue->white
 * ->red scale (-wmax .. +wmax, white = irrotational). Ticks are labelled
 * with the actual physical values, not just "0" and "max", so red is no
 * longer a mystery. */
static void draw_legend(View *v, const Lbm *s)
{
    int x = v->W - 210, y = 56, w = 180, h = 16;
    double U = inlet_speed(s);

    if (v->color_by == 0) {
        for (int i = 0; i < w; i++) {
            Uint8 r, g, b; speed_color((double)i / w, &r, &g, &b);
            SDL_SetRenderDrawColor(v->ren, r, g, b, 255);
            SDL_RenderDrawLine(v->ren, x + i, y, x + i, y + h);
        }
        draw_text(v, x, y - 14, "FLOW SPEED |U|", 1, 215, 220, 230);
        /* full scale is 1.8 U; mark 0, the inlet speed U, and 1.8 U */
        char b0[24], bU[24], bM[24];
        snprintf(b0, sizeof b0, "0");
        snprintf(bU, sizeof bU, "U=%.3f", U);
        snprintf(bM, sizeof bM, "%.3f", 1.8 * U);
        draw_text(v, x, y + h + 4, b0, 1, 200, 205, 215);
        /* U sits at fraction 1/1.8 along the bar */
        int ux = x + (int)(w / 1.8) - 14;
        draw_text(v, ux, y + h + 4, bU, 1, 200, 205, 215);
        SDL_SetRenderDrawColor(v->ren, 235, 238, 245, 255);
        SDL_RenderDrawLine(v->ren, x + (int)(w / 1.8), y - 2,
                                   x + (int)(w / 1.8), y + h + 2);
        draw_text(v, x + w - 36, y + h + 4, bM, 1, 200, 205, 215);
        draw_text(v, x, y + h + 16,
                  "BLUE SLOW  RED FAST (LATTICE UNITS)", 1, 150, 158, 172);
    } else {
        for (int i = 0; i < w; i++) {
            double t = 2.0 * ((double)i / w) - 1.0;   /* -1 .. +1 */
            Uint8 r, g, b; vort_color(t, &r, &g, &b);
            SDL_SetRenderDrawColor(v->ren, r, g, b, 255);
            SDL_RenderDrawLine(v->ren, x + i, y, x + i, y + h);
        }
        draw_text(v, x, y - 14, "VORTICITY (SIGN)", 1, 215, 220, 230);
        /* full scale is +/- 3 U / dx (dx = 1/NPS); white = zero */
        double wmax = 3.0 * U * NPS;
        char bn[24], bp[24];
        snprintf(bn, sizeof bn, "-%.2f CW", wmax);
        snprintf(bp, sizeof bp, "+%.2f CCW", wmax);
        draw_text(v, x, y + h + 4, bn, 1, 150, 180, 235);
        draw_text(v, x + w/2 - 8, y + h + 4, "0", 1, 210, 210, 210);
        draw_text(v, x + w - 56, y + h + 4, bp, 1, 235, 170, 170);
        draw_text(v, x, y + h + 16,
                  "BLUE CW  WHITE IRROT.  RED CCW", 1, 150, 158, 172);
    }
}

static void draw_walls(View *v, const Lbm *s)
{
    SDL_SetRenderDrawColor(v->ren, 90, 110, 140, 255);
    int have = 0, px0 = 0, pyb = 0, pyt = 0;
    for (int p = 0; p <= 240; p++) {
        double wx = LX * p / 240.0;
        double yb, yt; geom_bounds(s->geom, wx, &yb, &yt);
        int sx = (int)w2sx(v, wx);
        int syb = (int)w2sy(v, yb);
        int syt = (int)w2sy(v, yt);
        if (have) {
            SDL_RenderDrawLine(v->ren, px0, pyb, sx, syb);
            SDL_RenderDrawLine(v->ren, px0, pyt, sx, syt);
        }
        px0 = sx; pyb = syb; pyt = syt; have = 1;
    }
    for (int o = 0; o < s->nobs; o++) {
        const Obstacle *ob = &s->obs[o];
        if (ob->type == OBS_CIRCLE) {
            int steps = 64;
            int ox = 0, oy = 0, fx = 0, fy = 0;
            for (int t = 0; t <= steps; t++) {
                double a = 2.0 * PI * t / steps;
                int X = (int)w2sx(v, ob->cx + ob->r * cos(a));
                int Y = (int)w2sy(v, ob->cy + ob->r * sin(a));
                if (t) SDL_RenderDrawLine(v->ren, ox, oy, X, Y);
                else { fx = X; fy = Y; }
                ox = X; oy = Y;
            }
            (void)fx; (void)fy;
        } else {
            for (int e = 0; e < ob->nv; e++) {
                int f = (e + 1) % ob->nv;
                SDL_RenderDrawLine(v->ren,
                    (int)w2sx(v, ob->px[e]), (int)w2sy(v, ob->py[e]),
                    (int)w2sx(v, ob->px[f]), (int)w2sy(v, ob->py[f]));
            }
        }
    }
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    if (getenv("LB_SELFTEST")) return run_selftest();

    Lbm s; lbm_alloc(&s);

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    View v; memset(&v, 0, sizeof v);
    v.W = WIN_W; v.H = WIN_H;
    v.win = SDL_CreateWindow("LBM wind tunnel (D2Q9-BGK)",
                SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                v.W, v.H, SDL_WINDOW_SHOWN);
    if (!v.win) { fprintf(stderr, "window: %s\n", SDL_GetError()); return 1; }
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
    v.ren = SDL_CreateRenderer(v.win, -1,
                SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!v.ren) { fprintf(stderr, "renderer: %s\n", SDL_GetError()); return 1; }

    v.scale = (v.W - 60) / LX;
    v.cam_x = -30.0 / v.scale;
    v.cam_y = -(v.H - LY * v.scale) * 0.5 / v.scale;
    v.show_hud = 1;
    v.show_legend = 1;
    v.color_by = 0;
    field_init(&v);

    int running = 1, quit = 0, substeps = 8;
    double fps = 0.0; int fps_c = 0; Uint32 fps_t0 = SDL_GetTicks();

    /* Two physical knobs for LBM: inlet Mach and relaxation time tau
     * (viscosity / Reynolds number). Same widget as the MD sliders. */
    Slider sl[2] = {
        { 16,  v.H - 96, 220, 12, 0.00, 0.55, &s.mach, "INLET MACH", 0 },
        { 16,  v.H - 64, 220, 12, 0.51, 1.50, &s.tau,  "TAU (visc)", 0 },
    };
    int dragging_slider = 0, sl_dirty = 0;

    while (!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = 1;
            else if (e.type == SDL_KEYDOWN) {
                SDL_Keycode k = e.key.keysym.sym;
                if (k == SDLK_ESCAPE || k == SDLK_q) quit = 1;
                else if (k == SDLK_SPACE) running = !running;
                else if (k == SDLK_r) {
                    rebuild_mask(&s); lbm_init_field(&s);
                }
                else if (k == SDLK_g) {
                    s.geom = (Geometry)((s.geom + 1) % GEOM_COUNT);
                    rebuild_mask(&s); lbm_init_field(&s);
                }
                else if (k == SDLK_o) {
                    s.obs_preset = (s.obs_preset + 1) % OBS_PRESET_COUNT;
                    rebuild_mask(&s); lbm_init_field(&s);
                }
                else if (k == SDLK_c) {
                    s.obs_preset = 0; rebuild_mask(&s); lbm_init_field(&s);
                }
                else if (k == SDLK_EQUALS || k == SDLK_PLUS
                         || k == SDLK_KP_PLUS) {
                    s.obs_scale *= 1.12;
                    if (s.obs_scale > 6.0) s.obs_scale = 6.0;
                    rebuild_mask(&s); lbm_init_field(&s);
                }
                else if (k == SDLK_MINUS || k == SDLK_KP_MINUS) {
                    s.obs_scale /= 1.12;
                    if (s.obs_scale < 0.15) s.obs_scale = 0.15;
                    rebuild_mask(&s); lbm_init_field(&s);
                }
                else if (k == SDLK_0) {
                    s.obs_scale = 1.0;
                    rebuild_mask(&s); lbm_init_field(&s);
                }
                else if (k == SDLK_LEFTBRACKET)  { if (substeps > 1) substeps--; }
                else if (k == SDLK_RIGHTBRACKET) { if (substeps < 60) substeps++; }
                else if (k == SDLK_UP) {
                    s.mach += 0.02;  if (s.mach > 1.2) s.mach = 1.2;
                }
                else if (k == SDLK_DOWN) {
                    s.mach -= 0.02;  if (s.mach < 0.0) s.mach = 0.0;
                }
                else if (k == SDLK_f) v.color_by = !v.color_by;
                else if (k == SDLK_p) {
                    /* arm a fresh transient seed from the current step;
                     * pressing again before it expires disarms it */
                    if (s.perturb) {
                        s.perturb = 0;
                    } else if (s.obs_preset == 1) {
                        s.perturb = 1;
                        s.perturb_start = s.step;
                    }
                }
                else if (k == SDLK_v) {
                    if (v.rec) rec_stop(&v); else rec_start(&v);
                }
                else if (k == SDLK_l) v.show_legend = !v.show_legend;
                else if (k == SDLK_h) v.show_hud = !v.show_hud;
            }
            else if (e.type == SDL_MOUSEBUTTONDOWN) {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    if (slider_drag(sl, 2, e.button.x, e.button.y)) {
                        dragging_slider = 1; sl_dirty = 1;
                    }
                }
            }
            else if (e.type == SDL_MOUSEBUTTONUP) {
                if (dragging_slider && sl_dirty) {
                    /* parameter changed -> the old distribution is no
                     * longer consistent; restart the field cleanly */
                    if (s.tau <= 0.5) s.tau = 0.51;
                    rebuild_mask(&s); lbm_init_field(&s);
                    running = 1; sl_dirty = 0;
                }
                dragging_slider = 0;
            }
            else if (e.type == SDL_MOUSEMOTION) {
                if (dragging_slider) {
                    slider_drag(sl, 2, e.motion.x, e.motion.y);
                    sl_dirty = 1;
                }
            }
        }

        int diverged = 0;
        if (running) {
            for (int t = 0; t < substeps; t++) {
                lbm_step(&s);
            }
            double um = lbm_umax(&s);
            if (!isfinite(um) || um > 0.7) { diverged = 1; running = 0; }
        }

        SDL_SetRenderDrawColor(v.ren, 16, 18, 24, 255);
        SDL_RenderClear(v.ren);
        draw_field(&v, &s);
        draw_walls(&v, &s);
        for (int i = 0; i < 2; i++) draw_slider(&v, &sl[i]);
        if (v.show_legend) draw_legend(&v, &s);

        if (v.show_hud) {
            fps_c++;
            Uint32 now = SDL_GetTicks();
            if (now - fps_t0 > 500) {
                fps = 1000.0 * fps_c / (now - fps_t0);
                fps_c = 0; fps_t0 = now;
            }
#ifdef _OPENMP
            int nth = omp_get_max_threads();
#else
            int nth = 1;
#endif
            double um = lbm_umax(&s);
            int have_D = 0;
            double Re = reynolds(&s, &have_D);
            int perturb_active = (s.perturb && s.step >= s.perturb_start
                && s.step < s.perturb_start + s.perturb_window);
            char restr[40];
            if (have_D)
                snprintf(restr, sizeof restr, "Re %.0f (cyl D)", Re);
            else
                snprintf(restr, sizeof restr, "Re %.0f (duct)", Re);
            char pstr[44];
            if (perturb_active) {
                long left = s.perturb_start + s.perturb_window - s.step;
                snprintf(pstr, sizeof pstr,
                         "   SEED ON (%ld steps left)", left);
            } else if (s.perturb) {
                snprintf(pstr, sizeof pstr, "   SEED SPENT");
            } else {
                pstr[0] = '\0';
            }
            char hud[640];
            snprintf(hud, sizeof hud,
                "LBM D2Q9-BGK   %d x %d   THREADS %d   FPS %.0f   SUB/FR %d\n"
                "MACH %.2f   TAU %.3f   NU %.4f   %s   |U|MAX %.3f / cs %.3f\n"
                "GEOM %s   OBSTACLE %s x%.2f   COLOR %s   %s%s%s",
                NX, NY, nth, fps, substeps,
                s.mach, s.tau, CS2 * (s.tau - 0.5), restr, um, sqrt(CS2),
                geom_name(s.geom), obs_name(s.obs_preset), s.obs_scale,
                v.color_by ? "VORTICITY" : "SPEED",
                running ? "RUNNING " : "STOPPED ",
                diverged ? " *DIVERGED: lower MACH / raise TAU*" : "",
                pstr);
            draw_text(&v, 14, 12, hud, 2, 220, 225, 235);
            if (v.rec) draw_text(&v, v.W - 90, 12, "*REC*", 2,
                                 235, 80, 80);
            draw_text(&v, 14, v.H - 26,
                "DRAG SLIDERS: MACH / TAU   SPACE RUN  R RESET  G GEOM  "
                "O OBSTACLE  +/- SIZE  P SEED-WAKE  V REC  [ ] SPEED  "
                "F SPEED/VORT  L LEGEND  H HUD  Q QUIT",
                1, 150, 160, 180);
        }

        if (v.rec) rec_frame(&v);
        SDL_RenderPresent(v.ren);
#ifdef __EMSCRIPTEN__
        /* Yield to the browser (ASYNCIFY). Without this, the while
         * loop never returns and the page hangs. 1 ms is enough; the
         * actual pacing comes from SDL_RenderPresent on VSYNC. */
        emscripten_sleep(1);
#endif
    }

    if (v.rec) rec_stop(&v);
    free(v.ftpx);
    free(v.frame);
    if (v.ftex) SDL_DestroyTexture(v.ftex);
    SDL_DestroyRenderer(v.ren);
    SDL_DestroyWindow(v.win);
    SDL_Quit();
    lbm_free(&s);
    return 0;
}
