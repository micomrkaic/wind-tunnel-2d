/* ===========================================================================
 *  wind_tunnel.c  --  2D molecular-dynamics wind tunnel
 *
 *  A "gas" of pseudo-hard spheres enters from the left with a controllable
 *  drift (set as a Mach number), flows through a shaped tunnel past
 *  obstacles, and exits on the right. The dynamics are real molecular
 *  dynamics: velocity-Verlet integration, a linked cell list, and a
 *  rebuilt-on-demand Verlet neighbour list, with a stiff purely-repulsive
 *  WCA core standing in for the hard sphere (true hard spheres are
 *  event-driven and cannot be Verlet-integrated -- see README).
 *
 *  Reduced units: sigma = m = epsilon = k_B = 1.
 *
 *  Controls
 *    SPACE  start / stop the experiment (start = reseed the tunnel)
 *    P      pause / resume (keeps particles)
 *    V      start / stop video recording (pipes to ffmpeg)
 *    R      reset / clear the tunnel
 *    G      cycle tunnel geometry
 *    O      cycle obstacle
 *    C      clear user-placed obstacles
 *    M      cycle particle render mode (points / discs)
 *    L      toggle speed-colour legend
 *    H      toggle HUD
 *    mouse wheel      zoom (about cursor)
 *    right/middle drag pan
 *    left click       place a circular obstacle at the cursor
 *    left drag slider adjust a parameter
 *    ESC / Q          quit
 *
 *  Headless self-test / benchmark (no window, no SDL video):
 *    WT_SELFTEST=1 [WT_STEPS=n] [WT_N hint] ./wind_tunnel
 * ===========================================================================
 */

#define _POSIX_C_SOURCE 200809L

#include <SDL.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* --------------------------------------------------------------------------
 *  Configuration
 * ------------------------------------------------------------------------ */

typedef double real;

#define PI 3.14159265358979323846

#define WIN_W 1280
#define WIN_H 720

#define LX 320.0          /* tunnel length  (sigma) */
#define LY 100.0          /* domain height  (sigma) */

#define CAP 320000        /* particle array capacity */

#define RCUT  1.122462048309373   /* 2^(1/6): WCA cutoff */
#define SKIN  0.45                /* Verlet skin           */
#define RVERL (RCUT + SKIN)       /* neighbour-list cutoff */

#define MAX_OBSTACLES 16
#define MAX_POLY_V    64
#define WALL_EPS 1.0e-3

/* --------------------------------------------------------------------------
 *  Fast PRNG (xoshiro256**) + Gaussian
 * ------------------------------------------------------------------------ */

static uint64_t rng_s[4];

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_seed(uint64_t seed)
{
    uint64_t s = seed ? seed : 0x12345678ULL;
    for (int i = 0; i < 4; i++) rng_s[i] = splitmix64(&s);
}

static uint64_t rng_next(void)
{
    const uint64_t r = rotl(rng_s[1] * 5, 7) * 9;
    const uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0];
    rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2];
    rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t;
    rng_s[3] = rotl(rng_s[3], 45);
    return r;
}

/* uniform in [0,1) */
static inline real urand(void)
{
    return (real)((rng_next() >> 11) * (1.0 / 9007199254740992.0));
}

/* standard normal (Marsaglia polar) */
static real grand(void)
{
    static int have = 0;
    static real cached = 0.0;
    if (have) { have = 0; return cached; }
    real u, v, s;
    do {
        u = 2.0 * urand() - 1.0;
        v = 2.0 * urand() - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    real m = sqrt(-2.0 * log(s) / s);
    cached = v * m; have = 1;
    return u * m;
}

/* --------------------------------------------------------------------------
 *  Geometry: tunnel profile (smooth) and obstacles (circles / polygons)
 * ------------------------------------------------------------------------ */

typedef enum {
    GEOM_STRAIGHT = 0,
    GEOM_NOZZLE,        /* converging-diverging (de Laval) */
    GEOM_VENTURI,       /* symmetric throat                */
    GEOM_DIFFUSER,      /* gradual expansion               */
    GEOM_BUMP,          /* transonic bump on the floor     */
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

/* Half-height h(x) of the tunnel about the centreline LY/2. */
static real geom_halfheight(Geometry g, real x)
{
    real hwide = 0.40 * LY;     /* wide section half-height  */
    real hthr  = 0.17 * LY;     /* throat half-height        */
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
            return (0.55 * hthr + 0.45 * hwide) + (hwide - (0.55 * hthr + 0.45 * hwide)) * s
                   + (hthr - 0.55 * hthr) * 0.0;
        }
        case GEOM_BUMP:
        default:
            return hwide;
    }
}

/* Lower / upper wall y at column x. */
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

/* --- obstacles --- */

typedef enum { OBS_CIRCLE = 0, OBS_POLY } ObsType;

typedef struct {
    ObsType type;
    /* circle */
    real cx, cy, r;
    /* polygon (CCW) */
    int   nv;
    real  px[MAX_POLY_V], py[MAX_POLY_V];
    real  nx[MAX_POLY_V], ny[MAX_POLY_V];  /* outward edge normals */
    /* bounding box */
    real  bbminx, bbminy, bbmaxx, bbmaxy;
} Obstacle;

static void poly_finalize(Obstacle *o)
{
    o->type = OBS_POLY;
    o->bbminx = o->bbminy =  1e30;
    o->bbmaxx = o->bbmaxy = -1e30;

    /* Signed area (shoelace). >0 == CCW, <0 == CW. We must NOT assume a
     * winding order: make_wing traces the airfoil clockwise while the
     * rectangle/wedge are counter-clockwise. The CCW outward-normal
     * formula (dy,-dx) points *inward* for a CW polygon, which would
     * reflect every entering particle the wrong way and pump unbounded
     * spurious energy into the gas. Orient from the actual signed area. */
    real area2 = 0.0;
    for (int i = 0; i < o->nv; i++) {
        int j = (i + 1) % o->nv;
        area2 += o->px[i] * o->py[j] - o->px[j] * o->py[i];
    }
    real sgn = (area2 >= 0.0) ? 1.0 : -1.0;   /* +1 CCW, -1 CW */

    for (int i = 0; i < o->nv; i++) {
        int j = (i + 1) % o->nv;
        real dx = o->px[j] - o->px[i];
        real dy = o->py[j] - o->py[i];
        real L  = sqrt(dx * dx + dy * dy);
        if (L < 1e-12) L = 1e-12;
        /* outward normal, winding-corrected */
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
    /* CCW */
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
    /* wedge pointing upstream (apex to the left), CCW */
    o.px[0] = cx - 0.6 * len; o.py[0] = cy;
    o.px[1] = cx + 0.4 * len; o.py[1] = cy - half;
    o.px[2] = cx + 0.4 * len; o.py[2] = cy + half;
    poly_finalize(&o);
    return o;
}

/* Symmetric NACA 00xx airfoil polygon, chord c, thickness frac tt,
 * rotated by angle of attack aoa (radians), centred near (cx,cy). */
static Obstacle make_wing(real cx, real cy, real c, real tt, real aoa)
{
    Obstacle o; memset(&o, 0, sizeof o);
    const int np = 24;                 /* points per surface */
    real ca = cos(-aoa), sa = sin(-aoa);
    int k = 0;
    /* upper surface leading->trailing, then lower trailing->leading */
    for (int i = 0; i <= np && k < MAX_POLY_V; i++) {
        real xc = 0.5 * (1.0 - cos(PI * (real)i / (real)np)); /* cosine cluster */
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

/* nearest point of segment a->b to p, returned in (qx,qy) */
static void seg_nearest(real ax, real ay, real bx, real by,
                        real px, real py, real *qx, real *qy)
{
    real dx = bx - ax, dy = by - ay;
    real L2 = dx * dx + dy * dy;
    real t = (L2 > 1e-18) ? ((px - ax) * dx + (py - ay) * dy) / L2 : 0.0;
    if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
    *qx = ax + t * dx;
    *qy = ay + t * dy;
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

/* --------------------------------------------------------------------------
 *  Simulation state
 * ------------------------------------------------------------------------ */

typedef struct {
    /* particle SoA */
    int   n;
    real *x, *y, *vx, *vy, *ax, *ay;
    real *x0, *y0;            /* positions at last neighbour rebuild */

    /* parameters (live, slider-driven) */
    real density;             /* number density at inlet (sigma^-2) */
    real mach;                /* inlet drift / speed of sound        */
    real temp;                /* temperature (k_B = 1)               */
    real psize;               /* render size scale                   */
    real dt;                  /* time step (auto-capped)             */

    Geometry geom;

    Obstacle obs[MAX_OBSTACLES];   /* active set = preset + user list */
    int nobs;
    int obs_preset;           /* which built-in obstacle is active   */
    double obs_scale;         /* size multiplier for the preset       */
    Obstacle user_obs[MAX_OBSTACLES]; /* mouse-placed, persistent     */
    int n_user;

    /* cell list */
    int   ncx, ncy, ncell;
    real  cell;
    int  *cell_head;          /* ncell             */
    int  *cell_next;          /* CAP               */

    /* Verlet neighbour list (full, CSR) */
    int  *nbr_start;          /* CAP+1             */
    int  *nbr;                /* nbr_cap           */
    long  nbr_cap;
    int   need_rebuild;

    /* bookkeeping */
    double sim_time;
    long   step;
    double inject_carry;
} Sim;

static real speed_of_sound(const Sim *s)
{
    /* 2D monatomic ideal gas: gamma = (d+2)/d = 2 */
    return sqrt(2.0 * s->temp);
}

/* time step capped so the fastest expected particle moves < ~0.12 sigma */
static void update_dt(Sim *s)
{
    real vth = sqrt(s->temp);
    real u   = s->mach * speed_of_sound(s);
    real vref = u + 6.0 * vth + 1e-6;

    /* Two independent stability limits; take the smaller.
     *
     * (1) Kinematic: the fastest particle must not fly more than ~0.1 sigma
     *     per step or it can tunnel through a core between neighbour-list
     *     rebuilds.
     *
     * (2) Stiffness: velocity-Verlet is stable only for dt below the
     *     fastest oscillation period of the potential. The WCA force grows
     *     as r^-13, so the effective spring constant at a contact of depth
     *     delta is enormous. Empirically a contact at r ~ 0.9 sigma has an
     *     effective curvature k_eff ~ U''(r) of order ~3000 in reduced
     *     units; velocity-Verlet needs dt < 2/sqrt(k_eff) for stability and
     *     dt < ~0.1/sqrt(k_eff) for accuracy. Without this term a dense
     *     gas (small spacing -> deep transient overlaps) detonates on the
     *     first step regardless of how slow the drift is. */
    real dt_kin = 0.10 / vref;

    real k_eff  = 3000.0;                 /* WCA curvature scale, reduced  */
    real dt_stiff = 0.10 / sqrt(k_eff);   /* ~1.8e-3                       */

    real dt = dt_kin < dt_stiff ? dt_kin : dt_stiff;
    if (dt > 0.004) dt = 0.004;           /* hard ceiling                  */
    s->dt = dt;
}

/* --------------------------------------------------------------------------
 *  Allocation
 * ------------------------------------------------------------------------ */

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}

static void sim_alloc(Sim *s)
{
    memset(s, 0, sizeof *s);
    s->x  = xmalloc(sizeof(real) * CAP);
    s->y  = xmalloc(sizeof(real) * CAP);
    s->vx = xmalloc(sizeof(real) * CAP);
    s->vy = xmalloc(sizeof(real) * CAP);
    s->ax = xmalloc(sizeof(real) * CAP);
    s->ay = xmalloc(sizeof(real) * CAP);
    s->x0 = xmalloc(sizeof(real) * CAP);
    s->y0 = xmalloc(sizeof(real) * CAP);

    s->cell    = RVERL;
    s->ncx     = (int)(LX / s->cell) + 2;
    s->ncy     = (int)(LY / s->cell) + 2;
    s->ncell   = s->ncx * s->ncy;
    s->cell_head = xmalloc(sizeof(int) * (size_t)s->ncell);
    s->cell_next = xmalloc(sizeof(int) * CAP);

    s->nbr_start = xmalloc(sizeof(int) * (CAP + 1));
    s->nbr_cap   = (long)CAP * 24;
    s->nbr       = xmalloc(sizeof(int) * (size_t)s->nbr_cap);

    s->density = 0.16;
    s->mach    = 0.80;
    s->temp    = 1.0;
    s->psize   = 1.6;
    s->geom    = GEOM_NOZZLE;
    s->obs_preset = 0;
    s->obs_scale  = 1.0;
    update_dt(s);
}

static void sim_free(Sim *s)
{
    free(s->x);  free(s->y);  free(s->vx); free(s->vy);
    free(s->ax); free(s->ay); free(s->x0); free(s->y0);
    free(s->cell_head); free(s->cell_next);
    free(s->nbr_start); free(s->nbr);
}

/* --------------------------------------------------------------------------
 *  Obstacles: rebuild the active built-in obstacle
 * ------------------------------------------------------------------------ */

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
#define OBS_PRESET_COUNT 5

/* Active obstacle set = the selected built-in preset (slot 0, if any)
 * followed by every persistent mouse-placed obstacle. Rebuilding the
 * preset must NOT discard what the user dropped by hand. */
static void rebuild_obstacles(Sim *s)
{
    s->nobs = 0;
    real cx = 0.42 * LX, cy = 0.5 * LY;
    double k = s->obs_scale;          /* preset size multiplier */
    /* Scale every preset's characteristic dimensions by k. The wing's
     * thickness fraction is left fixed (it is a shape parameter, not a
     * size); only its chord scales, so the airfoil stays a proper NACA
     * section as it grows. Sizes are clamped so a huge obstacle cannot
     * exceed the duct and choke it to a single particle wide. */
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
    for (int u = 0; u < s->n_user && s->nobs < MAX_OBSTACLES; u++)
        s->obs[s->nobs++] = s->user_obs[u];
}

/* Forward decls used by carve_particles. */
static bool in_fluid(const Sim *s, real x, real y);

/* Delete every particle that now lies inside a solid region. This is the
 * correct response to an obstacle appearing on top of the gas: a stiff
 * repulsive core must never be asked to eject an already-overlapping
 * particle (that dumps enormous spurious energy and blows the run up).
 * Carve, don't catapult. Swap-delete keeps the SoA arrays compact. */
static void carve_particles(Sim *s)
{
    int n = s->n;
    for (int i = 0; i < n; ) {
        if (!in_fluid(s, s->x[i], s->y[i])) {
            int last = --n;
            s->x[i]  = s->x[last];  s->y[i]  = s->y[last];
            s->vx[i] = s->vx[last]; s->vy[i] = s->vy[last];
            s->ax[i] = s->ax[last]; s->ay[i] = s->ay[last];
            s->x0[i] = s->x0[last]; s->y0[i] = s->y0[last];
        } else {
            i++;
        }
    }
    s->n = n;
    s->need_rebuild = 1;
}

/* Single entry point for any obstacle/geometry change. Rebuilds the active
 * set, carves trapped gas, and (if the sim is live) refreshes the cell and
 * neighbour structures so the next force evaluation is consistent. */
static void build_cells(Sim *s);
static void build_neighbors(Sim *s);
static void compute_forces(Sim *s);
static void obstacles_changed(Sim *s, int live)
{
    rebuild_obstacles(s);
    carve_particles(s);
    if (live && s->n > 0) {
        build_cells(s);
        build_neighbors(s);
        compute_forces(s);
    }
}

/* --------------------------------------------------------------------------
 *  Domain test + seeding
 * ------------------------------------------------------------------------ */

static bool in_fluid(const Sim *s, real x, real y)
{
    if (x < 0.0 || x > LX) return false;
    real yb, yt;
    geom_bounds(s->geom, x, &yb, &yt);
    if (y < yb + 0.5 || y > yt - 0.5) return false;
    for (int o = 0; o < s->nobs; o++) {
        const Obstacle *ob = &s->obs[o];
        if (ob->type == OBS_CIRCLE) {
            real dx = x - ob->cx, dy = y - ob->cy;
            if (dx * dx + dy * dy < (ob->r + 0.5) * (ob->r + 0.5)) return false;
        } else if (poly_inside(ob, x, y)) {
            return false;
        }
    }
    return true;
}

/* Fill the tunnel with gas at the current density / temperature / drift. */
static void sim_seed(Sim *s)
{
    s->n = 0;
    s->sim_time = 0.0;
    s->step = 0;
    s->inject_carry = 0.0;
    update_dt(s);

    /* Lattice spacing for the requested density. The initial condition
     * MUST NOT place particles inside each other's WCA core: at high
     * density 1/sqrt(rho) drops below RCUT and, with the placement
     * jitter, thousands of pairs start deeply overlapped -> the stiff
     * force detonates the integrator on step one (this is the "raise
     * density a lot, particles vanish, rest go hypersonic" failure).
     * Floor the spacing at RCUT and shrink the jitter so the nearest
     * possible pair still sits at >= the core boundary. If the user asks
     * for a density too high to seed without overlap, we seed at the
     * densest safe lattice; the kinetic-flux inlet then fills the tunnel
     * toward the true steady state without an explosive transient. */
    real d = 1.0 / sqrt(s->density);              /* nominal spacing */
    const real d_min = RCUT * 1.06;               /* >= core + margin */
    if (d < d_min) d = d_min;
    real jit = 0.30 * (d - RCUT);                 /* keeps pairs >= RCUT */
    if (jit < 0.0)  jit = 0.0;
    if (jit > 0.35 * d) jit = 0.35 * d;

    real vth = sqrt(s->temp);
    real u   = s->mach * speed_of_sound(s);

    for (real yy = 1.0; yy < LY && s->n < CAP; yy += d) {
        for (real xx = 1.0; xx < LX && s->n < CAP; xx += d) {
            real px = xx + jit * (urand() - 0.5);
            real py = yy + jit * (urand() - 0.5);
            if (!in_fluid(s, px, py)) continue;
            int i = s->n++;
            s->x[i] = px;  s->y[i] = py;
            s->vx[i] = u + vth * grand();
            s->vy[i] = vth * grand();
            s->ax[i] = s->ay[i] = 0.0;
        }
    }
    s->need_rebuild = 1;
}

/* --------------------------------------------------------------------------
 *  Cell list  (serial, O(N), rebuilt every step)
 * ------------------------------------------------------------------------ */

static inline int cell_index(const Sim *s, real x, real y)
{
    int cx = (int)(x / s->cell) + 1;
    int cy = (int)(y / s->cell) + 1;
    if (cx < 0) cx = 0; else if (cx >= s->ncx) cx = s->ncx - 1;
    if (cy < 0) cy = 0; else if (cy >= s->ncy) cy = s->ncy - 1;
    return cy * s->ncx + cx;
}

static void build_cells(Sim *s)
{
    for (int c = 0; c < s->ncell; c++) s->cell_head[c] = -1;
    for (int i = 0; i < s->n; i++) {
        int c = cell_index(s, s->x[i], s->y[i]);
        s->cell_next[i] = s->cell_head[c];
        s->cell_head[c] = i;
    }
}

/* --------------------------------------------------------------------------
 *  Verlet neighbour list  (full list: i sees j and j sees i)
 *  Two passes (count, fill); both parallel-safe over i.
 * ------------------------------------------------------------------------ */

static void build_neighbors(Sim *s)
{
    const real rv2 = RVERL * RVERL;
    int *cnt = s->nbr_start;       /* reuse: counts then prefix-summed */

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < s->n; i++) {
        int cx = (int)(s->x[i] / s->cell) + 1;
        int cy = (int)(s->y[i] / s->cell) + 1;
        if (cx < 1) cx = 1; else if (cx > s->ncx - 2) cx = s->ncx - 2;
        if (cy < 1) cy = 1; else if (cy > s->ncy - 2) cy = s->ncy - 2;
        int c = 0;
        for (int gy = cy - 1; gy <= cy + 1; gy++)
            for (int gx = cx - 1; gx <= cx + 1; gx++) {
                int j = s->cell_head[gy * s->ncx + gx];
                while (j >= 0) {
                    if (j != i) {
                        real dx = s->x[i] - s->x[j];
                        real dy = s->y[i] - s->y[j];
                        if (dx * dx + dy * dy < rv2) c++;
                    }
                    j = s->cell_next[j];
                }
            }
        cnt[i] = c;
    }

    /* exclusive prefix sum -> start offsets */
    long total = 0;
    for (int i = 0; i < s->n; i++) {
        int c = cnt[i];
        s->nbr_start[i] = (int)total;
        total += c;
    }
    s->nbr_start[s->n] = (int)total;

    if (total > s->nbr_cap) {
        while (s->nbr_cap < total) s->nbr_cap *= 2;
        free(s->nbr);
        s->nbr = xmalloc(sizeof(int) * (size_t)s->nbr_cap);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < s->n; i++) {
        int cx = (int)(s->x[i] / s->cell) + 1;
        int cy = (int)(s->y[i] / s->cell) + 1;
        if (cx < 1) cx = 1; else if (cx > s->ncx - 2) cx = s->ncx - 2;
        if (cy < 1) cy = 1; else if (cy > s->ncy - 2) cy = s->ncy - 2;
        int w = s->nbr_start[i];
        for (int gy = cy - 1; gy <= cy + 1; gy++)
            for (int gx = cx - 1; gx <= cx + 1; gx++) {
                int j = s->cell_head[gy * s->ncx + gx];
                while (j >= 0) {
                    if (j != i) {
                        real dx = s->x[i] - s->x[j];
                        real dy = s->y[i] - s->y[j];
                        if (dx * dx + dy * dy < rv2) s->nbr[w++] = j;
                    }
                    j = s->cell_next[j];
                }
            }
    }

    for (int i = 0; i < s->n; i++) { s->x0[i] = s->x[i]; s->y0[i] = s->y[i]; }
    s->need_rebuild = 0;
}

static int neighbors_stale(const Sim *s)
{
    const real thr = 0.25 * SKIN, thr2 = thr * thr;
    int stale = 0;
    #pragma omp parallel for schedule(static) reduction(max:stale)
    for (int i = 0; i < s->n; i++) {
        real dx = s->x[i] - s->x0[i];
        real dy = s->y[i] - s->y0[i];
        if (dx * dx + dy * dy > thr2) stale = 1;
    }
    return stale;
}

/* --------------------------------------------------------------------------
 *  Forces: WCA (purely repulsive, shifted LJ).  Full list -> race-free
 *  parallel over i.
 * ------------------------------------------------------------------------ */

static void compute_forces(Sim *s)
{
    const real rc2 = RCUT * RCUT;
    /* Cap the magnitude of any single pair force. The WCA force diverges
     * as r^-13: a transient deep overlap (inflow packing at high density,
     * or a fast particle that slipped inside a core) would otherwise
     * return a multi-thousand-unit kick that flings the particle out of
     * the domain in one step ("particles vanish, the rest get fast").
     * Clamping bounds the worst case to a strong-but-survivable repulsion;
     * it engages only at separations well below sigma that should never
     * occur in steady state, so equilibrium dynamics are unchanged. The
     * pair still gets pushed firmly apart over the next few steps. */
    const real FR_MAX = 2.0e3;

    #pragma omp parallel for schedule(dynamic, 512)
    for (int i = 0; i < s->n; i++) {
        real xi = s->x[i], yi = s->y[i];
        real fx = 0.0, fy = 0.0;
        int a = s->nbr_start[i], b = s->nbr_start[i + 1];
        for (int k = a; k < b; k++) {
            int j = s->nbr[k];
            real dx = xi - s->x[j];
            real dy = yi - s->y[j];
            real r2 = dx * dx + dy * dy;
            if (r2 < rc2 && r2 > 1e-12) {
                real inv2 = 1.0 / r2;
                real inv6 = inv2 * inv2 * inv2;
                /* WCA: F/r = 48 r^-14 - 24 r^-8 (epsilon=sigma=1) */
                real fr = (48.0 * inv6 - 24.0) * inv6 * inv2;
                if (fr >  FR_MAX) fr =  FR_MAX;
                if (fr < -FR_MAX) fr = -FR_MAX;
                fx += fr * dx;
                fy += fr * dy;
            }
        }
        s->ax[i] = fx;        /* m = 1 */
        s->ay[i] = fy;
    }
}

/* --------------------------------------------------------------------------
 *  Boundary handling: tunnel walls + obstacles (specular reflection)
 * ------------------------------------------------------------------------ */

static inline void reflect(real *vx, real *vy, real nx, real ny)
{
    real d = 2.0 * (*vx * nx + *vy * ny);
    *vx -= d * nx;
    *vy -= d * ny;
}

/* slope of a wall via central difference of the profile */
static void wall_slopes(Geometry g, real x, real *sb, real *st)
{
    real h = 0.75, yb1, yt1, yb2, yt2;
    real xa = x - h, xc = x + h;
    if (xa < 0.0) xa = 0.0;
    if (xc > LX)  xc = LX;
    geom_bounds(g, xa, &yb1, &yt1);
    geom_bounds(g, xc, &yb2, &yt2);
    real inv = 1.0 / (xc - xa + 1e-12);
    *sb = (yb2 - yb1) * inv;
    *st = (yt2 - yt1) * inv;
}

/* Resolve one particle against all boundaries.
 *
 * Proper specular reflection mirrors BOTH velocity and position: a particle
 * that overshoots a surface by depth p is placed back a distance p on the
 * fluid side, not flattened onto a thin surface+eps shell. Shell-clamping
 * every incoming particle onto the same offset piles their stiff WCA cores
 * on top of each other and slowly pumps spurious energy into the gas (a
 * quiet, continuous version of the obstacle-overlap blow-up). Mirroring by
 * the true penetration depth conserves energy and keeps arrivals spread. */
static void resolve_particle(const Sim *s, int i)
{
    for (int it = 0; it < 6; it++) {
        int hit = 0;
        real x = s->x[i], y = s->y[i];

        real yb, yt;
        geom_bounds(s->geom, x, &yb, &yt);

        /* Closed upstream end. The inlet is a *source*, not an opening: a
         * wind tunnel has no exit on the left. At low Mach the drift u is
         * small next to the thermal speed sqrt(T), so a large fraction of
         * particles momentarily have vx<0 and diffuse back upstream. With
         * no wall here they cross x=0 and get deleted by apply_outflow --
         * a steady mass leak that empties the tunnel from the inlet end
         * (only invisible at high Mach, where almost nothing turns around).
         * Reflect them specularly back into the flow. */
        if (x < 0.0) {
            if (s->vx[i] < 0.0) s->vx[i] = -s->vx[i];
            s->x[i] = -x + WALL_EPS;             /* mirror across x=0 */
            hit = 1;
        }

        if (y < yb) {
            real sb, st; wall_slopes(s->geom, x, &sb, &st);
            real nx = sb, ny = -1.0;
            real L = sqrt(nx * nx + ny * ny);
            nx /= L; ny /= L;
            real vn = s->vx[i] * (-nx) + s->vy[i] * (-ny);
            if (vn < 0.0) reflect(&s->vx[i], &s->vy[i], nx, ny);
            s->y[i] = 2.0 * yb - y + WALL_EPS;   /* mirror across wall */
            hit = 1;
        } else if (y > yt) {
            real sb, st; wall_slopes(s->geom, x, &sb, &st);
            real nx = -st, ny = 1.0;
            real L = sqrt(nx * nx + ny * ny);
            nx /= L; ny /= L;
            real vn = s->vx[i] * (-nx) + s->vy[i] * (-ny);
            if (vn < 0.0) reflect(&s->vx[i], &s->vy[i], nx, ny);
            s->y[i] = 2.0 * yt - y - WALL_EPS;   /* mirror across wall */
            hit = 1;
        }

        for (int o = 0; o < s->nobs; o++) {
            const Obstacle *ob = &s->obs[o];
            if (ob->type == OBS_CIRCLE) {
                real dx = s->x[i] - ob->cx, dy = s->y[i] - ob->cy;
                real r2 = dx * dx + dy * dy;
                if (r2 < ob->r * ob->r) {
                    real r = sqrt(r2); if (r < 1e-9) r = 1e-9;
                    real nx = dx / r, ny = dy / r;
                    real vn = s->vx[i] * nx + s->vy[i] * ny;
                    if (vn < 0.0) reflect(&s->vx[i], &s->vy[i], nx, ny);
                    /* mirror: out by the penetration depth (r_obs - r) */
                    real ro = 2.0 * ob->r - r + WALL_EPS;
                    s->x[i] = ob->cx + nx * ro;
                    s->y[i] = ob->cy + ny * ro;
                    hit = 1;
                }
            } else if (poly_inside(ob, s->x[i], s->y[i])) {
                /* nearest edge; mirror out along its outward normal */
                real best = 1e30, bnx = 0, bny = 0, bqx = 0, bqy = 0;
                for (int e = 0; e < ob->nv; e++) {
                    int f = (e + 1) % ob->nv;
                    real qx, qy;
                    seg_nearest(ob->px[e], ob->py[e], ob->px[f], ob->py[f],
                                s->x[i], s->y[i], &qx, &qy);
                    real dx = s->x[i] - qx, dy = s->y[i] - qy;
                    real dd = dx * dx + dy * dy;
                    if (dd < best) {
                        best = dd; bnx = ob->nx[e]; bny = ob->ny[e];
                        bqx = qx; bqy = qy;
                    }
                }
                real vn = s->vx[i] * bnx + s->vy[i] * bny;
                if (vn < 0.0) reflect(&s->vx[i], &s->vy[i], bnx, bny);
                real depth = sqrt(best);             /* penetration depth */
                s->x[i] = bqx + bnx * (depth + WALL_EPS);
                s->y[i] = bqy + bny * (depth + WALL_EPS);
                hit = 1;
            }
        }
        if (!hit) break;
    }
}

/* --------------------------------------------------------------------------
 *  Outflow: remove particles past the right edge or strayed outside.
 * ------------------------------------------------------------------------ */

static void apply_outflow(Sim *s)
{
    int n = s->n;
    for (int i = 0; i < n; ) {
        int kill = 0;
        if (s->x[i] >= LX || s->x[i] < -2.0 ||
            s->y[i] < -2.0 || s->y[i] > LY + 2.0) kill = 1;
        if (kill) {
            int last = --n;
            s->x[i]  = s->x[last];  s->y[i]  = s->y[last];
            s->vx[i] = s->vx[last]; s->vy[i] = s->vy[last];
            s->ax[i] = s->ax[last]; s->ay[i] = s->ay[last];
            s->x0[i] = s->x0[last]; s->y0[i] = s->y0[last];
        } else {
            i++;
        }
    }
    s->n = n;
}

/* --------------------------------------------------------------------------
 *  Inflow: kinetic-flux injection of a drifting Maxwellian.
 * ------------------------------------------------------------------------ */

/* sample v_x / v_th from the flux distribution  X * exp(-(X-a)^2/2), X>=0 */
static real sample_flux_speed(real a)
{
    real mode = 0.5 * (a + sqrt(a * a + 4.0));
    real fmax = mode * exp(-0.5 * (mode - a) * (mode - a));
    real xmax = (a > 0.0 ? a : 0.0) + 7.0;
    for (int it = 0; it < 200; it++) {
        real X = urand() * xmax;
        real Y = urand() * fmax;
        if (Y <= X * exp(-0.5 * (X - a) * (X - a))) return X;
    }
    return mode;
}

static bool inlet_clear(const Sim *s, real x, real y)
{
    int cx = (int)(x / s->cell) + 1;
    int cy = (int)(y / s->cell) + 1;
    if (cx < 1) cx = 1; else if (cx > s->ncx - 2) cx = s->ncx - 2;
    if (cy < 1) cy = 1; else if (cy > s->ncy - 2) cy = s->ncy - 2;
    /* Reject injection inside any neighbour's repulsive core. RCUT is the
     * WCA cutoff: at this separation U=0 and F=0, so a particle placed no
     * closer than RCUT starts with zero pair potential and injects no
     * spurious energy. A smaller threshold (the old 0.85 sigma) seeds
     * particles inside live cores and slowly heats the whole gas. */
    const real rmin2 = RCUT * RCUT;
    for (int gy = cy - 1; gy <= cy + 1; gy++)
        for (int gx = cx - 1; gx <= cx + 1; gx++) {
            int j = s->cell_head[gy * s->ncx + gx];
            while (j >= 0) {
                real dx = x - s->x[j], dy = y - s->y[j];
                if (dx * dx + dy * dy < rmin2) return false;
                j = s->cell_next[j];
            }
        }
    return true;
}

static void apply_inflow(Sim *s)
{
    const real x_in = 0.6;
    real yb, yt;
    geom_bounds(s->geom, x_in, &yb, &yt);
    real y0 = yb + 0.8, y1 = yt - 0.8;
    real Hin = y1 - y0;
    if (Hin <= 0.0) return;

    real vth = sqrt(s->temp);
    real u   = s->mach * speed_of_sound(s);
    real S   = u / (1.41421356237 * vth);
    /* one-sided number flux per unit length of a drifting Maxwellian */
    real psi = vth / sqrt(2.0 * PI) * exp(-S * S)
               + 0.5 * u * (1.0 + erf(S));
    if (psi < 0.0) psi = 0.0;

    s->inject_carry += s->density * psi * Hin * s->dt;
    int want = (int)s->inject_carry;
    s->inject_carry -= want;

    /* Inject across a slab a few lattice spacings deep, NOT a vx*dt-thin
     * sheet. A sheet ~0.007 sigma wide guarantees every new particle
     * overlaps the previous one, the clear-check rejects almost all of
     * them, and the inlet silently runs far below the target flux. */
    real spacing = 1.0 / sqrt(s->density + 1e-9);
    real wslab   = spacing * 3.0;
    if (wslab < 3.0)  wslab = 3.0;
    if (wslab > 12.0) wslab = 12.0;

    real a = u / (vth + 1e-12);
    int placed_total = 0;
    for (int k = 0; k < want && s->n < CAP; k++) {
        int placed = 0;
        for (int tries = 0; tries < 24 && !placed; tries++) {
            real X  = sample_flux_speed(a);
            real vx = X * vth;
            real vy = vth * grand();
            real yy = y0 + urand() * Hin;
            real xx = x_in + urand() * wslab;
            if (xx >= LX) xx = LX - 0.5;
            if (!in_fluid(s, xx, yy)) continue;
            if (!inlet_clear(s, xx, yy)) continue;
            int i = s->n++;
            s->x[i] = xx;  s->y[i] = yy;
            s->vx[i] = vx; s->vy[i] = vy;
            s->ax[i] = s->ay[i] = 0.0;
            s->x0[i] = xx; s->y0[i] = yy;
            placed = 1;
            placed_total++;
        }
    }
    /* Carry the unfulfilled flux forward so the long-run mass injection
     * is exact, but cap the backlog so a genuinely saturated inlet
     * (density set absurdly high) can't accumulate an unbounded debt. */
    s->inject_carry += (real)(want - placed_total);
    real carry_cap = 2.0 * (real)(want + 1);
    if (s->inject_carry > carry_cap) s->inject_carry = carry_cap;
    s->need_rebuild = 1;
}

/* --------------------------------------------------------------------------
 *  One velocity-Verlet step
 * ------------------------------------------------------------------------ */

static void sim_step(Sim *s)
{
    real dt = s->dt, hdt = 0.5 * dt;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < s->n; i++) {
        s->vx[i] += hdt * s->ax[i];
        s->vy[i] += hdt * s->ay[i];
        s->x[i]  += dt * s->vx[i];
        s->y[i]  += dt * s->vy[i];
    }

    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < s->n; i++) resolve_particle(s, i);

    apply_outflow(s);

    build_cells(s);
    if (s->need_rebuild || neighbors_stale(s)) build_neighbors(s);

    compute_forces(s);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < s->n; i++) {
        s->vx[i] += hdt * s->ax[i];
        s->vy[i] += hdt * s->ay[i];
    }

    apply_inflow(s);

    s->sim_time += dt;
    s->step++;
}

/* --------------------------------------------------------------------------
 *  Diagnostics
 * ------------------------------------------------------------------------ */

static void measure(const Sim *s, real *T, real *mvx, real *ke)
{
    double sumv = 0.0, sx = 0.0, sy = 0.0;
    for (int i = 0; i < s->n; i++) {
        sumv += s->vx[i] * s->vx[i] + s->vy[i] * s->vy[i];
        sx   += s->vx[i];
        sy   += s->vy[i];
    }
    int n = s->n > 0 ? s->n : 1;
    double ux = sx / n, uy = sy / n;
    /* peculiar (drift-removed) granular temperature: <(v-<v>)^2>/(2 k_B) */
    double pec = sumv / n - (ux * ux + uy * uy);
    *ke  = (real)(0.5 * sumv);
    *T   = (real)(0.5 * pec);             /* 2 dof, k_B = m = 1 */
    *mvx = (real)ux;
}

/* ==========================================================================
 *  Rendering / UI  (everything below requires SDL video)
 * ======================================================================== */

/* compact 5x7 font, ASCII 0x20..0x5F, 5 columns, bit r = row r (top=0) */
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

    /* camera: world -> screen */
    double scale, cam_x, cam_y;

    int mode_disc;      /* 0 points, 1 discs */
    int show_legend;
    int show_hud;

    /* Time-averaged coarse-grained velocity field. At low Mach the bulk
     * drift u = M*c is small next to the thermal speed sqrt(T), so per-
     * particle motion is mostly zero-mean thermal noise and the flow is
     * visually invisible. The flow is a *statistical* object: bin particles
     * onto a coarse grid and exponentially time-average the cell mean
     * velocity. Thermal noise cancels in the mean; the coherent drift
     * survives. This is the standard "what the flow actually is" view. */
    int     field_mode;     /* 0 particles, 1 field+particles, 2 field+arrows */
    int     fcx, fcy;       /* coarse-grid dimensions      */
    double  fcell;          /* coarse-cell size (sigma)    */
    double *fvx, *fvy;      /* EMA of mean velocity / cell */
    double *fwt;            /* EMA of occupancy   / cell   */
    double *fsx, *fsy, *fsw;/* per-frame instantaneous sums */
    int     field_primed;   /* has the average warmed up?  */
    SDL_Texture *ftex;      /* supersampled RGBA, linear-filtered */
    uint8_t     *ftpx;      /* CPU staging for the texture        */
    int          ftw, fth;  /* texture dims (= grid * SS)         */
    double      *fsm;       /* spatially-smoothed speed per cell  */

    /* video */
    FILE   *ff;
    uint8_t *frame;
    int      rec;

    /* point batch */
    SDL_Point *pts;
    int        pts_cap;
} View;

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

/* blue -> cyan -> green -> yellow -> red colour map, t in [0,1] */
static void speed_color(double t, Uint8 *r, Uint8 *g, Uint8 *b)
{
    if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
    double x = t * 4.0;
    double R, G, B;
    if      (x < 1.0) { R = 0;          G = x;          B = 1; }
    else if (x < 2.0) { R = 0;          G = 1;          B = 2 - x; }
    else if (x < 3.0) { R = x - 2;      G = 1;          B = 0; }
    else              { R = 1;          G = 4 - x;      B = 0; }
    *r = (Uint8)(R * 255); *g = (Uint8)(G * 255); *b = (Uint8)(B * 255);
}

static void fill_circle(SDL_Renderer *ren, int cx, int cy, int rad)
{
    for (int dy = -rad; dy <= rad; dy++) {
        int dx = (int)floor(sqrt((double)(rad * rad - dy * dy)) + 0.5);
        SDL_RenderDrawLine(ren, cx - dx, cy + dy, cx + dx, cy + dy);
    }
}

/* --- Time-averaged coarse-grained velocity field ----------------------- */

static void field_init(View *v)
{
    v->fcell = 6.0;                                  /* ~6 sigma cells */
    v->fcx = (int)(LX / v->fcell) + 1;
    v->fcy = (int)(LY / v->fcell) + 1;
    size_t nc = (size_t)v->fcx * (size_t)v->fcy;
    v->fvx = xmalloc(sizeof(double) * nc);
    v->fvy = xmalloc(sizeof(double) * nc);
    v->fwt = xmalloc(sizeof(double) * nc);
    v->fsx = xmalloc(sizeof(double) * nc);
    v->fsy = xmalloc(sizeof(double) * nc);
    v->fsw = xmalloc(sizeof(double) * nc);
    for (size_t i = 0; i < nc; i++) { v->fvx[i] = v->fvy[i] = v->fwt[i] = 0.0; }
    v->field_mode = 0;
    v->field_primed = 0;

    /* The texture is supersampled: SS texels per coarse cell per axis.
     * Each texel's scalar value is bicubic (Catmull-Rom) interpolated
     * from the cell grid, then mapped to colour. Bicubic is C1 (its
     * gradient is continuous), so unlike GPU bilinear there are no
     * gradient creases at the 6 sigma cell boundaries -> genuinely
     * smooth contours, not a smoothed lattice. SS=8 is plenty; the
     * texture is still tiny (a few hundred px per axis). */
    const int SS = 8;
    v->ftw = v->fcx * SS;
    v->fth = v->fcy * SS;
    v->ftex = SDL_CreateTexture(v->ren, SDL_PIXELFORMAT_RGBA8888,
                                SDL_TEXTUREACCESS_STREAMING,
                                v->ftw, v->fth);
    if (v->ftex) SDL_SetTextureBlendMode(v->ftex, SDL_BLENDMODE_BLEND);
    v->ftpx = xmalloc((size_t)v->ftw * (size_t)v->fth * 4);
    v->fsm  = xmalloc(sizeof(double) * nc);
}

/* One accumulation pass: bin particles, blend each cell's instantaneous
 * mean velocity into a running exponential average. alpha sets the memory:
 * smaller -> smoother (more thermal-noise rejection), slower to respond. */
static void field_accumulate(View *v, const Sim *s)
{
    int nc = v->fcx * v->fcy;
    double *sx = v->fsx, *sy = v->fsy, *sw = v->fsw;
    for (int i = 0; i < nc; i++) { sx[i] = sy[i] = sw[i] = 0.0; }

    for (int i = 0; i < s->n; i++) {
        int cx = (int)(s->x[i] / v->fcell);
        int cy = (int)(s->y[i] / v->fcell);
        if (cx < 0) cx = 0; else if (cx >= v->fcx) cx = v->fcx - 1;
        if (cy < 0) cy = 0; else if (cy >= v->fcy) cy = v->fcy - 1;
        int c = cy * v->fcx + cx;
        sx[c] += s->vx[i];
        sy[c] += s->vy[i];
        sw[c] += 1.0;
    }

    const double alpha = 0.04;     /* EMA weight (~25-frame memory) */
    for (int c = 0; c < nc; c++) {
        double mvx = sw[c] > 0.0 ? sx[c] / sw[c] : 0.0;
        double mvy = sw[c] > 0.0 ? sy[c] / sw[c] : 0.0;
        v->fvx[c] += alpha * (mvx - v->fvx[c]);
        v->fvy[c] += alpha * (mvy - v->fvy[c]);
        v->fwt[c] += alpha * (sw[c] - v->fwt[c]);
    }
    v->field_primed = 1;
}

/* Per-cell smoothed speed. Dead cells (empty or outside the duct) are
 * marked with the sentinel DEAD so the hot bicubic loop can test validity
 * with a single comparison instead of recomputing geom_bounds 16x/texel. */
#define FIELD_DEAD (-1.0e30)

/* Exact field visibility at a world point: inside the duct AND outside
 * every obstacle. Tested per *texel* against the true analytic geometry
 * (not the coarse cell grid), so the field edge is pixel-sharp and follows
 * the real curved walls / circular / polygonal obstacle outlines exactly,
 * rather than a stair-stepped 6 sigma approximation. No wall margin: the
 * colour should reach right up to the surface. */
static int field_visible(const Sim *s, double x, double y)
{
    if (x < 0.0 || x > LX) return 0;
    double yb, yt; geom_bounds(s->geom, x, &yb, &yt);
    if (y < yb || y > yt) return 0;
    for (int o = 0; o < s->nobs; o++) {
        const Obstacle *ob = &s->obs[o];
        if (ob->type == OBS_CIRCLE) {
            double dx = x - ob->cx, dy = y - ob->cy;
            if (dx * dx + dy * dy < ob->r * ob->r) return 0;
        } else if (poly_inside(ob, x, y)) {
            return 0;
        }
    }
    return 1;
}

static double cell_speed(const View *v, int cx, int cy, double *valid)
{
    if (cx < 0) cx = 0; else if (cx >= v->fcx) cx = v->fcx - 1;
    if (cy < 0) cy = 0; else if (cy >= v->fcy) cy = v->fcy - 1;
    double s = v->fsm[cy * v->fcx + cx];
    if (s <= FIELD_DEAD * 0.5) { *valid = 0.0; return 0.0; }
    *valid = 1.0;
    return s;
}

/* Catmull-Rom: C1 cubic through 4 samples, fractional position t in [0,1]
 * between p1 and p2. Continuous value AND slope -> no boundary creases. */
static double catmull(double p0, double p1, double p2, double p3, double t)
{
    double t2 = t * t, t3 = t2 * t;
    return 0.5 * ((2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

static void draw_field(View *v, Sim *s)
{
    double c = speed_of_sound(s);
    /* scale colour to the commanded drift, not the sound speed: at low
     * Mach that keeps the field vivid instead of all-blue */
    double u = s->mach * c;
    double vref = (u > 0.15 ? u : 0.15) * 1.6;
    double inv = 1.0 / vref;

    /* (1) Per-cell speed, lightly spatially smoothed (3x3 binomial) so
     * contours track the flow, not per-cell statistical lumpiness. The
     * blur runs on the cell grid; empty cells are excluded from their
     * neighbours' averages so the wall edge stays crisp. */
    for (int cy = 0; cy < v->fcy; cy++) {
        for (int cx = 0; cx < v->fcx; cx++) {
            int ci = cy * v->fcx + cx;
            double wx = (cx + 0.5) * v->fcell;
            double wy = (cy + 0.5) * v->fcell;
            double yb, yt; geom_bounds(s->geom, wx, &yb, &yt);
            if (v->fwt[ci] < 0.05 || wy < yb || wy > yt) { v->fsm[ci] = FIELD_DEAD; continue; }
            double acc = 0.0, wsum = 0.0;
            static const double kern[3] = { 1.0, 2.0, 1.0 };
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = cx + dx, ny = cy + dy;
                    if (nx < 0 || nx >= v->fcx || ny < 0 || ny >= v->fcy) continue;
                    int nc2 = ny * v->fcx + nx;
                    double nwx = (nx + 0.5) * v->fcell;
                    double nwy = (ny + 0.5) * v->fcell;
                    double nyb, nyt; geom_bounds(s->geom, nwx, &nyb, &nyt);
                    if (v->fwt[nc2] < 0.05 || nwy < nyb || nwy > nyt) continue;
                    double w = kern[dx + 1] * kern[dy + 1];
                    double nsp = sqrt(v->fvx[nc2] * v->fvx[nc2]
                                    + v->fvy[nc2] * v->fvy[nc2]);
                    acc += w * nsp; wsum += w;
                }
            v->fsm[ci] = wsum > 0.0 ? acc / wsum : 0.0;
        }
    }

    /* (2) Fill the supersampled texture. The SPEED is bicubically
     * interpolated for smooth contours, but VISIBILITY is a hard per-
     * texel test against the true analytic geometry: the field reaches
     * exactly to the walls and is cut sharply at obstacle outlines, with
     * no soft fade and no stair-stepping on the cell grid. Where the 4x4
     * bicubic stencil of a visible texel reaches into dead cells (near a
     * wall), those taps are replaced by the nearest live value so the
     * colour stays correct right up to the sharp edge instead of being
     * pulled toward zero. */
    /* World extent the texture is stretched over (must match the blit
     * below exactly, or the sharp geometry mask lands offset from where
     * the texture is actually drawn). Texel (t+0.5)/dim spans [w0,w1]. */
    double wx0 = 0.5 * v->fcell, wx1 = (v->fcx - 0.5) * v->fcell;
    double wy0 = 0.5 * v->fcell, wy1 = (v->fcy - 0.5) * v->fcell;
    for (int ty = 0; ty < v->fth; ty++) {
        double ery = ((double)ty + 0.5) / v->fth;        /* 0..1 down */
        double wy = wy1 + ery * (wy0 - wy1);             /* world y   */
        double gy = wy / v->fcell - 0.5;                 /* cell coord */
        int   iy = (int)floor(gy);
        double fy = gy - iy;
        for (int tx = 0; tx < v->ftw; tx++) {
            double erx = ((double)tx + 0.5) / v->ftw;    /* 0..1 right */
            double wx = wx0 + erx * (wx1 - wx0);         /* world x    */
            double gx = wx / v->fcell - 0.5;
            int   ix = (int)floor(gx);
            double fx = gx - ix;

            uint32_t *px = (uint32_t *)&v->ftpx[((size_t)ty * v->ftw + tx) * 4];

            /* hard visibility: exact walls + obstacles, pixel-sharp */
            if (!field_visible(s, wx, wy)) { *px = 0; continue; }

            /* bicubic speed; dead stencil taps clamped to nearest live */
            double col[4];
            for (int m = -1; m <= 2; m++) {
                double p[4]; int liv[4];
                for (int n = -1; n <= 2; n++) {
                    double vv;
                    p[n + 1] = cell_speed(v, ix + n, iy + m, &vv);
                    liv[n + 1] = (vv > 0.0);
                }
                /* forward fill, then backward fill: every tap becomes the
                 * nearest live speed along the row (rows fully dead stay 0
                 * but are never sampled — the texel itself is visible) */
                double last = 0.0; int seen = 0;
                for (int n = 0; n < 4; n++) {
                    if (liv[n]) { last = p[n]; seen = 1; }
                    else if (seen) p[n] = last;
                }
                last = 0.0; seen = 0;
                for (int n = 3; n >= 0; n--) {
                    if (liv[n]) { last = p[n]; seen = 1; }
                    else if (seen) p[n] = last;
                }
                col[m + 1] = catmull(p[0], p[1], p[2], p[3], fx);
            }
            double sp = catmull(col[0], col[1], col[2], col[3], fy);
            if (sp < 0.0) sp = 0.0;

            Uint8 r, g, b; speed_color(sp * inv, &r, &g, &b);
            Uint8 a = (Uint8)(v->field_mode == 1 ? 150 : 235);
            *px = ((uint32_t)r << 24) | ((uint32_t)g << 16)
                | ((uint32_t)b << 8)  | (uint32_t)a;
        }
    }
    SDL_UpdateTexture(v->ftex, NULL, v->ftpx, v->ftw * 4);

    /* Blit stretched over the exact same world extent the fill loop used
     * (wx0,wx1,wy0,wy1 above), so the sharp geometry mask baked into the
     * texture lands precisely on the real walls and obstacle outlines. */
    SDL_Rect dst;
    dst.x = (int)w2sx(v, wx0);
    dst.y = (int)w2sy(v, wy1);
    dst.w = (int)w2sx(v, wx1) - dst.x;
    dst.h = (int)w2sy(v, wy0) - dst.y;
    SDL_RenderCopy(v->ren, v->ftex, NULL, &dst);

    if (v->field_mode != 2) return;

    /* velocity-vector overlay: an arrow per coarse cell */
    SDL_SetRenderDrawColor(v->ren, 235, 238, 245, 255);
    double L = 0.85 * v->fcell;
    for (int cy = 0; cy < v->fcy; cy++) {
        for (int cx = 0; cx < v->fcx; cx++) {
            int c2 = cy * v->fcx + cx;
            if (v->fwt[c2] < 0.05) continue;
            double wx = (cx + 0.5) * v->fcell;
            double wy = (cy + 0.5) * v->fcell;
            if (!field_visible(s, wx, wy)) continue;
            double sp = sqrt(v->fvx[c2] * v->fvx[c2]
                           + v->fvy[c2] * v->fvy[c2]) + 1e-9;
            double ux = v->fvx[c2] / sp, uy = v->fvy[c2] / sp;
            double mag = sp * inv; if (mag > 1.0) mag = 1.0;
            double ex = wx + ux * L * mag, ey = wy + uy * L * mag;
            int ax = (int)w2sx(v, wx), ay = (int)w2sy(v, wy);
            int bx = (int)w2sx(v, ex), by = (int)w2sy(v, ey);
            SDL_RenderDrawLine(v->ren, ax, ay, bx, by);
            /* small arrowhead */
            double hx = -ux - uy, hy = ux - uy;
            double gx = -ux + uy, gy = -ux - uy;
            int h1x = (int)w2sx(v, ex + 0.18 * L * mag * hx);
            int h1y = (int)w2sy(v, ey + 0.18 * L * mag * hy);
            int h2x = (int)w2sx(v, ex + 0.18 * L * mag * gx);
            int h2y = (int)w2sy(v, ey + 0.18 * L * mag * gy);
            SDL_RenderDrawLine(v->ren, bx, by, h1x, h1y);
            SDL_RenderDrawLine(v->ren, bx, by, h2x, h2y);
        }
    }
}

static void draw_world(View *v, Sim *s)
{
    /* tunnel walls */
    SDL_SetRenderDrawColor(v->ren, 90, 110, 140, 255);
    int prevx = 0, prevyb = 0, prevyt = 0, have = 0;
    for (int px = 0; px <= 240; px++) {
        double wx = LX * px / 240.0;
        double yb, yt;
        geom_bounds(s->geom, wx, &yb, &yt);
        int sx  = (int)w2sx(v, wx);
        int syb = (int)w2sy(v, yb);
        int syt = (int)w2sy(v, yt);
        if (have) {
            SDL_RenderDrawLine(v->ren, prevx, prevyb, sx, syb);
            SDL_RenderDrawLine(v->ren, prevx, prevyt, sx, syt);
        }
        prevx = sx; prevyb = syb; prevyt = syt; have = 1;
    }

    /* obstacles */
    SDL_SetRenderDrawColor(v->ren, 70, 80, 100, 255);
    for (int o = 0; o < s->nobs; o++) {
        const Obstacle *ob = &s->obs[o];
        if (ob->type == OBS_CIRCLE) {
            int cx = (int)w2sx(v, ob->cx);
            int cy = (int)w2sy(v, ob->cy);
            int rr = (int)(ob->r * v->scale);
            for (int yy = -rr; yy <= rr; yy++) {
                int dx = (int)sqrt((double)(rr * rr - yy * yy));
                SDL_RenderDrawLine(v->ren, cx - dx, cy + yy, cx + dx, cy + yy);
            }
        } else {
            SDL_Point poly[MAX_POLY_V + 1];
            for (int k = 0; k < ob->nv; k++) {
                poly[k].x = (int)w2sx(v, ob->px[k]);
                poly[k].y = (int)w2sy(v, ob->py[k]);
            }
            poly[ob->nv] = poly[0];
            SDL_SetRenderDrawColor(v->ren, 120, 135, 165, 255);
            SDL_RenderDrawLines(v->ren, poly, ob->nv + 1);
        }
    }

    /* coarse-grained, time-averaged flow field (the low-Mach fix) */
    if (v->field_mode != 0 && v->field_primed) draw_field(v, s);

    /* In pure field+arrows mode the particles add only noise; in
     * field+particles mode keep them faint as texture; otherwise full. */
    if (v->field_mode == 2) return;

    /* particles, coloured by speed relative to sound speed */
    double c = speed_of_sound(s);
    double invref = 1.0 / (1.8 * c + 1e-9);

    if (v->mode_disc) {
        for (int i = 0; i < s->n; i++) {
            double sp = sqrt(s->vx[i] * s->vx[i] + s->vy[i] * s->vy[i]);
            Uint8 r, g, b; speed_color(sp * invref, &r, &g, &b);
            SDL_SetRenderDrawColor(v->ren, r, g, b, 255);
            int sx = (int)w2sx(v, s->x[i]);
            int sy = (int)w2sy(v, s->y[i]);
            int rad = (int)(0.5 * s->psize * v->scale * 0.5);
            if (rad < 1) rad = 1;
            fill_circle(v->ren, sx, sy, rad);
        }
    } else {
        if (s->n > v->pts_cap) {
            v->pts_cap = s->n + 1024;
            free(v->pts);
            v->pts = xmalloc(sizeof(SDL_Point) * (size_t)v->pts_cap);
        }
        /* bucket by colour to limit state changes */
        for (int band = 0; band < 16; band++) {
            int cnt = 0;
            Uint8 r, g, b;
            speed_color((band + 0.5) / 16.0, &r, &g, &b);
            for (int i = 0; i < s->n; i++) {
                double sp = sqrt(s->vx[i] * s->vx[i] + s->vy[i] * s->vy[i]);
                int bb = (int)(sp * invref * 16.0);
                if (bb < 0) bb = 0; else if (bb > 15) bb = 15;
                if (bb != band) continue;
                v->pts[cnt].x = (int)w2sx(v, s->x[i]);
                v->pts[cnt].y = (int)w2sy(v, s->y[i]);
                cnt++;
            }
            if (cnt) {
                SDL_SetRenderDrawColor(v->ren, r, g, b, 255);
                SDL_RenderDrawPoints(v->ren, v->pts, cnt);
            }
        }
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

static void draw_legend(View *v, Sim *s)
{
    int x = v->W - 170, y = 60, w = 150, h = 14;
    for (int i = 0; i < w; i++) {
        Uint8 r, g, b; speed_color((double)i / w, &r, &g, &b);
        SDL_SetRenderDrawColor(v->ren, r, g, b, 255);
        SDL_RenderDrawLine(v->ren, x + i, y, x + i, y + h);
    }
    draw_text(v, x, y - 14, "SPEED / 1.8 C", 1, 210, 215, 225);
    char a[16]; snprintf(a, sizeof a, "0");
    draw_text(v, x, y + h + 4, a, 1, 200, 200, 200);
    double c = speed_of_sound(s);
    snprintf(a, sizeof a, "M=%.1f", 1.8 * c / c);
    draw_text(v, x + w - 36, y + h + 4, "M~1.8", 1, 200, 200, 200);
}

/* --------------------------------------------------------------------------
 *  Video recording (pipe raw RGB24 frames to ffmpeg)
 * ------------------------------------------------------------------------ */

static void rec_start(View *v)
{
    if (v->rec) return;
    time_t tt = time(NULL);
    struct tm *lt = localtime(&tt);
    char fname[128];
    strftime(fname, sizeof fname, "wind_tunnel_%Y%m%d_%H%M%S.mp4", lt);
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

/* --------------------------------------------------------------------------
 *  Main
 * ------------------------------------------------------------------------ */

static int run_selftest(void)
{
    Sim s; sim_alloc(&s);
    rng_seed(20240517u);
    const char *en = getenv("WT_STEPS");
    int steps = en ? atoi(en) : 600;
    const char *ed = getenv("WT_DENS");
    const char *em = getenv("WT_MACH");
    const char *et = getenv("WT_TEMP");
    const char *eg = getenv("WT_GEOM");
    const char *eo = getenv("WT_OBS");
    const char *eos = getenv("WT_OBSCALE");
    if (ed) s.density = atof(ed);
    if (em) s.mach    = atof(em);
    if (et) s.temp    = atof(et);
    if (eg) s.geom    = (Geometry)(atoi(eg) % GEOM_COUNT);
    if (eo) s.obs_preset = atoi(eo) % OBS_PRESET_COUNT;
    if (eos) {
        s.obs_scale = atof(eos);
        if (s.obs_scale < 0.15) s.obs_scale = 0.15;
        if (s.obs_scale > 6.0)  s.obs_scale = 6.0;
    }
    update_dt(&s);

    rebuild_obstacles(&s);
    sim_seed(&s);
    build_cells(&s);
    build_neighbors(&s);
    compute_forces(&s);

    int n0 = s.n;
    /* Wake probe, matched to the LBM harness: average particle v_y in a
     * small box 2.5 diameters downstream of the obstacle on the
     * centreline. Averaging many particles cancels the zero-mean thermal
     * jitter (same principle as the flow-field view), leaving the
     * coherent Karman oscillation. Its frequency -> Strouhal number,
     * directly comparable to the LBM's St and the literature ~0.20. */
    double obs_D = 0.0, obs_cx = 0.42 * LX, obs_cy = 0.5 * LY;
    if (s.obs_preset == 1 && s.nobs > 0 && s.obs[0].type == OBS_CIRCLE)
        obs_D = 2.0 * s.obs[0].r;
    double pbx = obs_cx + 2.5 * (obs_D > 0 ? obs_D : 20.0);
    double pby = obs_cy;
    double phw = (obs_D > 0 ? 0.5 * obs_D : 10.0);  /* probe box half-width */
    double *uy_ts = xmalloc(sizeof(double) * (steps + 1));
    int ts_n = 0;
    /* Regression hook: switch the obstacle preset partway through, exactly
     * as the interactive 'O' key does, to verify mid-run insertion no
     * longer ejects/heats the gas. WT_ADDOBS_AT=step  WT_ADDOBS=preset */
    const char *eaa = getenv("WT_ADDOBS_AT");
    const char *eap = getenv("WT_ADDOBS");
    int addobs_at = eaa ? atoi(eaa) : -1;
    int addobs_to = eap ? (atoi(eap) % OBS_PRESET_COUNT) : 1;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int k = 0; k < steps; k++) {
        if (k == addobs_at) {
            s.obs_preset = addobs_to;
            obstacles_changed(&s, 1);
            real Tx, vx2, kx; measure(&s, &Tx, &vx2, &kx);
            printf("  [insert obstacle '%s' @ step %d]  N->%d  "
                   "T(bulk-rel)=%.3f\n",
                   obs_name(addobs_to), k, s.n, (double)Tx);
        }
        sim_step(&s);
        /* sample the wake probe: mean v_y of particles in the box */
        double sy = 0.0; int sc = 0;
        for (int i = 0; i < s.n; i++) {
            if (s.x[i] >= pbx - phw && s.x[i] <= pbx + phw &&
                s.y[i] >= pby - phw && s.y[i] <= pby + phw) {
                sy += s.vy[i]; sc++;
            }
        }
        uy_ts[ts_n++] = sc > 0 ? sy / sc : 0.0;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double dt = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    real T, mvx, ke; measure(&s, &T, &mvx, &ke);
    double c = speed_of_sound(&s);

    /* Strouhal from probe zero-crossings over the second half (discard
     * the startup transient). The MD time series is in simulation-time
     * units: a "step" advances by s.dt, so the shedding frequency in
     * physical units is (cycles/step) / dt, and St = f D / u with
     * u = mach * c the inlet speed. Same estimator as the LBM harness. */
    double St = -1.0, shed_f = -1.0;
    if (obs_D > 0.0 && ts_n > 200) {
        int h0 = ts_n / 2;
        double mean = 0.0;
        for (int i = h0; i < ts_n; i++) mean += uy_ts[i];
        mean /= (ts_n - h0);
        int crossings = 0, first = -1, last = -1;
        for (int i = h0 + 1; i < ts_n; i++) {
            double a = uy_ts[i - 1] - mean, b = uy_ts[i] - mean;
            if ((a <= 0.0 && b > 0.0) || (a >= 0.0 && b < 0.0)) {
                if (first < 0) first = i;
                last = i; crossings++;
            }
        }
        if (crossings >= 4 && last > first) {
            double half_p = (double)(last - first) / (crossings - 1);
            double period_steps = 2.0 * half_p;
            double f_per_time = 1.0 / (period_steps * s.dt);
            shed_f = f_per_time;
            St = f_per_time * obs_D / (s.mach * c + 1e-12);
        }
    }
    free(uy_ts);

#ifdef _OPENMP
    int th = omp_get_max_threads();
#else
    int th = 1;
#endif
    printf("self-test: geom=%s obstacle=%s\n", geom_name(s.geom),
           obs_name(s.obs_preset));
    printf("  seeded N        : %d\n", n0);
    printf("  final  N        : %d\n", s.n);
    printf("  steps           : %d  (dt=%.5f)\n", steps, s.dt);
    printf("  threads         : %d\n", th);
    printf("  wall time       : %.3f s   (%.0f steps/s)\n",
           dt, steps / dt);
    printf("  particle-steps/s: %.3e\n", (double)n0 * steps / dt);
    printf("  measured T      : %.4f  (set %.4f)\n", (double)T, (double)s.temp);
    printf("  mean v_x        : %.4f  (sound c=%.4f, target u=%.4f)\n",
           (double)mvx, c, s.mach * c);
    printf("  KE/N            : %.4f\n", (double)(ke / (s.n ? s.n : 1)));
    if (St > 0.0)
        printf("  wake probe f    : %.3e /time   (zero-cross estimate; "
               "St=%.3f -- NOT noise-robust, see README)\n", shed_f, St);
    else
        printf("  wake probe f    : n/a  (no oscillation detected)\n");
    int ok = (s.n > 0 && isfinite((double)T) && isfinite((double)ke)
              && T > 0.0 && T < 100.0);
    printf("  status          : %s\n", ok ? "OK" : "FAIL");
    sim_free(&s);
    return ok ? 0 : 1;
}

static void slider_drag(Slider *sl, int n, int mx, int my)
{
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
        }
    }
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    if (getenv("WT_SELFTEST")) return run_selftest();

    Sim s; sim_alloc(&s);
    rng_seed((uint64_t)time(NULL));
    rebuild_obstacles(&s);

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    View v; memset(&v, 0, sizeof v);
    v.W = WIN_W; v.H = WIN_H;
    v.win = SDL_CreateWindow("MD wind tunnel",
                SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                v.W, v.H, SDL_WINDOW_SHOWN);
    if (!v.win) { fprintf(stderr, "window: %s\n", SDL_GetError()); return 1; }
    /* Linear texture filtering must be hinted before the renderer is
     * created to reliably take effect across SDL backends; this is what
     * makes the field texture interpolate smoothly instead of nearest. */
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
    v.ren = SDL_CreateRenderer(v.win, -1,
                SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!v.ren) { fprintf(stderr, "renderer: %s\n", SDL_GetError()); return 1; }

    v.scale = (v.W - 60) / LX;
    v.cam_x = -30.0 / v.scale;
    v.cam_y = -(v.H - LY * v.scale) * 0.5 / v.scale;
    v.mode_disc  = 0;
    v.show_legend = 1;
    v.show_hud    = 1;
    field_init(&v);
    v.pts_cap = CAP;
    v.pts = xmalloc(sizeof(SDL_Point) * (size_t)v.pts_cap);

    Slider sl[4] = {
        { 16,  v.H - 96,  220, 12, 0.02, 0.90, &s.density, "DENSITY", 0 },
        { 16,  v.H - 64,  220, 12, 0.00, 3.50, &s.mach,    "MACH",    0 },
        { 260, v.H - 96,  220, 12, 0.20, 4.00, &s.temp,    "TEMP",    0 },
        { 260, v.H - 64,  220, 12, 0.50, 5.00, &s.psize,   "PT SIZE", 0 },
    };

    int running = 0, paused = 0, dragging_slider = 0, panning = 0;
    int substeps = 8;   /* physics steps per rendered frame; [ ] adjust */
    int last_mx = 0, last_my = 0;
    Uint32 fps_t = SDL_GetTicks();
    int fps_c = 0; double fps = 0.0;

    int quit = 0;
    while (!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = 1;
            else if (e.type == SDL_KEYDOWN) {
                SDL_Keycode k = e.key.keysym.sym;
                if (k == SDLK_ESCAPE || k == SDLK_q) quit = 1;
                else if (k == SDLK_SPACE) {
                    running = !running;
                    if (running) {
                        update_dt(&s); rebuild_obstacles(&s); sim_seed(&s);
                        build_cells(&s); build_neighbors(&s);
                        compute_forces(&s); paused = 0;
                    }
                }
                else if (k == SDLK_p) paused = !paused;
                else if (k == SDLK_v) { if (v.rec) rec_stop(&v); else rec_start(&v); }
                else if (k == SDLK_r) {
                    running = 0; paused = 0; s.n = 0;
                    s.n_user = 0;
                    rebuild_obstacles(&s);
                }
                else if (k == SDLK_g) {
                    s.geom = (Geometry)((s.geom + 1) % GEOM_COUNT);
                    if (running) { rebuild_obstacles(&s); sim_seed(&s);
                        build_cells(&s); build_neighbors(&s);
                        compute_forces(&s); }
                    else rebuild_obstacles(&s);
                }
                else if (k == SDLK_o) {
                    s.obs_preset = (s.obs_preset + 1) % OBS_PRESET_COUNT;
                    obstacles_changed(&s, running);
                }
                else if (k == SDLK_c) {
                    s.obs_preset = 0; s.n_user = 0;
                    obstacles_changed(&s, running);
                }
                else if (k == SDLK_EQUALS || k == SDLK_PLUS
                         || k == SDLK_KP_PLUS) {
                    /* grow the preset obstacle (= is unshifted +) */
                    s.obs_scale *= 1.12;
                    if (s.obs_scale > 6.0) s.obs_scale = 6.0;
                    if (s.obs_preset != 0) obstacles_changed(&s, running);
                }
                else if (k == SDLK_MINUS || k == SDLK_KP_MINUS) {
                    s.obs_scale /= 1.12;
                    if (s.obs_scale < 0.15) s.obs_scale = 0.15;
                    if (s.obs_preset != 0) obstacles_changed(&s, running);
                }
                else if (k == SDLK_0) {
                    s.obs_scale = 1.0;            /* reset to default */
                    if (s.obs_preset != 0) obstacles_changed(&s, running);
                }
                else if (k == SDLK_LEFTBRACKET) {
                    if (substeps > 1) substeps--;
                }
                else if (k == SDLK_RIGHTBRACKET) {
                    if (substeps < 40) substeps++;
                }
                else if (k == SDLK_m) v.mode_disc = !v.mode_disc;
                else if (k == SDLK_f) {
                    v.field_mode = (v.field_mode + 1) % 3;
                }
                else if (k == SDLK_l) v.show_legend = !v.show_legend;
                else if (k == SDLK_h) v.show_hud = !v.show_hud;
            }
            else if (e.type == SDL_MOUSEWHEEL) {
                int mx, my; SDL_GetMouseState(&mx, &my);
                double wx = v.cam_x + mx / v.scale;
                double wy = v.cam_y + (v.H - my) / v.scale;
                double f = (e.wheel.y > 0) ? 1.15 : 1.0 / 1.15;
                v.scale *= f;
                v.cam_x = wx - mx / v.scale;
                v.cam_y = wy - (v.H - my) / v.scale;
            }
            else if (e.type == SDL_MOUSEBUTTONDOWN) {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    int mx = e.button.x, my = e.button.y;
                    if (my > v.H - 110) { dragging_slider = 1;
                        slider_drag(sl, 4, mx, my); }
                    else {
                        double wx = v.cam_x + mx / v.scale;
                        double wy = v.cam_y + (v.H - my) / v.scale;
                        if (s.n_user < MAX_OBSTACLES &&
                            wx > 0 && wx < LX && wy > 0 && wy < LY) {
                            s.user_obs[s.n_user++] =
                                make_circle(wx, wy, 0.05 * LY);
                            obstacles_changed(&s, running);
                        }
                    }
                } else if (e.button.button == SDL_BUTTON_RIGHT ||
                           e.button.button == SDL_BUTTON_MIDDLE) {
                    panning = 1; last_mx = e.button.x; last_my = e.button.y;
                }
            }
            else if (e.type == SDL_MOUSEBUTTONUP) {
                dragging_slider = 0; panning = 0;
            }
            else if (e.type == SDL_MOUSEMOTION) {
                if (dragging_slider) slider_drag(sl, 4, e.motion.x, e.motion.y);
                else if (panning) {
                    v.cam_x -= (e.motion.x - last_mx) / v.scale;
                    v.cam_y += (e.motion.y - last_my) / v.scale;
                    last_mx = e.motion.x; last_my = e.motion.y;
                }
            }
        }

        if (running && !paused) {
            update_dt(&s);
            for (int sub = 0; sub < substeps; sub++) sim_step(&s);
            field_accumulate(&v, &s);
        }

        SDL_SetRenderDrawColor(v.ren, 16, 18, 24, 255);
        SDL_RenderClear(v.ren);
        draw_world(&v, &s);

        for (int i = 0; i < 4; i++) draw_slider(&v, &sl[i]);
        if (v.show_legend) draw_legend(&v, &s);

        if (v.show_hud) {
            real T, mvx, ke; measure(&s, &T, &mvx, &ke);
            double c = speed_of_sound(&s);
#ifdef _OPENMP
            int nthreads = omp_get_max_threads();
#else
            int nthreads = 1;   /* built without OpenMP */
#endif
            const char *fv = (v.field_mode == 0) ? "PARTICLES"
                           : (v.field_mode == 1) ? "FIELD+PTS"
                                                  : "FIELD+ARROWS";
            char hud[512];
            char obsz[40];
            if (s.obs_preset != 0)
                snprintf(obsz, sizeof obsz, " x%.2f", s.obs_scale);
            else
                obsz[0] = '\0';
            snprintf(hud, sizeof hud,
                "N %d   FPS %.0f   THREADS %d   T(bulk-rel) %.2f   MEAN VX %.2f\n"
                "SOUND C %.2f   INLET M %.2f   STEP %ld   DT %.4f   SUB/FR %d\n"
                "GEOM %s   OBSTACLE %s%s   VIEW %s   %s%s",
                s.n, fps, nthreads, (double)T, (double)mvx,
                c, s.mach, s.step, s.dt, substeps,
                geom_name(s.geom), obs_name(s.obs_preset), obsz, fv,
                running ? (paused ? "PAUSED " : "RUNNING ") : "STOPPED ",
                v.rec ? " *REC*" : "");
            draw_text(&v, 14, 12, hud, 2, 220, 225, 235);
            draw_text(&v, 14, v.H - 26,
                "SPACE START/STOP  P PAUSE  V REC  R RESET  G GEOM  O OBSTACLE  "
                "+/- SIZE  C CLEAR  [ ] SPEED  F FLOW-FIELD  M MODE  WHEEL ZOOM",
                1, 150, 160, 180);
        }

        if (v.rec) rec_frame(&v);
        SDL_RenderPresent(v.ren);

        if (++fps_c >= 30) {
            Uint32 now = SDL_GetTicks();
            fps = 1000.0 * fps_c / (now - fps_t + 1);
            fps_t = now; fps_c = 0;
        }
    }

    if (v.rec) rec_stop(&v);
    free(v.pts);
    free(v.fvx); free(v.fvy); free(v.fwt);
    free(v.fsx); free(v.fsy); free(v.fsw);
    if (v.ftex) SDL_DestroyTexture(v.ftex);
    free(v.ftpx);
    free(v.fsm);
    free(v.frame);
    SDL_DestroyRenderer(v.ren);
    SDL_DestroyWindow(v.win);
    SDL_Quit();
    sim_free(&s);
    return 0;
}
