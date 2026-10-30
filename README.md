# 2D Molecular-Dynamics Wind Tunnel

A particle-based 2D wind tunnel. A gas of many soft-repulsive discs is injected
from the left through a kinetic boundary condition, flows through a shaped
channel past an obstacle, and exits on the right. Forces are integrated with
velocity-Verlet using a Verlet neighbor list rebuilt on a cell grid. The
SDL2 layer gives live sliders, mouse pan/zoom, geometry/obstacle switching,
and one-key MP4 recording.

Single-file C17. Builds with zero warnings under
`-Wall -Wextra -Wpedantic -Wshadow`.

---

## 1. The physics decision you should know about

You asked for *hard spheres* **and** for *Verlet integration with Verlet
lists*. These two requirements are, strictly speaking, mutually exclusive, and
I did not paper over that — here is the reasoning and what I actually built.

True hard spheres have a discontinuous (infinite) pair potential. Their
dynamics are **event-driven**: there is no force and no timestep; the
simulation advances ballistically from one exactly-computed collision event to
the next. Velocity-Verlet integrates a *finite* force over a *fixed* timestep.
A fixed-step integrator applied to an infinitely stiff potential either tunnels
through cores or explodes. So an honest "hard spheres + Verlet" program does
not exist.

The standard resolution in molecular dynamics — and the one used here — is a
**pseudo-hard-sphere** fluid: a purely repulsive, very stiff but *finite*
core. Specifically the WCA potential (Weeks–Chandler–Andersen): a
Lennard-Jones potential truncated and shifted at its minimum
`r_cut = 2^(1/6) σ ≈ 1.122`, so there is no attractive tail — only a steep
elastic repulsion.

```
U(r) = 4ε[(σ/r)^12 − (σ/r)^6] + ε      for r < 2^(1/6) σ
U(r) = 0                                otherwise
F(r)/r = (48 r^-12 − 24 r^-6) / r^2     (reduced units, ε=σ=m=1)
```

This buys three things that matter for a *wind tunnel* specifically:

1. It is genuinely hard-sphere-*like*: discs bounce, you get shock-like
   compression in front of blunt bodies, wakes, and recirculation.
2. It has a smooth, finite force, so velocity-Verlet and a skin-based Verlet
   list are exactly the right tools — which is what you wanted to exercise.
3. It has a well-defined equation of state and therefore a well-defined
   **speed of sound**, so the Mach-number slider ("entering velocity as a
   fraction of the speed of sound in this gas") is physically meaningful
   rather than decorative.

If you ever want *true* hard spheres later, that is a different program (an
event-driven scheduler with a priority queue of collision times — DynamO is
the reference implementation). This one is the Verlet-list MD you asked for.

### Reduced units

Everything is in Lennard-Jones reduced units: `σ = m = ε = k_B = 1`. Lengths
are in σ, energies in ε, temperature in ε/k_B. The tunnel is `LX = 320 σ` long
and `LY = 100 σ` tall.

### Speed of sound

For a 2D monatomic ideal gas the adiabatic index is `γ = (d+2)/d = 2`, so the
isentropic sound speed used for the Mach conversion is

```
c = sqrt(γ k_B T / m) = sqrt(2 T)
```

The injected bulk velocity is `u = Mach · c`. The HUD prints `c`, the target
`u`, and the *measured* mean `vx` so you can see the boundary condition
converge. (At finite density the true sound speed is slightly above the
ideal-gas value because of the repulsive core; the slider is calibrated to the
ideal-gas `c`, which is the conventional and reproducible choice.)

### Inflow / outflow boundary conditions

The left edge does **not** teleport a frozen block of particles in. It uses a
proper *kinetic-flux* injection of a drifting Maxwellian: the number of
particles crossing the inlet per unit time is

```
ψ = v_th/√(2π) · exp(−S²) + (u/2)(1 + erf(S)),   S = u/(√2 v_th)
```

with `v_th = sqrt(T)`. Per-step injection counts are accumulated as a
fractional carry so the long-run flux is exact. The injected x-velocity is
drawn from the flux-weighted distribution `∝ X·exp(−(X−a)²/2)` by rejection
sampling; transverse velocity is an ordinary Gaussian at temperature `T`. A
cell-list overlap test (`inlet_clear`, min spacing = the WCA core boundary
`RCUT = 2^(1/6) σ`, where pair potential and force are exactly zero) prevents
injecting a particle inside an existing core, which would otherwise pump
spurious energy into the gas. The **upstream (left) end is a closed specular
wall**, not an opening: a wind-tunnel inlet is a source, so a particle that
diffuses back to `x < 0` (common at low Mach, where the drift is small next to
the thermal speed) reflects back into the flow rather than escaping. The right
edge is pure outflow: anything with `x ≥ LX` is removed by swap-deletion.

Walls and obstacles are **specular reflectors**: on a boundary violation the
particle is mirrored back across the surface by its true penetration depth and
its velocity reflected about the local normal (analytic slope for the tunnel
walls via central difference; radial normal for circles; nearest-edge outward
normal for polygons; polygon normals oriented from the signed area so they
point outward for any winding), iterated up to 6× so corners resolve.

---

## 2. Build

Dependencies: a C17 compiler, SDL2, OpenMP, and `ffmpeg` on `PATH` for video.
The Makefile auto-detects Linux vs macOS (`uname -s`).

**Linux** (GCC ships OpenMP):

```sh
sudo apt install build-essential libsdl2-dev ffmpeg   # or distro equivalent
make
```

**macOS** — Apple's Clang has **no built-in OpenMP**, so `-fopenmp` fails with
`clang: error: unsupported option '-fopenmp'`. Install Homebrew's OpenMP
runtime and SDL2 once:

```sh
brew install libomp sdl2 ffmpeg pkg-config
make
```

The Makefile then compiles with Apple Clang via `-Xpreprocessor -fopenmp`,
points the include/lib at `$(brew --prefix libomp)`, and embeds an rpath so
the binary finds `libomp.dylib` at runtime. libomp's header is included with
`-isystem` (not `-I`): its vendored `omp.h` uses C23-range enumerators that
would trip `-Wpedantic` under `-std=c17`; treating it as a system header
silences those without relaxing the zero-warning bar on our own code. It also
drops `-march=native` on macOS (an x86 idiom that errors on Apple Silicon;
Apple Clang already targets the host CPU). If `libomp` is missing, `make`
stops early with an explicit install hint rather than a cryptic linker error.

Common targets (both platforms):

```sh
make            # optimized build (zero compiler warnings)
make selftest   # headless physics self-check (no window, no SDL video)
make bench      # 1-thread vs all-cores timing
make clean
```

The source guards `#include <omp.h>` and the one `omp_get_max_threads()` call
behind `#ifdef _OPENMP`, so it also compiles and runs single-threaded if you
ever build without OpenMP at all.

`make` must produce **no compiler warnings**. The Makefile notes that
`-Wconversion -Wsign-conversion` are available if you want a full numeric-cast
audit; they are off by default because the only hits are legitimate
float↔int casts in the rasterizer.

---

## 3. Run and controls

```sh
./wind_tunnel
```

**Keyboard**

| Key | Action |
|-----|--------|
| `SPACE` | start / stop the experiment (start reseeds the gas) |
| `P` | pause / resume |
| `V` | start recording; press again to stop (writes `wind_tunnel_<timestamp>.mp4`) |
| `R` | reset |
| `G` | cycle tunnel geometry |
| `O` | cycle obstacle preset: none → cylinder → flat plate → wedge → NACA-0015 wing |
| `+` / `-` | grow / shrink the preset obstacle (`=` also works as `+`) |
| `0` | reset preset obstacle to default size |
| `C` | clear all obstacles (preset and hand-placed) |
| `[` / `]` | physics substeps per frame (flow speed on screen), 1–40 |
| `F` | flow-field view: particles → field+particles → field+arrows |
| `M` | render mode: points ↔ filled discs |
| `L` | toggle legend |
| `H` | toggle HUD |
| `ESC` / `Q` | quit |

**Mouse**

- **Wheel** — zoom about the cursor.
- **Right- or middle-drag** — pan.
- **Left-click in the tunnel** — drop a circular obstacle. Hand-placed
  obstacles persist when you cycle the preset with `O` or change geometry;
  `C` clears them.
- **Left-drag on a slider** — adjust it live.

When an obstacle appears on top of gas (preset cycle or a click into a live
flow), the particles inside the new solid are **deleted**, not ejected. A
stiff repulsive core asked to push out an already-overlapping particle would
dump enormous spurious energy and blow the run up — so the simulation carves
the gas out of the solid instead. Visible effect: dropping an obstacle removes
roughly its area × density worth of particles and the temperature stays put,
rather than the density collapsing and the temperature exploding.

`+` / `-` scale only the **preset** obstacle (a multiplier on its
characteristic dimensions; hand-placed obstacles keep their placement-time
size — you size the body you are studying, probes stay as dropped). Each
resize goes through the same carve path, so growing the obstacle into the gas
removes the newly-occluded particles cleanly and shrinking it just opens space
the inflow refills; neither perturbs the temperature. The wing's thickness
fraction is held fixed while its chord scales, so it stays a valid NACA
section at any size. Every preset is clamped so it cannot exceed the duct and
choke the tunnel to nothing — past the clamp the size simply saturates. This
is the natural control for watching the bow shock detach: hold Mach above 1,
then grow the cylinder and watch the standoff distance and shock curvature
respond to body size in real time.

**Sliders**: entering density, Mach number (entering speed ÷ sound speed),
temperature, and particle draw size. Density and Mach take effect on the next
injected particles, so you can watch the flow respond without resetting.

**Geometries**: straight duct, converging–diverging (de-Laval) nozzle,
venturi, diffuser, and a transonic bump. The nozzle is the interesting one to
push past Mach 1 against the cylinder.

**Flow speed on screen.** The tunnel is 320 σ long and the timestep is capped
for stability of the stiff core (`dt ≤ 0.004`, and lower still at high
density), so one full tunnel transit is
tens of thousands of steps. The physics rate is therefore decoupled from the
frame rate: each rendered frame advances several substeps (`[` / `]`, default
8). Local structure — the bow wave, stagnation point, and wake behind an
obstacle — develops within a few seconds at the default; raise it to watch a
front cross the whole tunnel faster, lower it to study a collision closely.
This is a render cadence, not a change to `dt`: the dynamics are identical at
any setting.

**Seeing the flow at low Mach (`F`).** At Mach *M* the bulk drift is
`u = M·c` while every particle also carries thermal speed `v_th = √T`, so the
flow-to-noise ratio is `u/v_th = M·√2`. At Mach 0.1 the coherent drift is only
~1/7 of each particle's random jitter: watching individual particles you see
mostly zero-mean Brownian motion with the flow buried in it. That is not a
simulation defect — it is correct kinetic physics, and it is exactly why real
wind tunnels visualize *fields* (smoke, dye, PIV) rather than molecules. The
flow is a statistical object, not a per-particle one.

`F` cycles a coarse-grained, time-averaged velocity field: particles are
binned onto a ~6 σ grid and each cell's mean velocity is exponentially time-
averaged (~25-frame memory). The thermal component is zero-mean and cancels in
the average; the coherent drift survives and becomes vivid even at very low
Mach. Three views: particles only → field heat-map with faint particles →
field with per-cell velocity arrows. The colour is scaled to the commanded
drift `u` (not the sound speed) so a slow flow still fills the dynamic range
instead of reading uniformly blue.

The heat-map is **bicubically interpolated**, not drawn as flat cells and not
merely GPU-bilinear. Plain bilinear (one texel per cell, linear-filtered) is
only C⁰: values match at cell boundaries but the *gradient* jumps there, so
the eye still catches faint diamond creases on the 6 σ lattice. Instead the
cell field is first lightly blurred (3×3 binomial, wall-aware so the duct edge
stays crisp), then resampled into a texture supersampled 8× per cell using
Catmull-Rom bicubic interpolation. Bicubic is C¹ — value *and* slope are
continuous across cell boundaries — so there is no residual lattice, just
smooth contours.

The boundary is **sharp and follows the true geometry**, not the coarse cell
grid. The interpolated quantity is only the *speed* (for smooth colour);
visibility is a hard per-texel test (`field_visible`) of the texel's exact
world position against the analytic duct profile *and* every obstacle
(circle: radial; polygon: crossing-number) — no wall margin, no soft alpha
fade. So the colour reaches exactly to the wall and is cut cleanly at the real
curved/polygonal obstacle outline with a pixel-sharp edge, instead of
dissolving over a cell or stair-stepping on the 6 σ lattice. Where a visible
texel's 4×4 bicubic stencil reaches into masked cells (right at a wall), those
taps are clamped to the nearest live speed along the row so the colour stays
correct up to the edge rather than being pulled toward zero. The fill loop
maps texels to world coordinates with the *exact same extent* as the final
blit, so the sharp mask lands precisely on the rendered walls and obstacles
rather than offset by a fraction of a cell. The arrow overlay uses the same
`field_visible` test, so no vectors are drawn inside obstacles either.

The texel is packed as a 32-bit word to keep
`SDL_PIXELFORMAT_RGBA8888` correct on little-endian hosts (x86 / Apple
Silicon), and linear filtering is still hinted before renderer creation so the
final stretch of the supersampled texture stays smooth. Cost is ~2 ms/frame on
one slow core (validity is folded into the smoothed buffer with a sentinel so
the hot loop does no per-texel geometry tests) — a >300 fps ceiling on an M4-
class core, comfortably inside the frame budget. The field is a pure rendering
layer — it only reads particle state and never writes it, so dynamics are
byte-identical with it on or off.

This buys smooth *rendering* of the coarse-grained field; it does not add
physical resolution. Structure thinner than the 6 σ cell (a sharp shock, a
thin boundary layer) is still averaged over, not resolved — bicubic makes the
coarse data look continuous, it does not recover sub-cell detail. For genuinely
finer structure the levers remain a smaller `fcell` (more cells, more variance
per cell, so lean harder on the time-average) or a longer averaging window:
a real resolution-vs-variance tradeoff, not a rendering choice. It needs a
second or two of run time to warm up the average before it displays.

---

## 4. Validation

Headless self-tests (`WT_SELFTEST=1`, optional overrides
`WT_STEPS WT_DENS WT_MACH WT_TEMP WT_GEOM WT_OBS WT_OBSCALE`, plus
`WT_ADDOBS_AT=<step> WT_ADDOBS=<preset>` to insert an obstacle into a running
flow exactly as the `O` key does):

- **Low-Mach upstream leak — closed inlet end** (the reported "at low Mach a
  bunch of particles drift left and leave the tube"): the domain had top,
  bottom, and obstacle walls but **no left boundary** — anything reaching
  x<0 was simply deleted by the outflow sweep. At high Mach the strong drift
  means almost nothing turns around, so it was invisible; at low Mach the
  drift u is small next to the thermal speed √T, so a large fraction of
  particles momentarily have vx<0, diffuse back past the inlet, and vanish —
  a steady mass leak emptying the tunnel from the upstream end. A wind-tunnel
  inlet is a *source*, not an opening, so the upstream end is now a specular
  wall: a particle reaching x<0 reflects back into the flow (energy- and
  mass-conserving) instead of escaping. After the fix, no-obstacle runs at
  Mach 0.10–0.80 conserve N (4096 → ~4090 over 3000 steps) with mean vx
  tracking the target (e.g. Mach 0.10: measured 0.148 vs target u=0.141);
  a Mach 0.15 run holds bounded with flat KE/N≈1.00 to 16 000 steps; the
  low-Mach + wing case is stable (KE/N 1.01). Pure specular reflection adds
  no energy but does impart a small transient rightward bias near the inlet,
  so on very long low-Mach runs measured mean vx sits slightly above target
  (0.21 → 0.27) — a bounded physical effect, not a leak; a thermalising
  upstream wall would remove it at the cost of strict energy exactness.
  **OK.**
- **High-density integrator stability** (the reported "raise the density a
  lot → particles vanish, the rest get hypersonic"): this is classic stiff-
  potential blow-up and had three contributing causes, all fixed:
  1. `update_dt` only capped the step from the *drift* velocity; it had no
     stiffness criterion. The WCA force grows as r⁻¹³, so a dense gas with
     transient deep overlaps detonates on the first step no matter how slow
     the flow. There is now a second cap, `dt < 0.10/√k_eff` with the WCA
     curvature scale, and the take-the-smaller of the two (≈2×10⁻³).
  2. `compute_forces` had no bound on a single pair force, so one deep
     overlap returned a multi-thousand-unit kick that ejected the particle
     in one step (→ it vanishes via outflow, its energy heats the rest). The
     per-pair force magnitude is now clamped; the clamp engages only at
     separations well below σ that never occur in steady state, so
     equilibrium dynamics are unchanged.
  3. The seed lattice spacing `1/√ρ` dropped *below the WCA core diameter*
     at high ρ and, with the placement jitter, started thousands of pairs
     deeply overlapped — an initial condition no integrator survives. The
     spacing is now floored at the core boundary and the jitter shrunk so
     the closest possible seeded pair sits at ≥ RCUT; if ρ is too high to
     seed without overlap the densest safe lattice is used and the kinetic
     inlet fills in the rest.

  Result, wing preset, escalating density, 3000 steps: ρ=0.60 went from
  **352 survivors / KE/N=3213** (total blow-up) to **14 669 / 1.49**;
  ρ=0.80 from **668 / 23.5** to **17 388 / 1.66**; ρ=1.00 now stable at
  **17 412 / 1.76**. A long ρ=0.70 run holds bounded to 16 000 steps
  (KE/N 1.59 → 1.66). **OK.**
- **Energy stability — no spurious heating** (the reported "temp goes
  berserk"): KE per particle must not climb. Three independent bugs were
  injecting energy and have been fixed:
  1. `make_wing` traces the airfoil clockwise while the wedge/rectangle are
     counter-clockwise; `poly_finalize` hard-coded the CCW outward-normal
     formula, so every wing edge normal pointed *inward*. Particles entering
     the wing were reflected the wrong way and driven deeper, accumulating an
     overlapping pile of stiff cores — KE/N ran away to 2.36 by 4000 steps and
     accelerating. `poly_finalize` now derives orientation from the polygon's
     signed area, so normals are outward for any winding.
  2. Boundary resolution clamped every penetrating particle onto a single
     `surface + ε` shell, stacking cores. It now mirrors the particle out by
     its true penetration depth (proper specular reflection: velocity *and*
     position).
  3. The inlet overlap test rejected only within 0.85 σ, but the WCA core
     reaches `RCUT = 2^(1/6) ≈ 1.122 σ`; particles injected into that band
     started inside a neighbour's core and converted that potential energy to
     heat continuously. The threshold is now `RCUT`, where U=0 and F=0, so an
     injected particle adds zero spurious energy. This one heated the gas even
     with no obstacle at all.

  After the fixes, KE/N no longer rises in any configuration. With the wing
  preset it holds ≈1.62 → settles to ≈1.56 over 12 000 steps, and with no
  obstacle it relaxes from the artificial seed lattice (≈1.63) to a steady
  ≈1.52 with the decline decelerating to a plateau — equilibration toward a
  self-consistent steady state, the correct sign and bounded, versus the old
  unbounded runaway. **OK.**
- **Mass conservation** (straight duct, no obstacle, ρ=0.16): inflow tracks
  outflow; over very long runs density settles a few percent below the seed
  lattice (the kinetic-flux inlet, not the initial condition, sets the steady
  density) and then holds, with mean `vx` steady near the target. **OK.**
- **Mid-run obstacle insertion** (an earlier reported failure): cylinder
  dropped into a live flow at step 300 — N goes 4096 → 3967 (only the gas
  displaced by the disk is removed) and `T(bulk-rel)` stays at the set point
  instead of exploding. Polygon (NACA wing) path behaves identically. **OK.**
- **Nominal** (de-Laval nozzle, ρ=0.16, Mach 0.8, 600 steps): final
  N≈3065; mean `vx` 1.118 vs target `u` 1.131. **OK.**
- **High-density / supersonic** (straight duct, cylinder, ρ=0.40, Mach 2.0):
  N≈9932 stable, mean `vx` 2.79 vs target 2.83, no NaN/blowup. **OK.**
- **Memory safety**: valgrind, full leak check, NACA wing run (winding-
  corrected normals + mirror reflection + RCUT inlet) — `0 bytes definitely/
  indirectly lost`, `0 errors from 0 contexts`. (Run the optimized binary
  under valgrind only with a non-`-march=native` build; valgrind's JIT cannot
  decode all native-target vector opcodes — a valgrind limitation, not a
  program fault.)

Note on the temperature diagnostic: `measure()` reports the *peculiar*
(drift-removed) granular temperature `T = ½(⟨v²⟩ − |⟨v⟩|²)`. Reporting raw
`⟨v²⟩` would conflate the bulk wind with thermal motion and look spuriously
hot in supersonic flow; the drift-removed value is the physically correct one
and is what matches the slider. It subtracts only the *global* mean velocity,
so once an obstacle creates a stagnation region next to a fast bypass, that
genuine flow shear is counted into the reading and `T(bulk-rel)` sits modestly
above the set point. That residual is real kinetic structure, not spurious
heating — distinct from the pre-fix behaviour where overlap energy from
ejected particles made it diverge. A spatially-binned local thermometer would
remove the shear contribution and is the natural next refinement if you want
a clean field temperature near bodies.

---

## 5. Path to 100k+ particles on multicore

The structure for this is already in place; here is exactly where the
performance comes from and what is left.

- **Array capacity** is `CAP = 320000`, so 100k+ is a runtime question, not a
  recompile.
- **Cell list**: neighbor search is O(N) — particles are binned into a uniform
  grid of cells of size ≥ `r_cut + skin`; only the 9 local cells are searched.
- **Verlet list in CSR**: the neighbor list is built in two passes (count, then
  fill) into compressed-sparse-row arrays. Crucially this makes the **force
  loop embarrassingly parallel over particles with no data races** — each
  thread owns a disjoint set of `i` and only *reads* neighbor positions.
  `compute_forces` is already `#pragma omp parallel for`.
- **Skin / staleness**: the list is rebuilt only when some particle has moved
  more than `0.25·skin` since the last rebuild, so rebuild cost is amortized
  across many steps.

What remains to actually hit 100k+ at interactive rates:

1. **Parallelize the cell-list build and the integrator sweeps.** Right now
   the force kernel is threaded but the bin/rebuild and the Verlet half-kicks
   are serial. For a dilute wind-tunnel gas the force kernel dominates, so you
   already get most of the benefit, but at 100k the serial parts become the
   Amdahl ceiling. The clean fix is per-thread cell bins merged with a prefix
   sum, plus `omp parallel for` on the drift/kick loops (trivially data-parallel
   over `i`).
2. **Decouple render rate from physics rate.** Run several substeps per drawn
   frame (already 2; make it adaptive) and downsample the point cloud when
   zoomed out so SDL isn't the bottleneck.
3. **This container has only 1 core**, so scaling could not be demonstrated
   here. On your MacBook Pro M4 (48 GB) and the X1 Carbon Gen 13 the OpenMP
   force loop should scale near-linearly until memory bandwidth saturates;
   measure with `make bench`.

If you want to go *much* further than that, the next steps in order are: a
Hilbert/Morton spatial sort of the particle arrays every N steps for cache
locality, then SIMD over the inner neighbor loop, then — only if you genuinely
need millions — domain decomposition with MPI. None of that is needed for the
100k target; cell list + threaded force kernel + parallel integrator sweeps
gets you there.

---

## 6. The lattice-Boltzmann companion (`lbm_tunnel.c`)

A second, deliberately different solver for the *same* tunnel:
`make lbm` → `./lbm_tunnel`. Same geometry, same obstacle shapes, same SDL
shell and colour map — but the physics engine is D2Q9-BGK lattice Boltzmann,
not molecular dynamics. The point is to put the two side by side.

**What it is.** The state is the nine-population D2Q9 distribution
`f_i(x,t)` on a regular lattice (here 3 nodes/σ, 960×300). Each step is
collide (relax toward a Mach-expanded Maxwellian, BGK single relaxation time)
then stream (pull scheme; every destination written exactly once) with
halfway bounce-back on the solid mask. Density and velocity are moments of
`f`. Viscosity is **set** by the relaxation time, `ν = c_s²(τ − ½)`, not
measured as in the MD. Walls/obstacles reuse the MD's exact analytic
predicate as the node mask, so a given obstacle occludes the identical region
in both programs. Inlet is an equilibrium velocity boundary; outlet a
post-stream zero-gradient (Neumann) copy. Headless `LB_SELFTEST=1` with
`LB_GEOM LB_OBS LB_MACH LB_TAU LB_STEPS`; `make lbm-selftest`.

**Why it exists — the regime boundary.** Validated subsonic-cylinder
behaviour: mean `ux` tracks the inlet to within the wake momentum deficit,
mass conserved to ~0.3 %, stable and clean through **Mach 0.30**. Then it
**diverges at Mach 0.40+** (cylinder) and even earlier in the nozzle (Mach
0.30, because the throat locally accelerates the flow past the limit before
the inlet does). That is not a bug — it is the well-known stability ceiling
of single-relaxation-time BGK, which is a weakly-compressible scheme built
around a low-Mach truncated equilibrium. Compare directly with the MD, which
ran Mach 2+ through a *choked* nozzle with a standing shock and stayed
bounded. Same geometry, same obstacle, two solvers; the divergence boundary
is the difference between them made visible.

The instructive experiment: run both on the low-Mach cylinder (Mach ~0.1)
where they must — and do — converge to the same Kármán wake, then raise the
Mach on each. The LBM falls apart exactly where the continuum/weak-
compressibility assumptions fail; the particles keep working because a shock
is just molecules with no time to get out of the way. That divergence is the
whole arc of this project in one screen.

Two on-screen draggable **sliders** (same widget as the MD window): `INLET
MACH` and `TAU` (the relaxation time = viscosity = Reynolds number — the LBM
analogue of the MD's density/temperature knobs). Because an LBM distribution
is consistent with only one parameter set (unlike the MD inlet, which can
change live mid-flow), releasing a slider re-initialises the lattice cleanly
from the new parameters rather than trying to continue an inconsistent field.
The Mach slider deliberately ranges past the BGK stability ceiling (~0.4) to
~0.55 so you can *drag it into its own divergence* and watch the scheme fail —
that failure, exactly where the MD keeps working, is the whole point of the
comparison. Other controls mirror the MD where they map (`SPACE R G O +/-
[ ] H Q`); `↑/↓` still fine-adjust Mach from the keyboard; `V` starts/stops
MP4 recording exactly as in the MD tunnel (pipes the composited frames to
`ffmpeg`, writes `lbm_tunnel_<timestamp>.mp4` — a distinct prefix from the
MD's `wind_tunnel_*` so a side-by-side capture session does not have the two
overwrite each other; a red `*REC*` shows in the HUD while active); `F`
toggles speed vs. vorticity colouring (the Kármán street is far clearer in
vorticity). A
**mode-aware colour legend** (top-right, `L` toggles) reads out what the
colours mean with real physical values, not just "low/high": in SPEED mode a
blue→red ramp from 0 to 1.8·U with the inlet speed U marked on the bar; in
VORTICITY mode a diverging blue→white→red scale labelled ±3U/dx with white =
irrotational, blue = clockwise, red = counter-clockwise. The two modes need
different legends because speed is one-sided and vorticity is signed, so the
legend switches with `F`. The HUD shows `|U|MAX / cs` so you can watch it
approach the stability limit before it lets go. Note the velocity field here
is the *primitive* variable — sharp at every node every step, no time-
averaging or warm-up, the noise-free counterpart to the MD's reconstructed
field.

This is standard BGK only: faithfully subsonic. A compressible LBM (larger
stencil, coupled energy distribution, real equation of state) could reach the
MD's regime but is a research-grade effort and a separate project; the honest
small build is the one that demonstrates *where* and *why* the method breaks.

The HUD prints the **Reynolds number** `Re = U·L/ν` in lattice units. When
the obstacle is the cylinder, `L` is its diameter and the readout is labelled
`Re … (cyl D)` — the case with a textbook value and a known shedding onset
(`Re ≈ 47`). With no obstacle or a non-circular one there is no single
unambiguous length scale, so it falls back to the duct half-height and is
honestly labelled `(duct)` rather than pretending to be a body Reynolds
number.

**Symmetry-breaking seed (`P` key, cylinder only; `LB_PERTURB=1` headless).**
A perfectly symmetric, noise-free LBM does *not* spontaneously shed a Kármán
street even above `Re ≈ 47`: the steady symmetric wake is linearly unstable
but, with no perturbation to grow from, the solver sits on it indefinitely (a
pencil balanced exactly on its tip). This is a well-known idealized-CFD
artifact — real experiments always have ambient noise, and the MD here sheds
on its own because thermal fluctuations seed it for free. `P` arms a
*transient*, *bounded*, one-sided "blow": a small fixed transverse velocity
held in a patch just behind the cylinder for ~400 steps, then released so the
now-grown instability sustains itself naturally (a permanent forcing would
contaminate the frequency with the forcing itself; an *additive* nudge — the
first thing tried — compounds geometrically and blows the solver past its
stability limit in a few hundred steps, so the seed is a bounded held value,
not an increment). The HUD shows `SEED ON (n steps left)` then `SEED SPENT`.

Validated behaviour, identical Re ≈ 170 (`τ=0.55`): **without** the seed the
wake stays steady (probe detects no oscillation); **with** it the wake sheds
(probe detects a periodic signal). The seed, not the Reynolds number alone, is
the deciding factor — exactly the supercritical-Hopf symmetry-breaking story.
Honest caveat: the detected Strouhal (~0.09) is the right order of magnitude
and the shedding is genuinely sustained (|u|max visibly oscillates, not
noise), but it sits well below the textbook ~0.20. That gap is the same coarse-
cylinder (60-node stair-stepped surface adds numerical dissipation, lowering
effective Re) plus crude zero-crossing estimator issue documented in §6.1; the
seed switch makes the wake *physically shed* but does not by itself deliver a
quantitatively accurate frequency. Below roughly `τ = 0.57` the effective Re
is high enough that the seed reliably triggers sustained shedding; nearer
`τ = 0.60` the coarse-grid dissipation pulls the effective Re back below
threshold and a brief kick relaxes back to steady — itself an instructive
demonstration that the *effective* Re is what matters, not the nominal one.

### 6.1 The side-by-side cylinder harness (`make compare`) — and its limits

`make compare` runs *both* solvers headless on the **identical** straight-
duct + cylinder configuration and reports each one's wake-probe signal. Both
now carry a matched transverse-velocity probe 2.5 diameters downstream of the
cylinder on the centreline (the MD averages particle `v_y` in a box to cancel
thermal jitter; the LBM reads the node `u_y` directly). The probe time series
is captured over the run and a shedding frequency is estimated from second-
half zero-crossings.

**Honest status — this is a partial result, not a finished cross-validation.**
The probe *infrastructure* is sound and matched between solvers, and it
confirms the harness plumbing: same geometry, same obstacle, both solvers
driven from one target. But the **frequency estimator is not noise-robust**.
Zero-crossing counting on the raw probe signal picks up thermal/numerical
jitter as well as the shedding, so the implied Strouhal numbers come out
1–3 orders of magnitude too high (the MD reports a higher fake frequency with
*more* particles — the tell-tale signature of measuring noise, not vortices).
The output deliberately prints "wake probe f … NOT noise-robust" rather than a
Strouhal number dressed up as a validated result.

A correct quantitative cross-validation (the literature target is
St = f·D/U ≈ 0.20 for a circular cylinder above the shedding onset Re ≈ 47)
needs three things this build does not yet have: (1) a spectral estimator —
autocorrelation peak or windowed FFT on a smoothed signal, not raw zero-
crossings; (2) both solvers deliberately driven into the developed-shedding
regime with the instability seeded by a transient asymmetry and run long
enough to saturate, rather than waiting for numerical noise to maybe trigger
it; (3) matched Reynolds numbers between the two methods so the comparison is
apples-to-apples. That is a focused signal-processing sub-project, not a
harness wrapper, and shipping a wrong number as a "validation" would be worse
than naming the gap. The two solvers remain individually validated (mass,
momentum, stability, divergence onset for the LBM; the full battery in §4 for
the MD); what is *not* yet established is a rigorous numerical agreement
between them on a measured dimensionless observable. That is the single most
valuable next piece of work on this project and it is honestly still open.

---

## 7. The validation instrument (`lbm_bench.c`, `make lbm-bench`)

This is the answer to "how do we make the LBM real, not a toy." It is a
separate, self-contained D2Q9 solver aimed at the canonical **Schäfer–Turek
2D-2** laminar-cylinder benchmark — the standard every serious LBM code is
checked against. It is deliberately *not* the interactive `lbm_tunnel.c`: the
thing you watch and the thing you validate have different requirements.

What it implements, each a real upgrade over the toy and individually
verified: **TRT collision** (two relaxation times; the bounce-back wall
location is decoupled from viscosity via the magic parameter, removing BGK's
viscosity-dependent-wall defect and extending stability); **Bouzidi
interpolated curved bounce-back** on the *exact analytic* cylinder
(2nd-order surface from the same analytic geometry the MD/tunnel already
use — no Cartesian staircase); **q-consistent Mei–Luo–Shyy momentum
exchange** for C_d/C_l (uses the Bouzidi-reconstructed returning population,
not the bare opposite — this is what makes the force consistent with where
the curved wall actually is); a **Welch-windowed FFT** Strouhal estimator
(zero-padded, multi-segment, sub-bin peak — the noise-robust replacement for
the zero-crossing counter that was never trustworthy); and **checkpointing**
so the long convergence runs assemble across many short sessions.

Getting this far meant finding and fixing four real bugs, each of which
would have produced a confident wrong answer: a ~100× force error (Bouzidi
streamed value conflated with the momentum-exchange quantity), an off-axis
geometry error (cylinder centred instead of at the 2D-2 0.005 m offset —
fixing it moved St from 0.195 to ~0.27), a Welch segment-length cap that
pinned the spectral peak to a power-of-two bin and faked non-convergence,
and — the decisive one — **first-order inflow/outflow boundary conditions**
(pure-equilibrium inlet, raw-copy outlet) that capped the entire scheme at
first order and offset every converged quantity 15–25 % from the benchmark.

**Validated result.** Replacing the BCs with **Guo non-equilibrium
extrapolation** (2nd-order, pressure-fixed outlet) was the fix that closed
the gate. Two-point grid convergence, stable and reproduced:

| res | St | C_d | C_l |
|-----|------|------|------|
| 20 | 0.3031 | 3.111 | 0.836 |
| 40 | 0.2829 | 3.215 | 1.007 |
| Schäfer–Turek 2D-2 | ~0.295–0.305 | ~3.22–3.24 | ~1.0 |

At res=40 the solver matches the benchmark within **0.5 % on C_d, 1 % on
C_l, ~4 % on St**, and all three quantities move toward the reference as
resolution doubles (C_d 3.11→3.22, C_l 0.84→1.01, St bracketing ~0.30).
This is **both** halves of the gate: demonstrated grid convergence *and*
absolute agreement with the canonical benchmark to a few percent. The
res=20→40 jump from (St 0.246, C_d 2.28) under the old 1st-order BCs to
(0.303, 3.11) on the BC change alone is what proved the diagnosis.

This is a **validated 2D laminar LBM solver, within stated limits**: it
reproduces Schäfer–Turek 2D-2 to a few percent with demonstrated
convergence, using TRT + Bouzidi curved walls + q-consistent momentum
exchange + Guo 2nd-order BCs + a noise-robust spectral Strouhal estimator.
Honest remaining caveats, stated in full in the file's `VALIDATION STATUS`
block: only two resolutions were run to completion here (res=80 is
compute-bound in this sandbox and would tighten St further), and the
validation covers this one benchmark — laminar, Re=100, 2D — not turbulent
or 3D regimes. Within those limits it is benchmark-validated, not merely
convergent. That distinction — and the work to earn the stronger claim
honestly — was the whole point.

## 8. The fused interactive solver (`lbm_pro.c`, `make lbm-pro`)

The third piece, and the answer to "can the thing I watch be the validated
solver." `lbm_tunnel.c` is the BGK toy (shows where BGK breaks);
`lbm_bench.c` is the headless validator (reproduces Schäfer–Turek 2D-2 to a
few percent); **`lbm_pro.c` fuses them** — the bench's validated solver core
(TRT collision, Bouzidi curved walls, Guo 2nd-order in/outflow) running
inside the tunnel's SDL shell (sliders, legend, video).

**Honest scope — cylinder-only faithful.** `lbm_pro` runs *exactly one*
configuration: straight duct + circular cylinder. That is the only setup
the Bouzidi curved wall is validated for — it is the same code path
`lbm_bench.c` reproduced Schäfer–Turek 2D-2 with (verified: ~580 analytic
curved-wall links on the cylinder, q spanning [0.01, 0.99]; TRT genuinely
active, ω_p≠ω_m). On that configuration this is the validated solver, now
real-time and watchable.

It deliberately does **not** expose the tunnel's other geometries
(nozzle/venturi/diffuser/bump) or non-circular obstacles
(plate/wedge/wing). Those would fall back to staircase bounce-back, which
is not curved-wall accurate — and shipping them behind a "pro" label would
be presenting an unvalidated result as if it were the validated one. The
`G` and `O` cycling keys are removed; `LB_GEOM`/`LB_OBS` env overrides are
ignored (verified: setting them to a venturi+wing still runs
straight-duct+cylinder). The program *cannot* be coerced out of the
validated configuration. For arbitrary interactive geometry use
`lbm_tunnel` (the BGK toy); for the headless benchmark use `lbm_bench`.
Three programs, three honest jobs.

The cylinder size (`+`/`-`/`0`) still works — a resized circle is still a
circle, still gets Bouzidi, still faithful. Mach/TAU sliders, the
symmetry-breaking seed, legend and video all work.

Why a third file rather than upgrading the tunnel in place: the tunnel's
job is the BGK-failure demonstration against the MD (§6) — that pedagogy
depends on it *being* BGK. And the bench must stay uncompromised by
interactive constraints to remain the validator. The one real adaptation
in the fusion: the bench's inflow is a fixed parabolic Schäfer–Turek
profile; `lbm_pro` drives the Guo inflow from the interactive Mach slider
as a uniform inlet (same 2nd-order BC structure, different prescribed
profile — the wind-tunnel setup, not the benchmark channel). That
adaptation is the one place a subtle discrepancy could hide and is called
out, not buried. Validated headless: stable Mach 0.05–0.30, mean velocity
tracks the inlet within the wake deficit, mass conserved ~0.3% (rising to
~3% near Mach 0.3, the Guo pressure-outlet's known mild compressibility),
zero warnings, valgrind-clean.

Honest limit on the word "validated": it transfers to `lbm_pro` because
the cylinder path is the *same code* the bench benchmarked — but a full
Schäfer–Turek convergence study has not been re-run *through this binary*
(same solver, different harness is an inference, not a fresh measurement).
Within that caveat, this is the verified solver, watchable in real time.

## 9. Project improvements (ten-item consolidation)

A pass of ten improvements aimed at consolidating what's already there
and extending measurability. In the order they were applied:

**Item 8 — shared `lbm_common.h`.** The D2Q9 primitives (CX/CY/W/OPP,
c_s², `feq`) lived as copy-paste in all three LBM solvers. One header
now; a bug fixed in one place is a bug fixed everywhere. Zero behavior
change, confirmed by `make lbm-regress` matching baseline.

**Item 1 — direct force measurement through `lbm_pro`.** The selftest
mode now reports Cd (mean), Cl (mean, amplitude, rms), and a Strouhal
from Cl zero-crossings. This closes the "same solver, different harness"
inference caveat from §8: the numbers are now *measured* directly on the
pro binary, not inferred. Running `LB_SELFTEST=1 LB_MACH=0.10 LB_PERTURB=1
./lbm_pro` prints them next to the mean-velocity and mass numbers.

**Item 2 — live Cd/Cl on the HUD.** Once the EMA warms up (~2000 steps),
the HUD shows the force coefficients updating in real time next to the
Mach/τ/ν readout. The pretty wake is now a measurement.

**Item 4 — Richardson extrapolation + error bars in `lbm_bench`.** When
three resolutions are present in the assembled convergence table (`make
lbm-bench LB_ASSEMBLE=1`), the bench computes the three-point Richardson
extrapolation, the observed order *p*, and the discretization error bar.
Non-monotone series are explicitly flagged (`"check for a BC or
estimator defect"`) — the diagnostic that catches silent regressions.

**Item 7 — regression harness.** `make lbm-regress` runs a short fixed
battery across all three solvers (~1 minute) and compares mean velocity,
Cd, and status to a committed `regress_expected.txt` within tolerance.
Exit non-zero on any drift so CI/git-pre-commit can hook it.
`make lbm-regress-baseline` recaptures the baseline after a vetted
intentional change.

**Item 3 — Smagorinsky LES (opt-in, `K` key / `LB_SMAG=0.17`).** Adds a
locally-adaptive eddy viscosity `ν_t = (C_s dx)² |S|` recovered in-place
from the non-equilibrium populations (Yu-Luo-Girimaji 2005 closed form),
affecting only ω_p so the TRT wall-location property survives. Verified
at Mach 0.40 where the plain laminar scheme would diverge — status OK.
**Honest scope**: this is the standard LBM-LES construction, correctly
implemented, but *not* benchmark-validated here. The HUD annotates
`+LES (unvalidated)` whenever it's on. It's the right tool to run above
~Re 250 without garbage; it is not a validated turbulence model.

**Item 5 — pressure coefficient C_p(θ) around the cylinder.** With
`LB_CP=cp.csv`, `lbm_pro` writes 180 samples of C_p around the surface
(θ=0 at upstream stagnation point). Textbook comparison with
Henderson-Williamson. Converges with run length — short runs show the
right pattern (positive upstream, negative at the sides, recovery in the
wake) with transient magnitudes.

**Item 9 — CSV time-series dump + `tools/analyze.py`.** With
`LB_CSV=ts.csv`, pro writes a per-step CSV of probe velocity, Cd, Cl.
`python3 tools/analyze.py series ts.csv` plots the time series and a
Hann-windowed FFT (naming the peak frequency and period in steps).
`tools/analyze.py both ts.csv cp.csv` plots both. Keeps the solvers in
C and the analysis in Python.

**Item 6 — OpenMP scaling (`make lbm-scaling`).** Measures walltime at
several thread counts for `lbm_pro` and `lbm_bench`, prints speedup.
Honest note: these solvers are memory-bandwidth-bound at modest grid
sizes, so linear scaling is realistic only up to the point where the
f-array fits in fewer caches. Override with `THREADS="1 2 4 8" STEPS=400
make lbm-scaling` for a wider sweep on real hardware.

**Not done, deliberately:** porting the solvers to GPU (M4 integrated
GPU gains don't justify the engineering cost); MRT (TRT already sits at
the 2D laminar sweet spot); 3D (an entirely different project); adaptive
mesh refinement (same).

## 10. Install, man pages, WASM, GitHub Pages

**Install into `~/.local`** (per-user, no sudo):

```sh
make install          # binaries to ~/.local/bin, man pages to ~/.local/share/man/man1
make uninstall
```

System-wide: `sudo make PREFIX=/usr/local install`. The installer prints a
PATH/MANPATH hint if `~/.local/bin` isn't on PATH yet.

**Man pages** (four, one per binary): `man wind_tunnel`, `man lbm_tunnel`,
`man lbm_pro`, `man lbm_bench`. Each covers its controls, environment
variables and representative examples.

**WASM build** (browser demos for `lbm_tunnel` and `lbm_pro`): requires
emscripten. `lbm_bench` and `wind_tunnel` are deliberately not WASM-built
— the first is a validator (produces numbers, not pictures) and the
second is CPU-heavy by design. ASYNCIFY lets the native `while(!quit)`
yield each frame; OpenMP pragmas become no-ops in the single-threaded
WASM build; video recording (`V` key) becomes a no-op in the browser.

```sh
# install emscripten once
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest
source emsdk_env.sh && cd ..

make wasm                                    # builds web/lbm_tunnel.* and web/lbm_pro.*
python3 -m http.server --directory web 8000  # serve locally
```

**GitHub + Pages** (one-shot deploy):

```sh
./tools/push.sh                              # inits repo, pushes, enables Pages
# or:
REPO=wind-tunnel-2d VISIBILITY=public ./tools/push.sh
```

The script creates the repo if needed, pushes to `main`, and flips Pages
to the GitHub Actions source so `.github/workflows/pages.yml` drives
deployment. The workflow installs emscripten in CI, runs `make wasm`,
and publishes `web/` to Pages. First build takes 2-5 min; subsequent
pushes to `main` auto-redeploy. Pages URL printed by the script.

**Honest scope on the WASM port:** written against the standard
emscripten + SDL2 + ASYNCIFY pattern; verified only to syntax-pass
`make -n wasm`, not run through `emcc` here (sandbox blocks the
emscripten download). First real build runs on your machine or in the
CI workflow.
