# --- MD wind tunnel -------------------------------------------------------
# Cross-platform. Linux: GCC + -fopenmp. macOS: Apple Clang has NO built-in
# OpenMP, so we use Homebrew libomp:  brew install libomp
# -------------------------------------------------------------------------
UNAME_S := $(shell uname -s)

CC      ?= cc
CSTD     = -std=c17
# Genuinely zero-warning at this bar. Add -Wconversion -Wsign-conversion
# for a numeric-cast audit (expect intentional float<->int graphics casts).
WARN     = -Wall -Wextra -Wpedantic -Wshadow

# SDL2: prefer pkg-config, fall back to sdl2-config (Homebrew ships both).
SDL_CF  := $(shell pkg-config --cflags sdl2 2>/dev/null || sdl2-config --cflags)
SDL_LD  := $(shell pkg-config --libs   sdl2 2>/dev/null || sdl2-config --libs)

ifeq ($(UNAME_S),Darwin)
  # Apple Clang already optimizes for the host CPU; -march=native is an x86
  # idiom and errors on Apple Silicon, so we omit it here.
  OPT      = -O3 -funroll-loops
  LIBOMP  := $(shell brew --prefix libomp 2>/dev/null)
  # -isystem (not -I): libomp's vendored omp.h uses C23-range enumerators
  # that trip -Wpedantic/-Wc23-extensions. Treating it as a system header
  # silences those *without* relaxing the warning bar on our own code.
  OMP_CF   = -Xpreprocessor -fopenmp -isystem $(LIBOMP)/include
  OMP_LD   = -L$(LIBOMP)/lib -lomp -Wl,-rpath,$(LIBOMP)/lib
else
  OPT      = -O3 -march=native -funroll-loops
  OMP_CF   = -fopenmp
  OMP_LD   = -fopenmp
endif

CFLAGS   = $(CSTD) $(WARN) $(OPT) $(OMP_CF) $(SDL_CF)
LDFLAGS  = $(OMP_LD)
LDLIBS   = $(SDL_LD) -lm

# lbm_bench.c is headless (no SDL): same compiler/OpenMP handling as the
# SDL targets -- crucially the platform-aware $(OMP_CF)/$(OMP_LD), so it
# builds with Homebrew libomp on macOS Apple Clang and plain -fopenmp on
# Linux GCC -- but without the SDL cflags/libs.
BENCH_CFLAGS = $(CSTD) $(WARN) $(OPT) $(OMP_CF)
BENCH_LDLIBS = $(OMP_LD) -lm

TARGET   = wind_tunnel
SRC      = wind_tunnel.c
LBM      = lbm_tunnel
LBMSRC   = lbm_tunnel.c
PRO      = lbm_pro
PROSRC   = lbm_pro.c
BENCH    = lbm_bench
BENCHSRC = lbm_bench.c

all: omp-check $(TARGET) $(LBM) $(PRO) $(BENCH)

# the lattice-Boltzmann companion solver (separate binary, shared shell)
lbm: omp-check $(LBM)

# lbm_pro: the validated bench solver (TRT + Bouzidi + Guo 2nd-order BCs)
# fused into the interactive SDL shell. Same SDL/OpenMP handling as the
# other interactive targets (so it builds on macOS Apple Clang too).
lbm-pro: omp-check $(PRO)

# On macOS, fail early with a clear message if libomp is missing.
omp-check:
ifeq ($(UNAME_S),Darwin)
	@test -n "$(LIBOMP)" || { \
	  echo ""; \
	  echo "  ERROR: Homebrew libomp not found."; \
	  echo "  Apple Clang has no built-in OpenMP. Install it with:"; \
	  echo ""; \
	  echo "      brew install libomp"; \
	  echo ""; \
	  exit 1; }
endif

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

$(LBM): $(LBMSRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

$(PRO): $(PROSRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# headless physics / scaling check (no window)
selftest: $(TARGET)
	WT_SELFTEST=1 WT_STEPS=600 ./$(TARGET)

lbm-selftest: $(LBM)
	LB_SELFTEST=1 LB_STEPS=2000 ./$(LBM)

# Research-grade validation instrument (Schaefer-Turek 2D-2 cylinder:
# TRT + Bouzidi curved wall + q-consistent momentum exchange + Guo
# 2nd-order BCs + Welch-FFT Strouhal + checkpointed grid-convergence.
# Reproduces the Schaefer-Turek 2D-2 cylinder benchmark to a few percent
# at res=40 with demonstrated grid convergence. See the VALIDATION STATUS
# block at the top of lbm_bench.c for the exact numbers and caveats.
lbm-bench: omp-check $(BENCHSRC)
	$(CC) $(BENCH_CFLAGS) -o $(BENCH) $(BENCHSRC) $(BENCH_LDLIBS)

# convergence runs are long; checkpointing lets them resume across calls
lbm-bench-run: lbm-bench
	./$(BENCH) 20 40

# Side-by-side cylinder cross-validation. Runs BOTH solvers on the same
# straight-duct + cylinder configuration and prints the Strouhal number
# each predicts, next to the literature value (~0.20). This is the honest
# comparison: not "do they look alike" but "do two unrelated numerical
# methods agree on a measured dimensionless observable, and does each
# agree with reality". MD needs more steps (smaller dt, thermal averaging).
compare: $(TARGET) $(LBM)
	@echo "=============================================================="
	@echo " SIDE-BY-SIDE: cylinder wake, both solvers, same geometry"
	@echo "=============================================================="
	@echo " HONEST STATUS: this runs both solvers on the identical"
	@echo " straight-duct+cylinder config and reports each one's wake-"
	@echo " probe frequency. The probe infrastructure is matched and"
	@echo " sound, but the zero-crossing frequency estimator is NOT yet"
	@echo " noise-robust -- it counts thermal/numerical jitter, not just"
	@echo " shedding, so the St values are not trustworthy. A correct"
	@echo " Strouhal cross-validation needs a spectral estimator and"
	@echo " matched Reynolds numbers (README sec 6, 'validation gap')."
	@echo " The solvers themselves are individually validated."
	@echo "--------------------------------------------------------------"
	@echo "--- molecular dynamics ---"
	@WT_SELFTEST=1 WT_GEOM=0 WT_OBS=1 WT_MACH=0.30 WT_DENS=0.20 \
	 WT_STEPS=12000 ./$(TARGET) | grep -E "wake probe|status|mean v_x"
	@echo "--- lattice Boltzmann (D2Q9-BGK) ---"
	@LB_SELFTEST=1 LB_GEOM=0 LB_OBS=1 LB_MACH=0.10 \
	 LB_STEPS=8000 ./$(LBM) | grep -E "wake probe|status|mean ux"
	@echo "=============================================================="

bench: $(TARGET)
	WT_SELFTEST=1 WT_STEPS=400 OMP_NUM_THREADS=1 ./$(TARGET)
	WT_SELFTEST=1 WT_STEPS=400 ./$(TARGET)

# Regression harness: runs a fixed battery across the three LBM solvers
# and compares against regress_expected.txt. Catches silent numerical
# drift when editing collision/stream/BCs. See regress.sh for the test
# matrix and tolerances. Use `make lbm-regress-baseline` ONLY after a
# vetted intentional change.
lbm-regress: all lbm lbm-pro lbm-bench
	@./regress.sh

lbm-regress-baseline: all lbm lbm-pro lbm-bench
	@./regress.sh --baseline

# OpenMP scaling: wall-time vs OMP_NUM_THREADS for lbm_pro and lbm_bench.
# Override THREADS/STEPS/RES/MACH env for a different sweep.
lbm-scaling: lbm-pro lbm-bench
	@./tools/scaling.sh

# -------------------------------------------------------------------------
#  WASM / browser build (Emscripten).  Builds the two interactive solvers
#  (lbm_tunnel and lbm_pro) into web/ as self-contained .html/.js/.wasm
#  triples. lbm_bench is deliberately not WASM-built: it's a headless
#  validator, not an interactive demo, and native binaries are faster.
#  wind_tunnel is also skipped -- its particle count + rendering is CPU-
#  heavy and the point of the browser demos is the LBM comparison.
#
#  Build locally:
#      curl -L https://github.com/emscripten-core/emsdk/archive/main.tar.gz | tar xz
#      cd emsdk-main && ./emsdk install latest && ./emsdk activate latest
#      source emsdk_env.sh
#      cd .. && make wasm
#
#  Serve for local testing (SharedArrayBuffer not needed for single-threaded):
#      python3 -m http.server --directory web 8000
#      # visit http://localhost:8000/
# -------------------------------------------------------------------------
EMCC ?= emcc
EM_CFLAGS = -std=c17 -O3 -Wall -Wextra -Wpedantic -Wshadow \
            -s USE_SDL=2
# ASYNCIFY lets the native while(!quit) loop yield to the browser via
# emscripten_sleep() each frame. Keep ASYNCIFY_STACK_SIZE large enough
# for the deepest call site under the loop; 65536 is comfortable here.
EM_LDFLAGS = -s USE_SDL=2 -s ASYNCIFY=1 -s ASYNCIFY_STACK_SIZE=65536 \
             -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=128MB \
             -s EXIT_RUNTIME=0 -s ENVIRONMENT=web \
             --shell-file web/shell.html

WASM_OUT_TUNNEL = web/lbm_tunnel.html
WASM_OUT_PRO    = web/lbm_pro.html

wasm: $(WASM_OUT_TUNNEL) $(WASM_OUT_PRO)

$(WASM_OUT_TUNNEL): $(LBMSRC) lbm_common.h web/shell.html
	$(EMCC) $(EM_CFLAGS) -o $@ $(LBMSRC) $(EM_LDFLAGS)

$(WASM_OUT_PRO): $(PROSRC) lbm_common.h web/shell.html
	$(EMCC) $(EM_CFLAGS) -o $@ $(PROSRC) $(EM_LDFLAGS)

wasm-clean:
	rm -f web/lbm_tunnel.html web/lbm_tunnel.js web/lbm_tunnel.wasm \
	      web/lbm_pro.html    web/lbm_pro.js    web/lbm_pro.wasm

# -------------------------------------------------------------------------
#  install / uninstall into ~/.local by default (per-user, no sudo needed).
#  Override PREFIX for a system-wide install:  sudo make PREFIX=/usr/local install
#  BINDIR, MANDIR are derived; override them for distro packaging.
# -------------------------------------------------------------------------
PREFIX ?= $(HOME)/.local
BINDIR  = $(PREFIX)/bin
MANDIR  = $(PREFIX)/share/man/man1
BINS    = $(TARGET) $(LBM) $(PRO) $(BENCH)
MANS    = man/wind_tunnel.1 man/lbm_tunnel.1 man/lbm_pro.1 man/lbm_bench.1

install: all
	@install -d $(BINDIR) $(MANDIR)
	@install -m 755 $(BINS) $(BINDIR)
	@install -m 644 $(MANS) $(MANDIR)
	@echo "installed into $(PREFIX):"
	@for b in $(BINS); do echo "  $(BINDIR)/$$b"; done
	@for m in $(MANS); do echo "  $(MANDIR)/$$(basename $$m)"; done
	@case ":$$PATH:" in *":$(BINDIR):"*) : ;; *) \
	  echo ""; \
	  echo "  NOTE: $(BINDIR) is not on PATH. Add this to ~/.bashrc or ~/.zshrc:"; \
	  echo "        export PATH=\"$(BINDIR):\$$PATH\""; \
	  echo "  and  (for man):"; \
	  echo "        export MANPATH=\"$(PREFIX)/share/man:\$$MANPATH\""; \
	;; esac

uninstall:
	@rm -f $(BINDIR)/wind_tunnel $(BINDIR)/lbm_tunnel \
	       $(BINDIR)/lbm_pro     $(BINDIR)/lbm_bench
	@rm -f $(MANDIR)/wind_tunnel.1 $(MANDIR)/lbm_tunnel.1 \
	       $(MANDIR)/lbm_pro.1     $(MANDIR)/lbm_bench.1
	@echo "removed from $(PREFIX)"

clean:
	rm -f $(TARGET) $(LBM) $(PRO) $(BENCH)

.PHONY: all lbm lbm-pro clean selftest lbm-selftest lbm-bench lbm-bench-run bench compare omp-check lbm-regress lbm-regress-baseline lbm-scaling install uninstall wasm wasm-clean
