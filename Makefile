# Makefile — spring_grid
# Linux (sdl2-config / pkg-config) and macOS/Homebrew (Apple Silicon + Intel)
#   make            optimised build        ->  ./spring_grid
#   make debug      ASan + UBSan build      ->  ./spring_grid_debug
#   make test       headless physics tests (no SDL needed)
#   make install    binary -> ~/.local/bin, man page -> ~/.local/share/man/man1
#   make uninstall  remove both
#   make clean
#
# Install location: override PREFIX (e.g. make install PREFIX=/usr/local);
# DESTDIR is honoured for staged installs.

CC     ?= cc
TARGET  = spring_grid

CORE    = src/params.c src/modes.c src/util.c src/sim.c src/spectrum.c
GUI     = src/font.c src/render.c src/main.c
HDRS    = $(wildcard src/*.h)

# ── SDL2 ──────────────────────────────────────────────────────────────────────
SDL2_CONFIG := $(shell command -v sdl2-config 2>/dev/null || \
                 ls /opt/homebrew/bin/sdl2-config 2>/dev/null || \
                 ls /usr/local/bin/sdl2-config 2>/dev/null)
SDL2_CFLAGS  = $(shell $(SDL2_CONFIG) --cflags)
SDL2_LIBS    = $(shell $(SDL2_CONFIG) --libs)

# ── FFTW3 ─────────────────────────────────────────────────────────────────────
FFTW_CFLAGS := $(shell pkg-config --cflags fftw3 2>/dev/null || \
    ([ -f /opt/homebrew/include/fftw3.h ] && echo -I/opt/homebrew/include) || \
    ([ -f /usr/local/include/fftw3.h    ] && echo -I/usr/local/include))
FFTW_LIBS   := $(shell pkg-config --libs fftw3 2>/dev/null || \
    ([ -f /opt/homebrew/include/fftw3.h ] && echo -L/opt/homebrew/lib -lfftw3) || \
    ([ -f /usr/local/include/fftw3.h    ] && echo -L/usr/local/lib -lfftw3) || \
    echo -lfftw3)

WARN    = -std=c17 -Wall -Wextra -Wpedantic
RFLAGS  = -O2
DFLAGS  = -O0 -g3 -fsanitize=address,undefined

# ── install locations ─────────────────────────────────────────────────────────
PREFIX  ?= $(HOME)/.local
BINDIR   = $(DESTDIR)$(PREFIX)/bin
MANDIR   = $(DESTDIR)$(PREFIX)/share/man/man1
MANPAGE  = man/spring_grid.1

.PHONY: all debug test clean check-sdl install uninstall

all: $(TARGET)

check-sdl:
	@[ -n "$(SDL2_CONFIG)" ] || { echo "sdl2-config not found. Install SDL2: brew install sdl2  OR  apt install libsdl2-dev"; exit 1; }

$(TARGET): $(CORE) $(GUI) $(HDRS) | check-sdl
	$(CC) $(WARN) $(RFLAGS) $(SDL2_CFLAGS) $(FFTW_CFLAGS) $(CORE) $(GUI) -o $@ $(SDL2_LIBS) $(FFTW_LIBS) -lm

$(TARGET)_debug: $(CORE) $(GUI) $(HDRS) | check-sdl
	$(CC) $(WARN) $(DFLAGS) $(SDL2_CFLAGS) $(FFTW_CFLAGS) $(CORE) $(GUI) -o $@ $(SDL2_LIBS) $(FFTW_LIBS) -lm

debug: $(TARGET)_debug

tests/test_sim: tests/test_sim.c $(CORE) $(HDRS)
	$(CC) $(WARN) $(RFLAGS) $(FFTW_CFLAGS) $(CORE) tests/test_sim.c -o $@ $(FFTW_LIBS) -lm

test: tests/test_sim
	./tests/test_sim

install: $(TARGET) $(MANPAGE)
	install -d $(BINDIR) $(MANDIR)
	install -m 755 $(TARGET) $(BINDIR)/$(TARGET)
	install -m 644 $(MANPAGE) $(MANDIR)/spring_grid.1
	@echo "installed $(BINDIR)/$(TARGET)"
	@echo "installed $(MANDIR)/spring_grid.1"
	@case ":$$PATH:" in *":$(PREFIX)/bin:"*) ;; \
	  *) echo "note: $(PREFIX)/bin is not on your PATH; add  export PATH=\"$(PREFIX)/bin:\$$PATH\"  to your shell rc" ;; esac

uninstall:
	rm -f $(BINDIR)/$(TARGET) $(MANDIR)/spring_grid.1
	@echo "removed $(BINDIR)/$(TARGET) and $(MANDIR)/spring_grid.1"

clean:
	rm -f $(TARGET) $(TARGET)_debug tests/test_sim
