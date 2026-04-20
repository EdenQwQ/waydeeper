#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()    { echo -e "${GREEN}[waydeeper]${NC} $*"; }
warn()    { echo -e "${YELLOW}[waydeeper]${NC} $*"; }
error()   { echo -e "${RED}[waydeeper]${NC} $*" >&2; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install waydeeper on non-Nix systems.

Options:
  -p, --prefix DIR     Installation prefix (default: /usr/local for root, ~/.local for user)
  -u, --user           Install to user directory (~/.local) instead of system-wide
  -h, --help           Show this help message

Examples:
  $(basename "$0")                  # Auto-detect: user dir if not root, /usr/local if root
  $(basename "$0") --user           # Force user install (~/.local)
  $(basename "$0") --prefix /opt    # Custom prefix
  sudo $(basename "$0")             # System-wide install to /usr/local
EOF
    exit 0
}

PREFIX=""
USER_INSTALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--prefix) PREFIX="$2"; shift 2 ;;
        -u|--user)   USER_INSTALL=true; shift ;;
        -h|--help)   usage ;;
        *) error "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$PREFIX" ]]; then
    if $USER_INSTALL || [[ "$(id -u)" -ne 0 ]]; then
        PREFIX="$HOME/.local"
        info "Installing to user directory: $PREFIX"
    else
        PREFIX="/usr/local"
        info "Installing system-wide to: $PREFIX"
    fi
else
    info "Installing to custom prefix: $PREFIX"
fi

BIN_DIR="$PREFIX/bin"

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        error "$1 is required but not installed."
        return 1
    fi
}

info "Checking prerequisites..."
check_cmd cargo || exit 1
check_cmd rustc || exit 1
check_cmd pkg-config || exit 1
check_cmd cmake || exit 1

for lib in wayland-client wayland-protocols xkbcommon egl gl; do
    if ! pkg-config --exists "$lib" 2>/dev/null; then
        warn "pkg-config: $lib not found — build may fail. Install the corresponding -dev package."
    fi
done

if ! pkg-config --exists libonnxruntime 2>/dev/null; then
    warn "onnxruntime not found via pkg-config. You may need to set ORT_DYLIB_PATH after install."
fi

info "Building waydeeper (release mode)..."
cargo build --release

info "Installing to $PREFIX..."
mkdir -p "$BIN_DIR"
cp "target/release/waydeeper" "$BIN_DIR/waydeeper"
chmod +x "$BIN_DIR/waydeeper"
info "Binary installed to: $BIN_DIR/waydeeper"

if ! echo "$PATH" | tr ':' '\n' | grep -qxF "$BIN_DIR"; then
    warn "$BIN_DIR is not in your PATH. Add it with:"
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
    echo ""
    warn "Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.) to make it permanent."
fi

if ! pkg-config --exists libonnxruntime 2>/dev/null; then
    ORT_PATH=""
    for candidate in \
        /usr/lib/libonnxruntime.so \
        /usr/lib/x86_64-linux-gnu/libonnxruntime.so \
        /usr/local/lib/libonnxruntime.so \
        /opt/onnxruntime/lib/libonnxruntime.so; do
        if [[ -f "$candidate" ]]; then
            ORT_PATH="$candidate"; break
        fi
    done
    if [[ -n "$ORT_PATH" ]]; then
        info "Found onnxruntime at: $ORT_PATH"
        info "Set ORT_DYLIB_PATH before running:"
        echo "  export ORT_DYLIB_PATH=\"$ORT_PATH\""
    else
        warn "Could not find libonnxruntime.so. Install onnxruntime and set ORT_DYLIB_PATH."
    fi
fi

echo ""
info "Installation complete!"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Download depth estimation models?${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "waydeeper requires at least one depth model to work."
echo ""
echo "Available models:"
echo "  1. depth-anything-v3-base (default, balanced)"
echo "  2. midas-small (fast, lower quality)"
echo "  3. depth-pro-q4 (high quality, slow)"
echo ""
echo -n "Download now? [y/N]: "
read -r answer

if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo ""
    info "Running model downloader..."
    export PATH="$BIN_DIR:$PATH"
    "$BIN_DIR/waydeeper" download-model
else
    info "You can download models later with: waydeeper download-model"
fi

echo ""
info "Next steps:"
echo "  waydeeper --help"
echo "  waydeeper set /path/to/wallpaper.jpg"
