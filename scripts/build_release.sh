#!/usr/bin/env bash
set -euo pipefail

# Determine repository root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PACKAGE_NAME="chromatic_cognition_core"
VERSION="$(grep '^version' Cargo.toml | head -n1 | cut -d '"' -f2)"
TARGET="x86_64-unknown-linux-musl"
RELEASE_DIR="${REPO_ROOT}/release"
ARTIFACT_DIR="${RELEASE_DIR}/${PACKAGE_NAME}-${VERSION}"
ARCHIVE_NAME="${PACKAGE_NAME}-${VERSION}.tar.gz"

# Ensure we start from a clean slate
rm -rf "${ARTIFACT_DIR}"
mkdir -p "${ARTIFACT_DIR}/bin" "${ARTIFACT_DIR}/config"

# Make sure the MUSL target is available for static builds.
if ! rustup target list --installed | grep -q "^${TARGET}$"; then
  rustup target add "${TARGET}"
fi

# Build the statically linked binary.
CARGO_FLAGS=(--locked --release --target "${TARGET}")
cargo build "${CARGO_FLAGS[@]}"

BINARY_PATH="target/${TARGET}/release/${PACKAGE_NAME}"
if [[ ! -f "${BINARY_PATH}" ]]; then
  echo "Expected binary not found at ${BINARY_PATH}" >&2
  exit 1
fi

cp "${BINARY_PATH}" "${ARTIFACT_DIR}/bin/"

# Copy finalized configuration files.
cp config/engine.toml "${ARTIFACT_DIR}/config/"
cp config/bridge.toml "${ARTIFACT_DIR}/config/"

# Include the FINAL_BASELINE performance report.
cp docs/FINAL_BASELINE.txt "${ARTIFACT_DIR}/"

# Create a deterministic archive.
rm -f "${RELEASE_DIR}/${ARCHIVE_NAME}"
tar -C "${RELEASE_DIR}" -czf "${RELEASE_DIR}/${ARCHIVE_NAME}" "$(basename "${ARTIFACT_DIR}")"

echo "Release package created: ${RELEASE_DIR}/${ARCHIVE_NAME}"
