# Build and Packaging Guide

This guide defines the auditable process for producing the Chromatic Cognition Core release artifact.

## 1. Prerequisites

1. Install the stable Rust toolchain (1.70 or later).
2. Add the MUSL target to the toolchain:
   ```bash
   rustup target add x86_64-unknown-linux-musl
   ```
3. Ensure the MUSL toolchain is available on the host (for example, `musl-tools` on Debian/Ubuntu or `musl-gcc` on other distributions). This step enables fully static linking without containerization.

## 2. Build Command

The repository ships with a `.cargo/config.toml` that pins the build target to `x86_64-unknown-linux-musl` and forces static C runtime linkage. Use the release profile with link-time optimization enabled:

```bash
cargo build --release
```

The resulting binary is located at:

```
target/x86_64-unknown-linux-musl/release/chromatic_cognition_core
```

Because the target is MUSL, the executable is statically linked and portable across standard Linux distributions.

## 3. Artifact Verification

1. Confirm the binary is statically linked:
   ```bash
   file target/x86_64-unknown-linux-musl/release/chromatic_cognition_core
   ```
   The output must report `statically linked`.
2. Generate a checksum for integrity tracking:
   ```bash
   sha256sum target/x86_64-unknown-linux-musl/release/chromatic_cognition_core > chromatic_cognition_core.sha256
   ```
3. Record the toolchain version used:
   ```bash
   rustc --version --verbose
   ```

## 4. Deployment Notes

- No containerization is required or supported for the release artifact.
- Copy the compiled binary and the generated checksum to the target host.
- Re-run the checksum on the target host to verify integrity before execution.

Following this procedure guarantees a reproducible, portable release artifact that satisfies the build and packaging specification.
