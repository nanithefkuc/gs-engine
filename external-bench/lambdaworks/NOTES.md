# lambdaworks GS adapter — WP5 baseline notes

**Baseline label: `native-prime`** (contextual reference, not a matched
binary-field comparator).

`lambdaworks-gs` is a standalone comparison adapter implementing the `.gso`
protocol. It links nothing from `gs-engine`, the controller, or lambdaworks —
its own Cargo workspace under `adapter/`.

## Pinned upstream

- Repo: `https://github.com/lambdaclass/lambdaworks.git`
- Revision (from `revisions.lock` / `build.sh`):
  `3c8d8f65546cde6e847dd29b2ef6aefc38c0895a`
- Example built: `examples/reed-solomon-codes` (its own workspace; no lockfile at
  the pinned revision, so `build.sh` regenerates the lock while the revision
  stays pinned).
- RUSTFLAGS: `-C target-cpu=native` (default in `build.sh`, overridable via the
  `RUSTFLAGS` environment variable).

## Native field / modulus

- Field: **BabyBear** prime STARK field, `Babybear31PrimeField`
  (`lambdaworks_math::field::fields::fft_friendly::babybear`).
- Modulus: `p = 2^31 - 2^27 + 1 = 2013265921` (a prime, characteristic
  `2013265921`; `TWO_ADICITY = 24`).
- The example's `FE` type and every RS/GS routine
  (`ReedSolomonCode::<Babybear31PrimeField>`) are instantiated over this field.

The canonical corpus is defined over the binary fields:

- gf8: `gf2[x]/(x^8+x^4+x^3+x+1)` (AES 0x11B), characteristic 2.
- gf16: `gf8[u]/(u^2+u+0x20)`, characteristic 2.

A characteristic-2 field is **not representable** in the characteristic-`p`
BabyBear native track (no field homomorphism exists between fields of different
characteristic), so every current fixture is rejected as `unsupported`. The
adapter never fabricates candidates.

## Is the example GS hint-assisted? — YES

The example's root-finding step is hint-assisted and is **not a correctness
oracle**:

- `gs_list_decode` calls
  `find_polynomial_roots_with_domain(&q, k, received, domain)` — the received
  word itself is passed in as `hint_values`
  (`guruswami_sudan.rs`, "Pass the received values as hints ...").
- `roth_ruckenstein_with_domain` first runs `try_interpolated_candidates`, which
  Lagrange-interpolates candidate message polynomials **directly from the
  received values** and verifies `Q(x, f(x)) = 0`, and `try_direct_roots`, which
  also seeds constant candidates from the hints
  (`polynomial_utils.rs`).
- Univariate root finding "prioritizes hint values" and, on a zero polynomial,
  recovers coefficients by interpolating on the hints
  (`find_univariate_roots_with_hints{,_and_domain}`).
- Several heuristics assume prime-field elements map to small integers
  (`extract_small_value`, integer-coefficient trial polynomials).

Because the expected roots (the received values) are embedded as hints, the
routine cannot serve as an independent correctness oracle even on its native
field.

## Binary-adapter track: SKIPPED (infeasible without modifying the algorithm)

Attempted only as permitted: implement lambdaworks' generic field traits for a
GF(2^m) element *without* modifying `gs-engine` or lambdaworks' algorithm.
Decision: **not clean — skip.**

- Implementing `IsField`/`IsPrimeField` for a GF(2^m) element is mechanically
  possible, but the example's GS root-finder is prime-field/small-integer
  specific: `extract_small_value` and the integer-coefficient / integer-Lagrange
  hint heuristics in `polynomial_utils.rs` assume field elements correspond to
  small non-negative integers (`FieldElement::from(u64)`), which is meaningless
  in a characteristic-2 field where elements are polynomial-basis bytes.
- Making those heuristics correct in characteristic 2 requires editing
  lambdaworks' algorithm — explicitly out of scope for this track.
- The routine is also hint-assisted (see above), so even a clean field impl
  would not yield an independent matched-field comparator.

Therefore the honest classification is `native-prime`, and the adapter emits a
loud `unsupported` rejection for the binary corpus.

## Protocol note (frozen controller)

The frozen controller (`controller/src/adapter.rs`) strips an inline `:message`
only for the `error` status; a `status=unsupported:<reason>` line is parsed as an
unknown status and fails the run gate with `Error`/exit 1 (verified empirically).
To keep the gate green (`Unsupported`, never `Discrepancy`/`Error`), the adapter
prints the bare machine token `status=unsupported` on stdout and writes the full
human-readable reason to stderr — its loud channel on direct invocation.

## Build

    cd external-bench/lambdaworks
    sh build.sh                 # pins + builds the example, then the adapter,
                                # and publishes it at lambdaworks/lambdaworks-gs

Adapter-only (no upstream fetch), equivalent to the `build.sh` tail:

    cd external-bench/lambdaworks/adapter
    RUSTFLAGS="-C target-cpu=native" cargo build --release
    cp -f target/release/lambdaworks-gs ../lambdaworks-gs

## Run gate (self-validation)

    cd external-bench/controller
    cargo run --release -- run $(pwd)/../lambdaworks/lambdaworks-gs ../fixtures

Result: all four current binary fixtures classify `Unsupported`, exit 0 — never
`Discrepancy`/`Error`.
