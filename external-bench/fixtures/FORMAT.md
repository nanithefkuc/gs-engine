# GS decoder fixture format

`.gsf` is a line-oriented, versioned interchange format for decoder inputs and
complete expected candidate sets. It is intentionally parseable without a JSON
library so the Rust, C, and C++ harnesses consume identical bytes.

## Version 1

The first line is exactly:

```text
gs-engine-fixture-v1
```

It is followed by these `key=value` records in order:

```text
name=<ASCII identifier>
field=gf8|gf16
field-definition=<ASCII field description>
domain=arbitrary|additive|affine
n=<decimal>
k=<decimal>
target-radius=<decimal>
multiplicity=<decimal>
y-degree=<decimal>
weighted-degree=<decimal>
support=<element>[,<element>...]
received=<element>[,<element>...]
expected-candidate=<element>[,<element>...]
[expected-candidate=...]
[expected-codeword=<element>[,<element>...]]
```

Singleton keys occur exactly once. `expected-candidate` and
`expected-codeword` may repeat. Blank lines, comments, unknown keys, duplicate
singleton keys, signs, whitespace around values, and trailing commas are
invalid. Files end with one LF byte.

Each element is lowercase hexadecimal encoding of the field's fixed-width
little-endian representation: two hex digits for GF8 and four for GF16. A
candidate value lists polynomial coefficients from constant term upward. No
trailing zero coefficient is allowed, except that the zero polynomial is one
zero element. Support and received arrays contain exactly `n` elements.
Expected codewords, when present, also contain exactly `n` elements.

Expected candidates are a complete set, sorted first by polynomial degree and
then lexicographically by their packed little-endian coefficient bytes.
Duplicate candidates are invalid. Comparator-specific output order is sorted
outside timed regions before comparison.

The `field-definition` value is semantic, not commentary. Version 1 fixtures
use these exact definitions:

- `gf2[x]/(x^8+x^4+x^3+x+1);le-polynomial-basis`
- `gf8[u]/(u^2+u+0x20);gf8=aes-0x11b;le-components`

An adapter using an isomorphic representation records and validates an explicit
forward/inverse map before timing. The `support` array is always authoritative;
`domain` describes its construction but never permits an adapter to reorder it.

## Batch version 1

A batch fixture carries several received words sharing one geometry and one
expected candidate set, for parallel/sequential batch-decode comparison. The
first line is:

```text
gs-engine-batch-v1
```

The records are the same singletons as version 1 up to and including
`support=`, then:

```text
received-words=<count>
received-word=<element>[,<element>...]
[received-word=...]
expected-candidate=<element>[,<element>...]
[expected-candidate=...]
```

`received-words` gives the exact number of `received-word` records that follow,
each `n` elements. The expected candidate set is shared: every received word
is a corruption of the same codeword and decodes to it, so the batch does
uniform work. Validation decodes every word and requires each to produce the
shared set. The element encoding and ordering rules are identical to version 1.
