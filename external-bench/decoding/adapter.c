/*
 * DECODING adapter: decoding-gs
 *
 * Standalone comparison adapter for the gs-engine external benchmark harness.
 * It adapts Guillaume Quintin's DECODING library (GPL, linked ONLY into this
 * separate executable) to the `.gso` protocol.
 *
 *   decoding-gs <fixture.gsf>
 *
 * prints:
 *   status=radius|unsupported:<reason>|error:<message>
 *   candidate=<hex>[,<hex>...]   (message polynomial, constant term first)
 *
 * Fields:
 *   gf8  = gf2[x]/(x^8+x^4+x^3+x+1)          (AES 0x11B, le polynomial basis)
 *   gf16 = gf8[u]/(u^2+u+0x20), gf8 = AES     (le components: low byte deg-0)
 *
 * DECODING's gf2n_word ring uses FIXED irreducible polynomials
 * (include/decoding/rings/gf2n_word_irr.h):
 *   m=8  : x^8 + x^4 + x^3 + x + 1        = 0x11B   (identical to canonical gf8)
 *   m=16 : x^16 + x^5 + x^3 + x + 1       = 0x1002B (differs from canonical gf16)
 *
 * The canonical field arithmetic below is implemented clean-room from the
 * contract/FORMAT.md spec. For each field we build an explicit GF(2)-linear
 * isomorphism canonical<->DECODING by matching a primitive element through its
 * minimal polynomial, then map support/received IN and codewords OUT.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <decoding/decoding.h>

/* Two DECODING rings in one translation unit (see samples/sample3.c). Each is
 * prefixed by its RING_NAME so there is no symbol clash. ring_reset.h (pulled
 * in by algos.c) undefines RING_NAME/EXT_DEGREE between the two includes. */
#define RING_NAME gf8d
#define EXT_DEGREE 8
#include <decoding/rings/gf2n_word.c>
#include <decoding/algos.c>

#define RING_NAME gf16d
#define EXT_DEGREE 16
#include <decoding/rings/gf2n_word.c>
#include <decoding/algos.c>

/* ===================================================================== *
 *  Canonical field arithmetic (clean-room)                              *
 * ===================================================================== */

/* gf8: multiply in gf2[x]/(0x11B), little-endian polynomial basis. */
static unsigned cmul8(unsigned a, unsigned b) {
    unsigned r = 0;
    int i;
    a &= 0xff;
    b &= 0xff;
    for (i = 0; i < 8; i++) {
        if (b & 1u) r ^= a;
        b >>= 1;
        a <<= 1;
        if (a & 0x100u) a ^= 0x11Bu;
    }
    return r & 0xff;
}

/* gf16 tower: element v = a0 | (a1<<8) over gf8[u]/(u^2+u+0x20).
 * (a0+a1 u)(b0+b1 u) = (a0 b0 + 0x20 a1 b1) + (a0 b1 + a1 b0 + a1 b1) u. */
static unsigned cmul16(unsigned a, unsigned b) {
    unsigned a0 = a & 0xff, a1 = (a >> 8) & 0xff;
    unsigned b0 = b & 0xff, b1 = (b >> 8) & 0xff;
    unsigned a1b1 = cmul8(a1, b1);
    unsigned c0 = cmul8(a0, b0) ^ cmul8(0x20u, a1b1);
    unsigned c1 = cmul8(a0, b1) ^ cmul8(a1, b0) ^ a1b1;
    return (c0 & 0xff) | ((c1 & 0xff) << 8);
}

static unsigned cmul(int m, unsigned a, unsigned b) {
    return (m == 8) ? cmul8(a, b) : cmul16(a, b);
}

static unsigned cfpow(int m, unsigned a, unsigned long e) {
    unsigned r = 1;
    while (e) {
        if (e & 1u) r = cmul(m, r, a);
        a = cmul(m, a, a);
        e >>= 1;
    }
    return r;
}

static unsigned cinv(int m, unsigned a) {
    /* a^(2^m - 2) */
    return cfpow(m, a, ((1UL << m) - 2UL));
}

/* ===================================================================== *
 *  DECODING field arithmetic wrappers (faithful to the linked library)  *
 * ===================================================================== */

static unsigned dmul8(unsigned a, unsigned b) {
    gf8d ra, rb, rr;
    ra[0] = a; rb[0] = b;
    gf8d_mul(rr, ra, rb);
    return rr[0];
}

static unsigned dmul16(unsigned a, unsigned b) {
    gf16d ra, rb, rr;
    ra[0] = a; rb[0] = b;
    gf16d_mul(rr, ra, rb);
    return rr[0];
}

static unsigned dmul(int m, unsigned a, unsigned b) {
    return (m == 8) ? dmul8(a, b) : dmul16(a, b);
}

/* ===================================================================== *
 *  GF(2)-linear isomorphism canonical <-> DECODING                      *
 * ===================================================================== */

struct isomorphism {
    int m;              /* extension degree */
    unsigned size;      /* 1 << m */
    unsigned *fwd;       /* canonical -> DECODING */
    unsigned *inv;       /* DECODING  -> canonical */
    unsigned alpha;      /* canonical primitive element */
    unsigned beta;       /* its DECODING image (root of same minpoly) */
    unsigned minpoly;    /* minimal polynomial bits (canonical/GF(2)) */
};

/* prime factors of 2^m - 1 for the fields we support */
static int order_factors(int m, unsigned long *primes) {
    if (m == 8) {           /* 255 = 3 * 5 * 17 */
        primes[0] = 3; primes[1] = 5; primes[2] = 17;
        return 3;
    }
    /* m == 16: 65535 = 3 * 5 * 17 * 257 */
    primes[0] = 3; primes[1] = 5; primes[2] = 17; primes[3] = 257;
    return 4;
}

static unsigned find_primitive_canonical(int m) {
    unsigned long order = (1UL << m) - 1UL;
    unsigned long primes[4];
    int np = order_factors(m, primes);
    unsigned g;
    for (g = 2; g < (1u << m); g++) {
        int ok = 1, i;
        for (i = 0; i < np; i++) {
            if (cfpow(m, g, order / primes[i]) == 1u) { ok = 0; break; }
        }
        if (ok) return g;
    }
    return 0; /* unreachable for valid fields */
}

/* Minimal polynomial of primitive alpha over GF(2), as a bit mask.
 * p(X) = prod_{i=0}^{m-1} (X - alpha^{2^i}); coefficients live in {0,1}. */
static unsigned minpoly_of(int m, unsigned alpha) {
    unsigned coeff[17];      /* canonical-field coefficients, deg <= 16 */
    int deg = 0, i;
    unsigned root = alpha;
    coeff[0] = 1;            /* start with the constant polynomial 1 */
    for (i = 0; i < m; i++) {
        /* multiply current poly by (X - root) == (X + root) in char 2 */
        int j;
        unsigned carry = 0; /* coeff[deg+1] slot */
        for (j = deg + 1; j >= 0; j--) {
            unsigned hi = (j >= 1) ? coeff[j - 1] : 0;          /* X * lower */
            unsigned lo = (j <= deg) ? cmul(m, coeff[j], root) : 0; /* root*  */
            coeff[j] = hi ^ lo;
        }
        (void)carry;
        deg++;
        root = cmul(m, root, root); /* next conjugate: square */
    }
    /* coefficients must all be 0 or 1 (GF(2)); pack to a bit mask */
    {
        unsigned mask = 0;
        for (i = 0; i <= deg; i++) {
            if (coeff[i] == 1u) mask |= (1u << i);
            else if (coeff[i] != 0u) return 0; /* not a GF(2) polynomial */
        }
        return mask;
    }
}

/* Evaluate a GF(2)-coefficient polynomial (bit mask) at d in the DECODING
 * field: XOR of d^i over the set bits i. */
static unsigned eval_minpoly_decoding(int m, unsigned mask, unsigned d) {
    unsigned acc = 0, dp = 1; /* d^0 */
    int i;
    for (i = 0; i <= m; i++) {
        if (mask & (1u << i)) acc ^= dp;
        dp = dmul(m, dp, d);
    }
    return acc;
}

static unsigned find_beta_decoding(int m, unsigned mask) {
    unsigned d;
    for (d = 2; d < (1u << m); d++) {
        if (eval_minpoly_decoding(m, mask, d) == 0u) return d;
    }
    return 0; /* unreachable: minpoly has m roots in the field */
}

/* Build forward/inverse tables via matched generator powers. */
static int build_iso(struct isomorphism *iso, int m) {
    unsigned i, size = (1u << m);
    unsigned cur_c, cur_d;

    iso->m = m;
    iso->size = size;
    iso->fwd = (unsigned *)malloc(sizeof(unsigned) * size);
    iso->inv = (unsigned *)malloc(sizeof(unsigned) * size);
    if (!iso->fwd || !iso->inv) return -1;

    iso->alpha = find_primitive_canonical(m);
    iso->minpoly = minpoly_of(m, iso->alpha);
    if (iso->minpoly == 0) return -1;
    iso->beta = find_beta_decoding(m, iso->minpoly);
    if (iso->beta == 0) return -1;

    for (i = 0; i < size; i++) { iso->fwd[i] = 0; iso->inv[i] = 0; }
    iso->fwd[0] = 0;
    iso->inv[0] = 0;
    cur_c = 1;  /* alpha^0 in canonical field (mult. identity) */
    cur_d = 1;  /* beta^0  in DECODING field                    */
    for (i = 0; i < size - 1; i++) {
        iso->fwd[cur_c] = cur_d;
        iso->inv[cur_d] = cur_c;
        cur_c = cmul(m, cur_c, iso->alpha);
        cur_d = dmul(m, cur_d, iso->beta);
    }
    return 0;
}

/* Verify homomorphism. Exhaustive for gf8; deterministic sample for gf16.
 * Returns 0 on success, -1 on failure (with *why set). */
static unsigned lcg_state;
static unsigned lcg_next(void) {
    lcg_state = lcg_state * 1103515245u + 12345u;
    return lcg_state;
}

static int verify_iso(const struct isomorphism *iso, const char **why) {
    int m = iso->m;
    unsigned size = iso->size;

    if (iso->fwd[0] != 0) { *why = "fwd(0)!=0"; return -1; }
    if (iso->fwd[1] != 1) { *why = "fwd(1)!=1"; return -1; }

    if (m == 8) {
        unsigned a, b;
        for (a = 0; a < size; a++) {
            /* inverse preservation */
            if (a != 0) {
                if (dmul(m, iso->fwd[a], iso->fwd[cinv(m, a)]) != 1u) {
                    *why = "inverse not preserved"; return -1;
                }
            }
            for (b = 0; b < size; b++) {
                if (iso->fwd[a ^ b] != (iso->fwd[a] ^ iso->fwd[b])) {
                    *why = "addition not preserved"; return -1;
                }
                if (iso->fwd[cmul(m, a, b)] != dmul(m, iso->fwd[a], iso->fwd[b])) {
                    *why = "multiplication not preserved"; return -1;
                }
            }
        }
        return 0;
    }

    /* gf16: large deterministic sample */
    {
        long trials = 4000000L, t;
        lcg_state = 0xC0FFEEu;
        for (t = 0; t < trials; t++) {
            unsigned a = lcg_next() & (size - 1);
            unsigned b = lcg_next() & (size - 1);
            if (iso->fwd[a ^ b] != (iso->fwd[a] ^ iso->fwd[b])) {
                *why = "addition not preserved"; return -1;
            }
            if (iso->fwd[cmul(m, a, b)] != dmul(m, iso->fwd[a], iso->fwd[b])) {
                *why = "multiplication not preserved"; return -1;
            }
            if (a != 0 && dmul(m, iso->fwd[a], iso->fwd[cinv(m, a)]) != 1u) {
                *why = "inverse not preserved"; return -1;
            }
        }
        return 0;
    }
}

/* ===================================================================== *
 *  DECODING Guruswami-Sudan glue (one wrapper per ring)                 *
 * ===================================================================== */

/* Each wrapper builds an rs_code with the mapped support, a received word in
 * the DECODING representation, runs GS at radius tau, and copies the returned
 * codewords (DECODING representation) out as flat unsigned arrays. */
#define GEN_RUN_GS(FN, RING)                                                   \
static int FN(const unsigned *dsupp, const unsigned *drecv,                    \
              int n, int k, int tau, unsigned **out, int *out_nc) {            \
    RING##_rs_code rs;                                                         \
    RING##_vec y;                                                              \
    RING##_vec *c;                                                             \
    uma nc, i, j;                                                              \
    RING##_rs_code_init_with_support(rs, (uma)n, (uma)k);                      \
    for (i = 0; i < (uma)n; i++) rs[0].supp[0].a[i][0] = dsupp[i];             \
    RING##_vec_init(y);                                                        \
    RING##_vec_adjust_size(y, (uma)n);                                         \
    for (i = 0; i < (uma)n; i++) y[0].a[i][0] = drecv[i];                      \
    nc = RING##_rs_code_guruswami_sudan_koetter(&c, rs, y, (uma)tau);          \
    *out_nc = (int)nc;                                                         \
    *out = NULL;                                                               \
    if (nc > 0) {                                                              \
        unsigned *buf = (unsigned *)malloc(sizeof(unsigned) * (size_t)nc * n); \
        for (j = 0; j < nc; j++)                                              \
            for (i = 0; i < (uma)n; i++) buf[j * n + i] = c[j][0].a[i][0];     \
        *out = buf;                                                            \
        RING##_vec_ptr_clear(c, nc);                                           \
    }                                                                          \
    RING##_vec_clear(y);                                                       \
    RING##_rs_code_clear(rs);                                                  \
    return 0;                                                                  \
}

GEN_RUN_GS(run_gs_gf8, gf8d)
GEN_RUN_GS(run_gs_gf16, gf16d)

/* ===================================================================== *
 *  Message interpolation (Lagrange over the canonical field)            *
 * ===================================================================== */

/* Interpolate the degree-<k message polynomial from the first k codeword
 * positions. x[] canonical support, v[] canonical codeword values.
 * Result coefficients (constant term first) written to f[0..k-1]. */
static void interpolate_message(int m, const unsigned *x, const unsigned *v,
                                int k, unsigned *f) {
    int j, l, d;
    for (j = 0; j < k; j++) f[j] = 0;
    for (j = 0; j < k; j++) {
        unsigned num[8];      /* numerator poly, deg < k <= 8 here */
        int numdeg = 0;
        unsigned denom = 1, scale;
        num[0] = 1;
        for (l = 0; l < k; l++) {
            if (l == j) continue;
            /* num *= (X + x[l]) */
            for (d = numdeg + 1; d >= 0; d--) {
                unsigned hi = (d >= 1) ? num[d - 1] : 0;
                unsigned lo = (d <= numdeg) ? cmul(m, num[d], x[l]) : 0;
                num[d] = hi ^ lo;
            }
            numdeg++;
            denom = cmul(m, denom, x[j] ^ x[l]);
        }
        scale = cmul(m, v[j], cinv(m, denom));
        for (d = 0; d <= numdeg; d++)
            f[d] ^= cmul(m, num[d], scale);
    }
}

/* ===================================================================== *
 *  Fixture parsing (strict enough; corpus is pre-validated)             *
 * ===================================================================== */

struct fixture {
    int m;            /* 8 or 16 */
    int n, k, tau;
    unsigned *support;  /* canonical values, length n */
    unsigned *received; /* canonical values, length n */
};

static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

/* Parse one canonical element token into a canonical packed value. */
static int parse_element(int m, const char *tok, size_t len, unsigned *out) {
    size_t want = (m == 8) ? 2 : 4;
    unsigned bytes[2];
    size_t i;
    if (len != want) return -1;
    for (i = 0; i < want / 2; i++) {
        int hi = hexval((unsigned char)tok[2 * i]);
        int lo = hexval((unsigned char)tok[2 * i + 1]);
        if (hi < 0 || lo < 0) return -1;
        bytes[i] = (unsigned)((hi << 4) | lo);
    }
    /* byte0 = degree-0 (low), byte1 = degree-1 (high) */
    *out = (m == 8) ? bytes[0] : (bytes[0] | (bytes[1] << 8));
    return 0;
}

/* Parse a comma-separated element list into a freshly allocated array.
 * Returns count, or -1 on error. */
static int parse_elements(int m, const char *val, unsigned **out) {
    size_t cap = 8, cnt = 0;
    unsigned *arr = (unsigned *)malloc(sizeof(unsigned) * cap);
    const char *p = val;
    for (;;) {
        const char *q = p;
        while (*q && *q != ',') q++;
        if (q == p) { free(arr); return -1; }
        if (cnt == cap) { cap *= 2; arr = (unsigned *)realloc(arr, sizeof(unsigned) * cap); }
        if (parse_element(m, p, (size_t)(q - p), &arr[cnt]) != 0) { free(arr); return -1; }
        cnt++;
        if (!*q) break;
        p = q + 1;
    }
    *out = arr;
    return (int)cnt;
}

/* trim trailing newline/carriage return in place */
static void rstrip(char *s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = 0;
}

static int parse_fixture(const char *path, struct fixture *fx, const char **err) {
    FILE *fp = fopen(path, "rb");
    char line[8192];
    int have_field = 0, have_n = 0, have_k = 0, have_tau = 0;
    int nsupp = -1, nrecv = -1;
    if (!fp) { *err = "cannot open fixture"; return -1; }

    fx->support = fx->received = NULL;
    fx->m = fx->n = fx->k = fx->tau = 0;

    if (!fgets(line, sizeof line, fp)) { *err = "empty fixture"; fclose(fp); return -1; }
    rstrip(line);
    if (strcmp(line, "gs-engine-fixture-v1") != 0) {
        *err = "bad magic line"; fclose(fp); return -1;
    }

    while (fgets(line, sizeof line, fp)) {
        char *eq;
        rstrip(line);
        if (line[0] == 0) continue;
        eq = strchr(line, '=');
        if (!eq) { *err = "line without '='"; fclose(fp); return -1; }
        *eq = 0;
        {
            const char *key = line;
            const char *val = eq + 1;
            if (strcmp(key, "field") == 0) {
                if (strcmp(val, "gf8") == 0) fx->m = 8;
                else if (strcmp(val, "gf16") == 0) fx->m = 16;
                else { *err = "unknown field"; fclose(fp); return -1; }
                have_field = 1;
            } else if (strcmp(key, "n") == 0) {
                fx->n = atoi(val); have_n = 1;
            } else if (strcmp(key, "k") == 0) {
                fx->k = atoi(val); have_k = 1;
            } else if (strcmp(key, "target-radius") == 0) {
                fx->tau = atoi(val); have_tau = 1;
            } else if (strcmp(key, "support") == 0) {
                if (!have_field) { *err = "support before field"; fclose(fp); return -1; }
                nsupp = parse_elements(fx->m, val, &fx->support);
                if (nsupp < 0) { *err = "bad support"; fclose(fp); return -1; }
            } else if (strcmp(key, "received") == 0) {
                if (!have_field) { *err = "received before field"; fclose(fp); return -1; }
                nrecv = parse_elements(fx->m, val, &fx->received);
                if (nrecv < 0) { *err = "bad received"; fclose(fp); return -1; }
            }
            /* other keys are irrelevant to decoding; ignore. */
        }
    }
    fclose(fp);

    if (!have_field || !have_n || !have_k || !have_tau) {
        *err = "missing required key"; return -1;
    }
    if (fx->n <= 0 || fx->k <= 0 || fx->k > fx->n || fx->tau < 0) {
        *err = "bad n/k/tau"; return -1;
    }
    if (nsupp != fx->n || nrecv != fx->n) {
        *err = "support/received length != n"; return -1;
    }
    return 0;
}

/* ===================================================================== *
 *  Output                                                               *
 * ===================================================================== */

static void print_element(int m, unsigned v) {
    if (m == 8) printf("%02x", v & 0xff);
    else printf("%02x%02x", v & 0xff, (v >> 8) & 0xff);
}

/* Print one candidate (message polynomial), trailing zeros trimmed. */
static void print_candidate(int m, const unsigned *f, int len) {
    int hi = len - 1, i;
    while (hi > 0 && f[hi] == 0) hi--;
    printf("candidate=");
    for (i = 0; i <= hi; i++) {
        if (i) printf(",");
        print_element(m, f[i]);
    }
    printf("\n");
}

/* ===================================================================== *
 *  Decode drivers                                                       *
 * ===================================================================== */

/* k == 1 repetition code: DECODING's GS parameter formulas divide by (k-1),
 * so decode the radius directly in the canonical field (complete set for the
 * radius): every constant whose codeword is within tau of the received word. */
static void decode_k1(const struct fixture *fx) {
    unsigned c, size = (1u << fx->m);
    struct timespec _t0, _t1; clock_gettime(CLOCK_MONOTONIC, &_t0);
    for (c = 0; c < size; c++) {
        int dist = 0, i;
        for (i = 0; i < fx->n; i++)
            if (fx->received[i] != c) dist++;
        if (dist <= fx->tau) {
            unsigned f = c;
            print_candidate(fx->m, &f, 1);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &_t1);
    long long _dns = (long long)(_t1.tv_sec - _t0.tv_sec) * 1000000000LL + (long long)(_t1.tv_nsec - _t0.tv_nsec);
    printf("status=radius\n");
    printf("decode-ns=%lld\n", _dns);
}

static void decode_general(const struct fixture *fx, const struct isomorphism *iso) {
    int m = fx->m, n = fx->n, k = fx->k, tau = fx->tau, nc = 0, j;
    unsigned *dsupp = (unsigned *)malloc(sizeof(unsigned) * n);
    unsigned *drecv = (unsigned *)malloc(sizeof(unsigned) * n);
    unsigned *codewords = NULL;
    int i;

    for (i = 0; i < n; i++) {
        dsupp[i] = iso->fwd[fx->support[i]];
        drecv[i] = iso->fwd[fx->received[i]];
    }

    struct timespec _t0, _t1;
    if (m == 8) { clock_gettime(CLOCK_MONOTONIC, &_t0); run_gs_gf8(dsupp, drecv, n, k, tau, &codewords, &nc); clock_gettime(CLOCK_MONOTONIC, &_t1); }
    else        { clock_gettime(CLOCK_MONOTONIC, &_t0); run_gs_gf16(dsupp, drecv, n, k, tau, &codewords, &nc); clock_gettime(CLOCK_MONOTONIC, &_t1); }
    long long _dns = (long long)(_t1.tv_sec - _t0.tv_sec) * 1000000000LL + (long long)(_t1.tv_nsec - _t0.tv_nsec);

    printf("status=radius\n");
    printf("decode-ns=%lld\n", _dns);
    for (j = 0; j < nc; j++) {
        unsigned cx[64], cv[64], f[8];
        for (i = 0; i < n; i++) {
            cx[i] = fx->support[i];                 /* canonical support */
            cv[i] = iso->inv[codewords[j * n + i]]; /* codeword -> canonical */
        }
        interpolate_message(m, cx, cv, k, f);
        print_candidate(m, f, k);
    }

    free(codewords);
    free(dsupp);
    free(drecv);
}

/* ===================================================================== *
 *  main / diagnostics                                                   *
 * ===================================================================== */

/* --dump <8|16>: write the raw forward table (little-endian, width bytes per
 * entry) to stdout so the build script can fingerprint it with sha256sum. */
static int dump_forward(int m) {
    struct isomorphism iso;
    unsigned i;
    if (build_iso(&iso, m) != 0) return 1;
    for (i = 0; i < iso.size; i++) {
        unsigned v = iso.fwd[i];
        if (m == 8) {
            unsigned char b = (unsigned char)(v & 0xff);
            fwrite(&b, 1, 1, stdout);
        } else {
            unsigned char b[2];
            b[0] = (unsigned char)(v & 0xff);
            b[1] = (unsigned char)((v >> 8) & 0xff);
            fwrite(b, 1, 2, stdout);
        }
    }
    return 0;
}

/* --iso <8|16>: print the ring/poly/generator diagnostics to stdout. */
static int print_iso_info(int m) {
    struct isomorphism iso;
    const char *why = "";
    if (build_iso(&iso, m) != 0) { printf("build failed\n"); return 1; }
    printf("field=gf%d\n", m);
    printf("decoding_irr_poly=0x%X\n", (m == 8) ? 0x11Bu : 0x1002Bu);
    printf("canonical_primitive=0x%X\n", iso.alpha);
    printf("minpoly_bits=0x%X\n", iso.minpoly);
    printf("decoding_image_beta=0x%X\n", iso.beta);
    printf("verify=%s\n", (verify_iso(&iso, &why) == 0) ? "ok" : why);
    return 0;
}

int main(int argc, char **argv) {
    struct fixture fx;
    struct isomorphism iso;
    const char *err = "", *why = "";

    gf8d_ring_init();
    gf16d_ring_init();

    if (argc == 3 && strcmp(argv[1], "--dump") == 0)
        return dump_forward(atoi(argv[2]));
    if (argc == 3 && strcmp(argv[1], "--iso") == 0)
        return print_iso_info(atoi(argv[2]));

    if (argc != 2) {
        fprintf(stderr, "usage: %s <fixture.gsf>\n", argv[0]);
        printf("status=error:usage\n");
        return 2;
    }

    if (parse_fixture(argv[1], &fx, &err) != 0) {
        fprintf(stderr, "parse error: %s\n", err);
        printf("status=error:%s\n", err);
        return 1;
    }

    if (fx.k == 1) {
        decode_k1(&fx);
        free(fx.support); free(fx.received);
        return 0;
    }

    if (build_iso(&iso, fx.m) != 0) {
        printf("status=error:isomorphism build failed\n");
        free(fx.support); free(fx.received);
        return 1;
    }
    if (verify_iso(&iso, &why) != 0) {
        /* Loud, required rejection rather than a wrong candidate set. */
        printf("status=unsupported:gf%d isomorphism verification failed (%s)\n",
               fx.m, why);
        free(fx.support); free(fx.received);
        return 0;
    }

    decode_general(&fx, &iso);
    free(iso.fwd); free(iso.inv);
    free(fx.support); free(fx.received);
    return 0;
}
