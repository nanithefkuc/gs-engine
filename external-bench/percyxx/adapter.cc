// percyxx-gs: standalone Guruswami-Sudan adapter wrapping Percy++'s
// RSDecoder_GF2E (Kotter interpolation + Roth-Ruckenstein factorization) for
// the WP5 .gsf protocol. Links only NTL + the Percy++ RS-decoder support
// objects; never the PIR client/server.
//
// Field mapping: gf8 is the identity (NTL GF2E with the AES modulus 0x11B is
// byte-for-byte the canonical field). gf16 builds an explicit GF(2)-linear
// isomorphism between the canonical tower gf8[u]/(u^2+u+0x20) and Percy++'s
// native GF(2^16) modulus x^16+x^5+x^3+x^2+1 (0x1002D), by matching a primitive
// element's minimal polynomial and aligning powers.
//
// The decoder (interpolate_kotter) chooses its own internal multiplicity m and
// list size L from (v, n, t), so every decode is reported as status=radius
// (same code/radius, decoder-chosen internal parameters).

#define TEST_FINDPOLYS 1

#include <map>
#include <algorithm>
#include <NTL/GF2X.h>
#include <NTL/GF2E.h>
// These counters are `extern`-declared in rsdecoder_impl.h and defined in
// rsdecoder.cc (which we deliberately do not compile — it drags in the ZZ_p
// specialization). Provide the definitions here so the GF2E template bodies
// instantiated from this TU link.
uint64_t hasseop = 0, kotter_usec = 0;
#include <NTL/GF2EXFactoring.h>
#include <NTL/vec_GF2E.h>

#include "rsdecoder.h"
#include "rsdecoder_impl.h"
#include "gf2e.h"   // GF28_mult_table (AES), used for the clean-room tower field

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <chrono>
#include <unordered_map>
#include <set>

NTL_CLIENT
using namespace std;

// ---------------------------------------------------------------------------
// Canonical gf8 (AES 0x11B) and gf16 tower (gf8[u]/(u^2+u+0x20)) arithmetic.
//
// gf8 element: one byte, bit j = coefficient of x^j (AES polynomial basis).
//   add = XOR, mul = AES multiply (reuse Percy++'s verified GF28_mult_table).
//
// gf16 element: two bytes (low = degree-0 gf8 component a0, high = degree-1
//   component a1), element = a0 + a1*u. We pack as canon16 = (a1<<8)|a0.
//   add = XOR of both bytes (so canon16 add = XOR).
//   mul: (a0+a1u)(b0+b1u) = a0b0 + (a0b1+a1b0)u + a1b1 u^2, and
//        u^2 = u + 0x20, so a1b1 u^2 -> a1b1*0x20 (low) + a1b1 (high).
//        low  = a0b0 ^ (a1b1 mul 0x20)
//        high = a0b1 ^ a1b0 ^ a1b1
// ---------------------------------------------------------------------------

static inline uint8_t gf8_mul(uint8_t a, uint8_t b) { return GF28_mult_table[a][b]; }

static inline uint16_t gf16_add(uint16_t a, uint16_t b) { return a ^ b; }

static inline uint16_t gf16_mul(uint16_t a, uint16_t b) {
    uint8_t a0 = a & 0xff, a1 = (a >> 8) & 0xff;
    uint8_t b0 = b & 0xff, b1 = (b >> 8) & 0xff;
    uint8_t a1b1 = gf8_mul(a1, b1);
    uint8_t lo = gf8_mul(a0, b0) ^ gf8_mul(a1b1, 0x20);
    uint8_t hi = gf8_mul(a0, b1) ^ gf8_mul(a1, b0) ^ a1b1;
    return ((uint16_t)hi << 8) | lo;
}

static uint16_t gf16_pow(uint16_t a, unsigned long e) {
    uint16_t r = 0x0001; // 1
    while (e) {
        if (e & 1) r = gf16_mul(r, a);
        a = gf16_mul(a, a);
        e >>= 1;
    }
    return r;
}

// Order of a in the multiplicative group (0 -> 0).
static unsigned long gf16_order(uint16_t a) {
    if (a == 0) return 0;
    uint16_t acc = 0x0001;
    for (unsigned long i = 1; i <= 65535; ++i) {
        acc = gf16_mul(acc, a);
        if (acc == 0x0001) return i;
    }
    return 0; // unreachable for nonzero
}

// Find a primitive element (multiplicative order 65535) of the canonical
// gf16 tower by testing candidates with the prime-factor test.
static uint16_t gf16_find_primitive() {
    // 65535 = 3 * 5 * 17 * 257.
    static const unsigned long primes[4] = {3, 5, 17, 257};
    for (uint32_t c = 2; c < 65536; ++c) {
        uint16_t cc = (uint16_t)c;
        bool prim = true;
        for (int q = 0; q < 4 && prim; ++q) {
            if (gf16_pow(cc, 65535 / primes[q]) == 0x0001) prim = false;
        }
        if (prim) return cc;
    }
    return 0; // unreachable
}

// Minimal polynomial of a primitive element g over GF(2) via the Frobenius
// product prod_{k=0}^{15} (x - g^(2^k)). Coefficients come out as GF(2)
// elements (canon16 0x0000 or 0x0001). Returns the polynomial as a GF2X
// (bit i = coefficient of x^i), monic of degree 16.
static GF2X canon_minpoly(uint16_t g) {
    // conjugates a_k = g^(2^k)
    uint16_t conj[16];
    conj[0] = g;
    for (int k = 1; k < 16; ++k) conj[k] = gf16_mul(conj[k-1], conj[k-1]);
    // polynomial with canonical-field coefficients (uint16), degree grows to 16
    vector<uint16_t> poly;
    poly.push_back(0x0001); // 1
    for (int k = 0; k < 16; ++k) {
        uint16_t a = conj[k];
        // poly <- poly * (x + a)  (char 2, x - a == x + a)
        vector<uint16_t> npoly(poly.size() + 1, 0);
        for (size_t d = 0; d < poly.size(); ++d) {
            npoly[d + 1] ^= poly[d];              // shift up (multiply by x)
            npoly[d]     ^= gf16_mul(poly[d], a);  // multiply by a
        }
        poly.swap(npoly);
    }
    GF2X m;
    for (size_t i = 0; i < poly.size(); ++i) {
        if (poly[i] == 0x0001) SetCoeff(m, (long)i);
        else if (poly[i] != 0x0000) {
            // minpoly over GF(2) must have only 0/1 coefficients
            fprintf(stderr, "percyxx: minpoly coefficient not in GF(2) at %zu: %04x\n", i, poly[i]);
            return GF2X();
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// NTL GF2E <-> uint16 rep helpers. The rep is little-endian bit order: rep
// bit i = coefficient of x^i.
// ---------------------------------------------------------------------------

static inline uint16_t gf2e_to_u16(const GF2E& e) {
    const GF2X& r = rep(e);
    uint16_t v = 0;
    for (int i = 0; i < 16; ++i) if (IsOne(coeff(r, i))) v |= (uint16_t)(1u << i);
    return v;
}

static inline uint8_t gf2e_to_u8(const GF2E& e) {
    const GF2X& r = rep(e);
    uint8_t v = 0;
    for (int i = 0; i < 8; ++i) if (IsOne(coeff(r, i))) v |= (uint8_t)(1u << i);
    return v;
}

static inline GF2E u16_to_gf2e(uint16_t v) {
    GF2X px;
    for (int i = 0; i < 16; ++i) if (v & (1u << i)) SetCoeff(px, i);
    return to_GF2E(px);
}

static inline GF2E u8_to_gf2e(uint8_t v) {
    GF2X px;
    for (int i = 0; i < 8; ++i) if (v & (1u << i)) SetCoeff(px, i);
    return to_GF2E(px);
}

// ---------------------------------------------------------------------------
// Isomorphism between canonical gf16 and NTL's GF(2^16) (0x1002D).
//   fwd[canon16] = percy_rep16   (for mapping support/received IN)
//   inv[percy_rep16] = canon16   (for mapping candidates OUT)
// Built by aligning powers of a matched primitive element.
// ---------------------------------------------------------------------------

struct Iso16 {
    bool ok = false;
    string reason;
    uint16_t fwd[65536]; // canon -> percy rep
    uint16_t inv[65536]; // percy rep -> canon
};

static Iso16 build_iso16() {
    Iso16 iso;
    // 1. canonical primitive element
    uint16_t g_canon = gf16_find_primitive();
    if (g_canon == 0) { iso.reason = "no canonical primitive element"; return iso; }
    // 2. minpoly over GF(2)
    GF2X mp = canon_minpoly(g_canon);
    if (deg(mp) != 16) {
        iso.reason = "canonical minpoly not degree 16";
        return iso;
    }
    // 3. lift minpoly to GF2EX over the NTL field, then find g_percy
    // DETERMINISTICALLY: scan NTL elements by ascending 16-bit rep and pick
    // the first that is a root of the canonical minimal polynomial. (NTL's
    // FindRoots uses randomized equal-degree factoring, so roots[0] is a
    // different Frobenius conjugate each run — unsuitable for a stable
    // fingerprint.) Any root of a primitive polynomial is itself primitive.
    GF2EX P;
    for (long i = 0; i <= 16; ++i)
        if (IsOne(coeff(mp, i))) SetCoeff(P, i, to_GF2E(GF2(1)));
    GF2E g_percy;
    bool found = false;
    for (uint32_t cand = 0; cand < 65536 && !found; ++cand) {
        GF2E e = u16_to_gf2e((uint16_t)cand);
        // Horner evaluation of P at e
        GF2E acc; acc = to_GF2E(GF2(0));
        for (long i = deg(P); i >= 0; --i) {
            acc = acc * e + coeff(P, i);
        }
        if (IsZero(acc)) { g_percy = e; found = true; }
    }
    if (!found) { iso.reason = "no root of minpoly in NTL field"; return iso; }
    // sanity: g_percy must be primitive (order 65535). Minpoly is a primitive
    // polynomial, so any root has order 65535; confirm with the factor test.
    {
        static const unsigned long pr[4] = {3, 5, 17, 257};
        for (int q = 0; q < 4; ++q) {
            GF2E r = to_GF2E(GF2(1)), base = g_percy; unsigned long e = 65535 / pr[q];
            while (e) { if (e & 1) r = r * base; base = base * base; e >>= 1; }
            if (IsOne(r)) { iso.reason = "g_percy not primitive"; return iso; }
        }
    }
    // 4. build fwd/inv by walking aligned powers
    for (size_t i = 0; i < 65536; ++i) { iso.fwd[i] = 0; iso.inv[i] = 0; }
    uint16_t c = 0x0001;      // g_canon^0
    GF2E p = to_GF2E(GF2(1)); // g_percy^0
    iso.fwd[c] = gf2e_to_u16(p);
    iso.inv[gf2e_to_u16(p)] = c;
    for (unsigned long i = 1; i < 65535; ++i) {
        c = gf16_mul(c, g_canon);
        p = p * g_percy;
        iso.fwd[c] = gf2e_to_u16(p);
        iso.inv[gf2e_to_u16(p)] = c;
    }
    iso.fwd[0] = 0; iso.inv[0] = 0;
    // 5. verify round-trip + homomorphism on a large deterministic sample
    for (uint32_t v = 0; v < 65536; ++v) {
        if (iso.inv[iso.fwd[v]] != (uint16_t)v) {
            iso.reason = "fwd/inv round-trip failed"; return iso;
        }
    }
    if (iso.fwd[0] != 0 || iso.inv[0] != 0) { iso.reason = "0 map broken"; return iso; }
    if (iso.fwd[1] != 1 || iso.inv[1] != 1) { iso.reason = "1 map broken"; return iso; }
    // sample b set: small primes + structural values
    static const uint16_t bsamp[] = {
        0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
        0x00ff, 0xff00, 0x0100, 0x0200, 0x0020, 0x1000, 0x8000,
        0x1234, 0xdead, 0xbeef, 0xfeed, 65534, 65533
    };
    for (uint32_t a = 0; a < 65536; ++a) {
        uint16_t A = (uint16_t)a;
        GF2E PA = u16_to_gf2e(iso.fwd[A]);
        for (size_t s = 0; s < sizeof(bsamp)/sizeof(bsamp[0]); ++s) {
            uint16_t B = bsamp[s];
            GF2E PB = u16_to_gf2e(iso.fwd[B]);
            // add
            uint16_t add_canon = gf16_add(A, B);
            GF2E add_percy = PA + PB;
            if (iso.fwd[add_canon] != gf2e_to_u16(add_percy)) {
                iso.reason = "add homomorphism broken"; return iso;
            }
            // mul
            uint16_t mul_canon = gf16_mul(A, B);
            GF2E mul_percy = PA * PB;
            if (iso.fwd[mul_canon] != gf2e_to_u16(mul_percy)) {
                iso.reason = "mul homomorphism broken"; return iso;
            }
        }
    }
    // inverse on the sample (nonzero)
    for (size_t s = 0; s < sizeof(bsamp)/sizeof(bsamp[0]); ++s) {
        uint16_t A = bsamp[s];
        if (A == 0) continue;
        uint16_t ai = gf16_pow(A, 65534); // Fermat inverse
        GF2E PA = u16_to_gf2e(iso.fwd[A]);
        GF2E PI = inv(PA); // NTL inverse
        if (iso.fwd[ai] != gf2e_to_u16(PI)) {
            iso.reason = "inverse homomorphism broken"; return iso;
        }
    }
    iso.ok = true;
    return iso;
}

// Exhaustive verification of the gf8 identity map (NTL AES field == canonical).
static bool verify_gf8_identity() {
    for (uint32_t a = 0; a < 256; ++a) {
        uint8_t A = (uint8_t)a;
        GF2E PA = u8_to_gf2e(A);
        for (uint32_t b = 0; b < 256; ++b) {
            uint8_t B = (uint8_t)b;
            GF2E PB = u8_to_gf2e(B);
            if (gf2e_to_u8(PA + PB) != (uint8_t)(A ^ B)) return false;
            if (gf2e_to_u8(PA * PB) != gf8_mul(A, B)) return false;
        }
    }
    for (uint32_t a = 1; a < 256; ++a) {
        uint8_t A = (uint8_t)a;
        // Fermat inverse in AES gf8: a^254
        uint8_t ai = 0;
        { uint8_t r = 1, base = A; unsigned e = 254; while (e) { if (e&1) r = gf8_mul(r,base); base = gf8_mul(base,base); e>>=1; } ai = r; }
        GF2E PA = u8_to_gf2e(A);
        GF2E PI = inv(PA);
        if (gf2e_to_u8(PI) != ai) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// .gsf parsing (strict enough for our use; we only decode, ignore expected-*).
// ---------------------------------------------------------------------------

struct Fixture {
    string name;
    bool is_gf16;        // true gf16, false gf8
    string field_def;
    string domain;
    unsigned long n = 0, k = 0, target_radius = 0;
    vector< vector<uint8_t> > support;   // raw bytes (1 for gf8, 2 for gf16)
    vector< vector<uint8_t> > received;
};

static void emit_error(const string& msg) {
    printf("status=error:%s\n", msg.c_str());
    fflush(stdout);
}

static vector<uint8_t> parse_elem(const string& tok, int width) {
    int hexlen = width * 2;
    if ((int)tok.size() != hexlen) return vector<uint8_t>();
    vector<uint8_t> bytes;
    for (int i = 0; i < hexlen; i += 2) {
        char hi = tok[i], lo = tok[i+1];
        auto hv = [](char c)->int { if (c>='0'&&c<='9') return c-'0'; if (c>='a'&&c<='f') return c-'a'+10; return -1; };
        int h = hv(hi), l = hv(lo);
        if (h < 0 || l < 0) return vector<uint8_t>();
        bytes.push_back((uint8_t)((h << 4) | l));
    }
    return bytes;
}

static vector< vector<uint8_t> > parse_csv(const string& v, int width) {
    vector< vector<uint8_t> > out;
    size_t i = 0;
    while (i <= v.size()) {
        size_t j = v.find(',', i);
        if (j == string::npos) j = v.size();
        string tok = v.substr(i, j - i);
        if (tok.empty()) { out.clear(); return out; }
        out.push_back(parse_elem(tok, width));
        if (out.back().empty()) { out.clear(); return out; }
        i = j + 1;
    }
    return out;
}

static bool parse_fixture(const string& text, Fixture& fx, string& err) {
    // line-oriented
    vector<string> lines;
    {
        string cur;
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '\n') { lines.push_back(cur); cur.clear(); }
            else cur += text[i];
        }
        if (!cur.empty()) { err = "file must end with LF"; return false; }
    }
    if (lines.empty()) { err = "empty file"; return false; }
    if (lines[0] != "gs-engine-fixture-v1") { err = "bad version header"; return false; }

    string name, field, fielddef, domain, n, k, tr, support, received;
    bool have_n=false,have_k=false,have_tr=false,have_sup=false,have_rec=false,have_field=false,have_fd=false,have_dom=false,have_name=false;
    for (size_t li = 1; li < lines.size(); ++li) {
        const string& line = lines[li];
        if (line.empty()) { err = "blank line"; return false; }
        size_t eq = line.find('=');
        if (eq == string::npos) { err = "record without '='"; return false; }
        string key = line.substr(0, eq);
        string val = line.substr(eq + 1);
        if (key == "name") { name = val; have_name = true; }
        else if (key == "field") { field = val; have_field = true; }
        else if (key == "field-definition") { fielddef = val; have_fd = true; }
        else if (key == "domain") { domain = val; have_dom = true; }
        else if (key == "n") { n = val; have_n = true; }
        else if (key == "k") { k = val; have_k = true; }
        else if (key == "target-radius") { tr = val; have_tr = true; }
        else if (key == "support") { support = val; have_sup = true; }
        else if (key == "received") { received = val; have_rec = true; }
        else if (key == "multiplicity" || key == "y-degree" || key == "weighted-degree" ||
                 key == "expected-candidate" || key == "expected-codeword") {
            // ignored by the decoder
        } else { err = "unknown key " + key; return false; }
    }
    if (!have_name||!have_field||!have_fd||!have_dom||!have_n||!have_k||!have_tr||!have_sup||!have_rec) {
        err = "missing required key"; return false;
    }
    if (field != "gf8" && field != "gf16") { err = "unknown field"; return false; }
    fx.is_gf16 = (field == "gf16");
    int width = fx.is_gf16 ? 2 : 1;
    fx.name = name;
    fx.field_def = fielddef;
    fx.domain = domain;
    auto parse_ul = [](const string& s, unsigned long& out)->bool {
        if (s.empty()) return false;
        for (char c : s) if (c < '0' || c > '9') return false;
        out = strtoul(s.c_str(), nullptr, 10);
        return true;
    };
    if (!parse_ul(n, fx.n) || !parse_ul(k, fx.k) || !parse_ul(tr, fx.target_radius)) {
        err = "bad numeric field"; return false;
    }
    fx.support = parse_csv(support, width);
    fx.received = parse_csv(received, width);
    if (fx.support.empty() || fx.received.empty()) { err = "bad support/received"; return false; }
    if (fx.support.size() != fx.n) { err = "support length != n"; return false; }
    if (fx.received.size() != fx.n) { err = "received length != n"; return false; }
    if (fx.k == 0) { err = "k must be >= 1"; return false; }
    return true;
}

// ---------------------------------------------------------------------------
// Decode.
// ---------------------------------------------------------------------------

// Encode one element to canonical hex (2 digits gf8, 4 gf16), little-endian.
static string hex_elem_gf8(uint8_t b) {
    char buf[4];
    snprintf(buf, sizeof buf, "%02x", b);
    return buf;
}
static string hex_elem_gf16(uint16_t v) {
    // canon16 = (a1<<8)|a0 -> bytes [a0, a1] little-endian
    char buf[8];
    snprintf(buf, sizeof buf, "%02x%02x", v & 0xff, (v >> 8) & 0xff);
    return buf;
}

int main(int argc, char** argv) {
    // Fingerprint mode: dump the raw gf16 forward table (65536 uint16 LE) for
    // sha256 hashing by the build script. Not a decode path.
    if (argc >= 2 && string(argv[1]) == "--fingerprint") {
        GF2X P; SetCoeff(P,16); SetCoeff(P,5); SetCoeff(P,3); SetCoeff(P,2); SetCoeff(P,0);
        GF2E::init(P);
        Iso16 iso = build_iso16();
        if (!iso.ok) { fprintf(stderr, "percyxx: fingerprint: isomorphism failed: %s\n", iso.reason.c_str()); return 1; }
        for (size_t i = 0; i < 65536; ++i) {
            uint16_t v = iso.fwd[i];
            unsigned char b[2] = { (unsigned char)(v & 0xff), (unsigned char)((v >> 8) & 0xff) };
            if (fwrite(b, 1, 2, stdout) != 2) return 1;
        }
        fflush(stdout);
        return 0;
    }

    if (argc < 2) { emit_error("no fixture path"); return 0; }
    const char* path = argv[1];

    FILE* fp = fopen(path, "rb");
    if (!fp) { emit_error(string("cannot open ") + path); return 0; }
    string text;
    {
        char buf[8192]; size_t r;
        while ((r = fread(buf, 1, sizeof buf, fp)) > 0) text.append(buf, r);
    }
    fclose(fp);

    Fixture fx;
    string err;
    if (!parse_fixture(text, fx, err)) { emit_error(err); return 0; }

    // Initialize the NTL field and (for gf16) the isomorphism. All startup.
    if (fx.is_gf16) {
        GF2X P; SetCoeff(P,16); SetCoeff(P,5); SetCoeff(P,3); SetCoeff(P,2); SetCoeff(P,0);
        GF2E::init(P);
    } else {
        GF2X P; SetCoeff(P,8); SetCoeff(P,4); SetCoeff(P,3); SetCoeff(P,1); SetCoeff(P,0);
        GF2E::init(P);
        if (!verify_gf8_identity()) { emit_error("gf8 identity map verification failed"); return 0; }
    }

    // Build percy elements for support + received.
    vec_GF2E indices, shares;
    indices.SetLength((long)fx.n);
    shares.SetLength((long)fx.n);
    vector<unsigned short> goods;
    goods.reserve(fx.n);
    for (unsigned long i = 0; i < fx.n; ++i) goods.push_back((unsigned short)i);

    if (fx.is_gf16) {
        Iso16 iso = build_iso16();
        if (!iso.ok) {
            printf("status=unsupported:gf16 isomorphism failed: %s\n", iso.reason.c_str());
            fflush(stdout);
            return 0;
        }
        for (unsigned long i = 0; i < fx.n; ++i) {
            const vector<uint8_t>& sb = fx.support[i];
            const vector<uint8_t>& rb = fx.received[i];
            uint16_t sc = (uint16_t)((sb[1] << 8) | sb[0]);
            uint16_t rc = (uint16_t)((rb[1] << 8) | rb[0]);
            indices[(long)i] = u16_to_gf2e(iso.fwd[sc]);
            shares[(long)i]  = u16_to_gf2e(iso.fwd[rc]);
        }
        // Reuse the same iso for output: stash it in a static for the output phase.
        // (We rebuild lazily below; keep the tables by caching in statics.)
        static uint16_t g_inv[65536];
        memcpy(g_inv, iso.inv, sizeof g_inv);
        // declare a decode closure that uses g_inv
        auto decode16 = [&](void) -> int {
            unsigned long max_degree = fx.k - 1;
            unsigned long t = fx.n - fx.target_radius;
            RSDecoder_GF2E dec;
            vector< RecoveryPoly<GF2EX> > polys;
            if (max_degree == 0) {
                auto _t0 = std::chrono::steady_clock::now();
                map<uint16_t, unsigned long> counts;
                for (unsigned long i = 0; i < fx.n; ++i) {
                    const vector<uint8_t>& rb = fx.received[i];
                    uint16_t rc = (uint16_t)((rb[1] << 8) | rb[0]);
                    counts[rc]++;
                }
                vector<uint16_t> consts;
                for (auto& pr : counts) if (pr.second >= t) consts.push_back(pr.first);
                auto _t1 = std::chrono::steady_clock::now();
                printf("status=radius\n");
                printf("decode-ns=%lld\n", std::chrono::duration_cast<std::chrono::nanoseconds>(_t1 - _t0).count());
                for (uint16_t c : consts) printf("candidate=%s\n", hex_elem_gf16(c).c_str());
                fflush(stdout);
                return 0;
            }
            auto _t0 = std::chrono::steady_clock::now();
            polys = dec.findpolys_gs((unsigned int)max_degree, (unsigned int)t, goods, indices, shares);
            auto _t1 = std::chrono::steady_clock::now();
            long long _dns = std::chrono::duration_cast<std::chrono::nanoseconds>(_t1 - _t0).count();
            printf("status=radius\n");
            printf("decode-ns=%lld\n", _dns);
            for (auto& rp : polys) {
                const GF2EX& phi = rp.phi;
                long d = deg(phi);
                string line = "candidate=";
                if (d < 0) {
                    line += hex_elem_gf16(0);
                } else {
                    // constant term first, up to degree, no trailing zeros.
                    // (findpolys_gs returns polys with no trailing zero beyond deg.)
                    for (long i = 0; i <= d; ++i) {
                        GF2E c = coeff(phi, i);
                        uint16_t percy = gf2e_to_u16(c);
                        uint16_t canon = g_inv[percy];
                        if (i > 0) line += ',';
                        line += hex_elem_gf16(canon);
                    }
                }
                printf("%s\n", line.c_str());
            }
            fflush(stdout);
            return 0;
        };
        return decode16();
    }

    // gf8 path
    unsigned long max_degree = fx.k - 1;
    unsigned long t = fx.n - fx.target_radius;
    for (unsigned long i = 0; i < fx.n; ++i) {
        indices[(long)i] = u8_to_gf2e(fx.support[i][0]);
        shares[(long)i]  = u8_to_gf2e(fx.received[i][0]);
    }
    RSDecoder_GF2E dec;
    if (max_degree == 0) {
        auto _t0 = std::chrono::steady_clock::now();
        map<uint8_t, unsigned long> counts;
        for (unsigned long i = 0; i < fx.n; ++i) counts[fx.received[i][0]]++;
        vector<uint8_t> consts;
        for (auto& pr : counts) if (pr.second >= t) consts.push_back(pr.first);
        auto _t1 = std::chrono::steady_clock::now();
        printf("status=radius\n");
        printf("decode-ns=%lld\n", std::chrono::duration_cast<std::chrono::nanoseconds>(_t1 - _t0).count());
        for (uint8_t c : consts) printf("candidate=%s\n", hex_elem_gf8(c).c_str());
        fflush(stdout);
        return 0;
    }
    auto _t0 = std::chrono::steady_clock::now();
    vector< RecoveryPoly<GF2EX> > polys =
        dec.findpolys_gs((unsigned int)max_degree, (unsigned int)t, goods, indices, shares);
    auto _t1 = std::chrono::steady_clock::now();
    long long _dns = std::chrono::duration_cast<std::chrono::nanoseconds>(_t1 - _t0).count();
    printf("status=radius\n");
    printf("decode-ns=%lld\n", _dns);
    for (auto& rp : polys) {
        const GF2EX& phi = rp.phi;
        long d = deg(phi);
        string line = "candidate=";
        if (d < 0) {
            line += hex_elem_gf8(0);
        } else {
            for (long i = 0; i <= d; ++i) {
                GF2E c = coeff(phi, i);
                uint8_t canon = gf2e_to_u8(c);
                if (i > 0) line += ',';
                line += hex_elem_gf8(canon);
            }
        }
        printf("%s\n", line.c_str());
    }
    fflush(stdout);
    return 0;
}
