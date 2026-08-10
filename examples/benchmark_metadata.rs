use fgf::field::Field;
use fgf::kernel::{backend, backend_for};
use fgf::{Gf8, Gf16};

fn main() {
    println!("benchmark-metadata-version=1");
    println!("selected-backend={}", backend().name());
    println!(
        "field=gf8;bits={};bytes={};backend={};definition=gf2[x]/(x^8+x^4+x^3+x+1);le-polynomial-basis",
        Gf8::BITS,
        Gf8::BYTES,
        backend_for::<Gf8>().name()
    );
    println!(
        "field=gf16;bits={};bytes={};backend={};definition=gf8[u]/(u^2+u+0x20);gf8=aes-0x11b;le-components",
        Gf16::BITS,
        Gf16::BYTES,
        backend_for::<Gf16>().name()
    );
}
