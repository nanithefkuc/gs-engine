//! Optional data-parallel decode support.
//!
//! The `parallel` feature layers Rayon over the naturally independent parts of
//! a decode: separate received words in a batch, independent affine root
//! families, and large candidate-scoring blocks. Every parallel path merges its
//! results in a fixed order, so output is byte-identical to the single-thread
//! path regardless of the thread schedule. Without the feature the marker below
//! is vacuous and the crate keeps its `no_std` core and single-thread behavior.

/// Field marker required by the optional parallel decode paths.
///
/// With the `parallel` feature this is `Send + Sync`; without it the bound is
/// vacuous. It is sealed by a blanket implementation and never implemented by
/// callers.
#[cfg(feature = "parallel")]
pub trait ParallelField: Send + Sync {}
#[cfg(feature = "parallel")]
impl<T: Send + Sync> ParallelField for T {}

/// Field marker required by the optional parallel decode paths.
///
/// With the `parallel` feature this is `Send + Sync`; without it the bound is
/// vacuous. It is sealed by a blanket implementation and never implemented by
/// callers.
#[cfg(not(feature = "parallel"))]
pub trait ParallelField {}
#[cfg(not(feature = "parallel"))]
impl<T> ParallelField for T {}

/// Extension marker: the field's element type is also thread-shareable.
///
/// Used by the parallel scoring path, which broadcasts `&[F::Elem]` to worker
/// closures. With `parallel` off the bound is vacuous.
#[cfg(feature = "parallel")]
pub trait ParallelElem: Send + Sync {}
#[cfg(feature = "parallel")]
impl<T: Send + Sync> ParallelElem for T {}

/// Extension marker: the field's element type is also thread-shareable.
///
/// Used by the parallel scoring path, which broadcasts `&[F::Elem]` to worker
/// closures. With `parallel` off the bound is vacuous.
#[cfg(not(feature = "parallel"))]
pub trait ParallelElem {}
#[cfg(not(feature = "parallel"))]
impl<T> ParallelElem for T {}

/// Minimum number of words in a batch before [`crate::GsPlan::decode_batch_into`]
/// spreads the words across the Rayon pool. Smaller batches decode in order on
/// the calling thread. See `BENCHMARKS.md`.
#[cfg(feature = "parallel")]
pub const PARALLEL_BATCH_CROSSOVER: usize = 4;

/// Minimum aggregate affine-family completion count before root materialization
/// runs across the Rayon pool. Below it, families are completed in order on the
/// calling thread. See `BENCHMARKS.md`.
#[cfg(feature = "parallel")]
pub const PARALLEL_ROOT_FAMILY_CROSSOVER: usize = 4_096;

/// Minimum candidate-by-point work (`candidates * points`) before Horner
/// candidate scoring distributes candidates across the Rayon pool. See
/// `BENCHMARKS.md`.
#[cfg(feature = "parallel")]
pub const PARALLEL_SCORING_CROSSOVER: usize = 16_384;
