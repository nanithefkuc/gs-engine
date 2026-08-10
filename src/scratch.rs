use alloc::vec::Vec;

use butterfly_fft::basis::conversion_scratch_elements;
use butterfly_fft::core::kernel::ButterflyKernels;

use crate::geometry::checked_product;
use crate::{
    AlekhnovichLimits, AlekhnovichScratch, BivariatePolynomial, ConfigError, GsParameters,
    KoetterScratch, Polynomial,
};

/// Caller-owned reusable workspaces used by end-to-end decoding.
pub struct DecodeScratch<F: ButterflyKernels> {
    pub(crate) interpolation: KoetterScratch<F>,
    pub(crate) interpolation_output: BivariatePolynomial<F>,
    pub(crate) cached_received: Vec<F::Elem>,
    pub(crate) cached_interpolation_parameters: Option<GsParameters>,
    pub(crate) roots: AlekhnovichScratch<F>,
    pub(crate) cached_root_input: Option<BivariatePolynomial<F>>,
    pub(crate) cached_root_geometry: Option<(usize, AlekhnovichLimits)>,
    pub(crate) root_candidates: Vec<Polynomial<F>>,
    pub(crate) packed_evaluations: Vec<u8>,
    pub(crate) conversion: Vec<u8>,
    pub(crate) distances: Vec<usize>,
}

impl<F: ButterflyKernels> DecodeScratch<F> {
    /// Construct empty reusable decode scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            interpolation: KoetterScratch::new(),
            interpolation_output: BivariatePolynomial::zero(),
            cached_received: Vec::new(),
            cached_interpolation_parameters: None,
            roots: AlekhnovichScratch::new(),
            cached_root_input: None,
            cached_root_geometry: None,
            root_candidates: Vec::new(),
            packed_evaluations: Vec::new(),
            conversion: Vec::new(),
            distances: Vec::new(),
        }
    }

    /// Retained byte capacity for batched candidate evaluations.
    #[must_use]
    pub fn evaluation_capacity_bytes(&self) -> usize {
        self.packed_evaluations.capacity()
    }

    /// Retained byte capacity for monomial-to-novel conversion scratch.
    #[must_use]
    pub fn conversion_capacity_bytes(&self) -> usize {
        self.conversion.capacity()
    }

    pub(crate) fn reserve_evaluation(
        &mut self,
        point_count: usize,
        candidate_count: usize,
    ) -> Result<(), ConfigError> {
        let row_bytes = checked_product(
            "planned candidate evaluation row bytes",
            candidate_count,
            F::BYTES,
        )?;
        let evaluation_bytes =
            checked_product("planned candidate evaluation bytes", point_count, row_bytes)?;
        let conversion_bytes = checked_product(
            "planned candidate conversion bytes",
            conversion_scratch_elements(point_count),
            row_bytes,
        )?;
        reserve_zeroed(
            &mut self.packed_evaluations,
            evaluation_bytes,
            "planned candidate evaluations",
        )?;
        reserve_zeroed(
            &mut self.conversion,
            conversion_bytes,
            "planned candidate conversion scratch",
        )?;
        reserve_zeroed(
            &mut self.distances,
            candidate_count,
            "planned candidate distances",
        )
    }
}

fn reserve_zeroed<T: Default + Clone>(
    values: &mut Vec<T>,
    required: usize,
    context: &'static str,
) -> Result<(), ConfigError> {
    if required > values.len() {
        values
            .try_reserve_exact(required - values.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context,
                elements: required,
                element_size: core::mem::size_of::<T>(),
            })?;
        values.resize(required, T::default());
    }
    Ok(())
}
impl<F: ButterflyKernels> Default for DecodeScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}
