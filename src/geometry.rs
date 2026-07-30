//! Checked arithmetic and allocation for decoder buffer geometry.

use alloc::vec::Vec;

use crate::ConfigError;

/// Multiply two geometry dimensions, reporting an address-space overflow.
pub fn checked_product(
    context: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, ConfigError> {
    left.checked_mul(right)
        .ok_or(ConfigError::GeometryOverflow { context })
}

/// Add two geometry dimensions, reporting an address-space overflow.
pub fn checked_sum(context: &'static str, left: usize, right: usize) -> Result<usize, ConfigError> {
    left.checked_add(right)
        .ok_or(ConfigError::GeometryOverflow { context })
}

/// Allocate `elements` default-initialized values without aborting on a
/// recoverable capacity or reservation failure.
pub fn try_zeroed<E: Clone + Default>(
    context: &'static str,
    elements: usize,
) -> Result<Vec<E>, ConfigError> {
    checked_product(context, elements, core::mem::size_of::<E>())?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| ConfigError::AllocationFailed {
            context,
            elements,
            element_size: core::mem::size_of::<E>(),
        })?;
    values.resize(elements, E::default());
    Ok(values)
}
