//! Polynomial roots used by Guruswami–Sudan root lifting.

mod alekhnovich;
mod field_roots;
mod roth_ruckenstein;

use alloc::vec::Vec;
use core::fmt;

use crate::PolynomialError;

pub(crate) use alekhnovich::alekhnovich_roots_into;
pub use alekhnovich::{
    AffineRootFamily, AlekhnovichLimits, AlekhnovichScratch, DEFAULT_ROTH_RUCKENSTEIN_CROSSOVER,
    alekhnovich_roots,
};
pub use field_roots::base_field_roots;
pub use roth_ruckenstein::{RothRuckensteinLimits, roth_ruckenstein_roots};

/// The roots of a univariate polynomial over its coefficient field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BaseFieldRoots<E> {
    /// The zero polynomial vanishes at every field element.
    All,
    /// A sorted, deduplicated finite root list.
    Finite(Vec<E>),
}

impl<E> BaseFieldRoots<E> {
    /// Borrow the finite root list, or return `None` for the zero polynomial.
    #[must_use]
    pub fn as_slice(&self) -> Option<&[E]> {
        match self {
            Self::All => None,
            Self::Finite(roots) => Some(roots),
        }
    }

    /// Consume the result, returning `None` when every field element is a root.
    #[must_use]
    pub fn into_finite(self) -> Option<Vec<E>> {
        match self {
            Self::All => None,
            Self::Finite(roots) => Some(roots),
        }
    }
}

/// Failure while isolating roots over the coefficient field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootError {
    /// Supporting polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Accelerated polynomial multiplication failed.
    Product(crate::ProductError),
    /// The field is not represented as a supported binary extension field.
    UnsupportedField {
        /// Number of elements in the field.
        field_order: u128,
        /// Bytes in the stable element representation.
        element_bytes: usize,
    },
    /// The zero bivariate polynomial has every bounded polynomial as a root.
    ZeroBivariatePolynomial,
    /// A caller-provided extraction resource limit was reached.
    ResourceLimitExceeded {
        /// Name of the bounded resource.
        resource: &'static str,
        /// Amount required to continue extraction.
        required: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// A factor known to split into distinct linear factors could not be split.
    FactorizationInvariant {
        /// Static explanation of the violated invariant.
        reason: &'static str,
    },
}

impl From<PolynomialError> for RootError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<crate::ProductError> for RootError {
    fn from(error: crate::ProductError) -> Self {
        Self::Product(error)
    }
}

impl From<crate::ConfigError> for RootError {
    fn from(error: crate::ConfigError) -> Self {
        Self::Polynomial(PolynomialError::Config(error))
    }
}

impl fmt::Display for RootError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Product(error) => error.fmt(formatter),
            Self::UnsupportedField {
                field_order,
                element_bytes,
            } => write!(
                formatter,
                "field order {field_order} with {element_bytes}-byte elements is not a supported binary field representation"
            ),
            Self::ZeroBivariatePolynomial => {
                formatter.write_str("the zero bivariate polynomial has every polynomial as a root")
            }
            Self::ResourceLimitExceeded {
                resource,
                required,
                limit,
            } => write!(
                formatter,
                "{resource} requires {required}, exceeding the root-extraction limit {limit}"
            ),
            Self::FactorizationInvariant { reason } => {
                write!(
                    formatter,
                    "polynomial root-extraction invariant failed: {reason}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RootError {}
