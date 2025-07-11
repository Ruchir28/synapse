use std::error::Error;
use std::fmt;

/// Error type for NDArray operations
#[derive(Debug, PartialEq)]
pub enum NDArrayError {
    /// Occurs when dimensions of arrays are incompatible for an operation
    DimensionMismatch { expected: Vec<usize>, found: Vec<usize> },
    /// Occurs when two shapes cannot be broadcast together.
    BroadcastError(Vec<usize>, Vec<usize>),
    /// Occurs when an index is out of bounds
    IndexOutOfBounds,
    /// Occurs when a value can't be converted to the desired type
    TypeConversionError,
    /// Generic error with a custom message
    Generic(String),
}

impl fmt::Display for NDArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NDArrayError::DimensionMismatch { expected, found } => {
                write!(f, "Dimension mismatch: expected {:?}, found {:?}", expected, found)
            }
            NDArrayError::BroadcastError(shape1, shape2) => {
                write!(f, "Could not broadcast shapes {:?} and {:?}", shape1, shape2)
            }
            NDArrayError::IndexOutOfBounds => write!(f, "Index out of bounds"),
            NDArrayError::TypeConversionError => write!(f, "Type conversion error"),
            NDArrayError::Generic(msg) => write!(f, "{}", msg),
        }
    }
}

impl Error for NDArrayError {}
