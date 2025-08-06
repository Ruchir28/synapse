use std::sync::Arc;

use crate::{NDArray, NDArrayError};

pub trait ShapeOps<T> {
    fn reshape(&self, new_shape: Vec<usize>) -> Result<NDArray<T>, NDArrayError>;
}

impl<T> ShapeOps<T> for NDArray<T>
where
    T: Copy + Clone + Default,
{
    fn reshape(&self, new_shape: Vec<usize>) -> Result<NDArray<T>, NDArrayError> {
        // 1) Validate logical element count (dims product), not raw buffer length.
        let current_elems: usize = self.dims().iter().product();
        let new_elems: usize = new_shape.iter().product();

        if new_elems != current_elems {
            return Err(NDArrayError::DimensionMismatch {
                expected: new_shape.clone(),
                found: self.dims().to_vec(),
            });
        }

        // 2) Reshape as a view is only valid for contiguous arrays.
        if !self.is_contiguous() {
            return Err(NDArrayError::Generic(
                "Cannot reshape non-contiguous view without copy".to_string(),
            ));
        }

        // 3) Compute row-major (C-order) strides for the new shape.
        let mut new_strides = vec![1isize; new_shape.len()];
        if new_strides.len() > 1 {
            for i in (0..new_strides.len() - 1).rev() {
                new_strides[i] = new_strides[i + 1] * new_shape[i + 1] as isize;
            }
        }

        // 4) Preserve the current offset (should be 0 for contiguous arrays).
        Ok(NDArray::from_parts(
            Arc::clone(&self.data()),
            new_shape,
            new_strides,
            0
        ))
    }
}


#[cfg(test)]
mod tests {
    use crate::ops::ShapeOps;
    use super::*;

    #[test]
    fn test_reshape() {
        // Create a 2x3 array
        let original_data = vec![1, 2, 3, 4, 5, 6];
        let a = NDArray::new(original_data.clone(), vec![2, 3]);
        
        // Verify initial state
        assert!(a.is_contiguous());
        assert_eq!(a.dims(), &[2, 3]);
        assert_eq!(a.data().as_ref(), &original_data);

        // Reshape to 3x2
        let new_shape = vec![3, 2];
        let b = a.reshape(new_shape.clone()).unwrap();

        assert_eq!(b.dims(), &new_shape);
        assert!(b.is_contiguous());
        
        // Verify data remains the same after reshape
        assert_eq!(b.data().as_ref(), &original_data);

        // Verify invalid reshape fails
        let invalid_shape = vec![2, 4]; // Different number of elements
        assert!(a.reshape(invalid_shape).is_err());
    }

}