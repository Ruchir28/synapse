use std::sync::Arc;

use crate::NDArray;


pub trait TransformOps<T> {
    fn permute_axis(&self, axes: &[usize]) -> Self;
}

impl<T> TransformOps<T> for NDArray<T> {
    fn permute_axis(&self, axes: &[usize]) -> Self {
        if axes.len() != self.dims().len() {
            panic!("Axes length must match the number of dimensions in the NDArray");
        }

        let mut new_dims = Vec::with_capacity(axes.len());
        let mut new_strides = Vec::with_capacity(axes.len());

        for &axis_idx in axes {
            if axis_idx >= self.dims().len() {
                panic!("Axis index out of bounds");
            }
            new_dims.push(self.dims()[axis_idx]);
            new_strides.push(self.strides()[axis_idx]);
        }

        NDArray::from_parts(
            Arc::clone(&self.data()),
            new_dims,
            new_strides,
            self.offset()
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{NDArray, ops::TransformOps};

    #[test]
    fn test_permute_axis_2d() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let array = NDArray::new(data, vec![2, 3]);
        
        // Permute axes to get a 3x2 matrix: [[1, 4], [2, 5], [3, 6]]
        let permuted = array.permute_axis(&[1, 0]);
        
        // Check dimensions
        assert_eq!(permuted.dims(), &[3, 2]);
        
        // Access elements to verify correct permutation
        assert_eq!(permuted.get(&[0, 0]), Some(&1)); // [0,0] -> original [0,0]
        assert_eq!(permuted.get(&[0, 1]), Some(&4)); // [0,1] -> original [1,0]
        assert_eq!(permuted.get(&[1, 0]), Some(&2)); // [1,0] -> original [0,1]
        assert_eq!(permuted.get(&[1, 1]), Some(&5)); // [1,1] -> original [1,1]
        assert_eq!(permuted.get(&[2, 0]), Some(&3)); // [2,0] -> original [0,2]
        assert_eq!(permuted.get(&[2, 1]), Some(&6)); // [2,1] -> original [1,2]
    }

    #[test]
    fn test_permute_axis_3d() {
        // Create a 2x3x2 array
        let data = vec![
            1, 2,   // [0,0,0], [0,0,1]
            3, 4,   // [0,1,0], [0,1,1]
            5, 6,   // [0,2,0], [0,2,1]
            7, 8,   // [1,0,0], [1,0,1]
            9, 10,  // [1,1,0], [1,1,1]
            11, 12, // [1,2,0], [1,2,1]
        ];
        let array = NDArray::new(data, vec![2, 3, 2]);
        
        // Permute to [2, 0, 1] -> 2x2x3
        let permuted = array.permute_axis(&[2, 0, 1]);
        
        // Check dimensions
        assert_eq!(permuted.dims(), &[2, 2, 3]);
        
        // Check a few elements
        assert_eq!(permuted.get(&[0, 0, 0]), Some(&1)); // [0,0,0] -> original [0,0,0]
        assert_eq!(permuted.get(&[1, 0, 0]), Some(&2)); // [1,0,0] -> original [0,0,1]
        assert_eq!(permuted.get(&[0, 1, 2]), Some(&11)); // [0,1,2] -> original [1,2,0]
    }
}
