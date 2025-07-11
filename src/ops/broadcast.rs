use crate::NDArrayError;

#[derive(Debug, Clone)]
pub struct BroadcastInfo {
    pub result_shape: Vec<usize>,
    pub a_strides: Vec<isize>,
    pub b_strides: Vec<isize>,
}

pub fn broadcast_shapes(
    a_dims: &[usize],
    b_dims: &[usize],
    a_strides: &[isize],
    b_strides: &[isize],
) -> Result<BroadcastInfo, NDArrayError> {
  

    
    let max_len = a_dims.len().max(b_dims.len());

    let mut result_shape = vec![0; max_len];
    let mut a_strides_new = vec![0; max_len];
    let mut b_strides_new = vec![0; max_len];

    for i in 1..=max_len {
        
        let a_dim_i = a_dims.len().checked_sub(i).map(|idx| a_dims[idx]).unwrap_or(1);
        
        let b_dim_i = b_dims.len().checked_sub(i).map(|idx| b_dims[idx]).unwrap_or(1);

        let a_strides_i = a_strides.len().checked_sub(i).map(|idx| a_strides[idx]).unwrap_or(0);

        let b_strides_i = b_strides.len().checked_sub(i).map(|idx| b_strides[idx]).unwrap_or(0);

        let result_dim_i = a_dim_i.max(b_dim_i);
        
        result_shape[max_len - i] = result_dim_i;

        if a_dim_i != result_dim_i && a_dim_i != 1 {
            return Err(NDArrayError::BroadcastError(
                a_dims.to_vec(), b_dims.to_vec())
            );
        }

        if b_dim_i != result_dim_i && b_dim_i != 1 {
            return Err(NDArrayError::BroadcastError(
                a_dims.to_vec(), b_dims.to_vec())
            );
        }

        a_strides_new[max_len - i] = if a_dim_i == result_dim_i  {a_strides_i} else { 0 };
        b_strides_new[max_len - i] = if b_dim_i == result_dim_i { b_strides_i } else { 0 };

    }

    Ok(BroadcastInfo{
        result_shape,
        a_strides: a_strides_new,
        b_strides: b_strides_new
    })

}


#[cfg(test)]
mod tests {
    use std::vec;

    use crate::{ broadcast::broadcast_shapes, NDArray};


    #[test]
    fn test_broadcast_vector_to_matrix() {
        // Broadcast a vector [2] to a matrix [3, 2]
        let a = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![3, 2]); // Strides [2, 1]
        let b = NDArray::new(vec![10, 20], vec![2]);             // Strides [1]

        let info = broadcast_shapes(a.dims(), b.dims(), a.strides(), b.strides()).unwrap();

        assert_eq!(info.result_shape, vec![3, 2]);
        assert_eq!(info.a_strides, vec![2, 1]);
        assert_eq!(info.b_strides, vec![0, 1]);
    }

    #[test]
    fn test_broadcast_scalar_to_matrix() {
        // Broadcast a scalar (represented as a 1-element array) to a matrix
        let a = NDArray::new(vec![1], vec![1]);                  // Strides [1]
        let b = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]); // Strides [2, 1]

        let info = broadcast_shapes(a.dims(), b.dims(), a.strides(), b.strides()).unwrap();

        assert_eq!(info.result_shape, vec![2, 2]);
        assert_eq!(info.a_strides, vec![0, 0]);
        assert_eq!(info.b_strides, vec![2, 1]);
    }

    #[test]
    fn test_broadcast_3d_and_2d() {
        // Broadcast a 2D array [3, 4] to a 3D array [2, 3, 4]
        let a = NDArray::new(vec![0; 24], vec![2, 3, 4]); // Strides [12, 4, 1]
        let b = NDArray::new(vec![0; 12], vec![3, 4]);    // Strides [4, 1]

        let info = broadcast_shapes(a.dims(), b.dims(), a.strides(), b.strides()).unwrap();

        assert_eq!(info.result_shape, vec![2, 3, 4]);
        assert_eq!(info.a_strides, vec![12, 4, 1]);
        assert_eq!(info.b_strides, vec![0, 4, 1]);
    }

    #[test]
    fn test_broadcast_incompatible_shapes() {
        // Incompatible shapes: [3, 2] and [3]
        let a = NDArray::new(vec![0; 6], vec![3, 2]);
        let b = NDArray::new(vec![0; 3], vec![3]);

        let result = broadcast_shapes(a.dims(), b.dims(), a.strides(), b.strides());

        assert!(result.is_err());
        match result.err().unwrap() {
            crate::NDArrayError::BroadcastError(shape1, shape2) => {
                assert_eq!(shape1, vec![3, 2]);
                assert_eq!(shape2, vec![3]);
            }
            _ => panic!("Expected BroadcastError"),
        }
    }

    #[test]
    fn test_broadcast_both_ways() {
        // Broadcast [4, 1] and [1, 3] to [4, 3]
        let a = NDArray::new(vec![0; 4], vec![4, 1]); // Strides [1, 1]
        let b = NDArray::new(vec![0; 3], vec![1, 3]); // Strides [3, 1]

        let info = broadcast_shapes(a.dims(), b.dims(), a.strides(), b.strides()).unwrap();

        assert_eq!(info.result_shape, vec![4, 3]);
        assert_eq!(info.a_strides, vec![1, 0]);
        assert_eq!(info.b_strides, vec![0, 1]);
    }
}
