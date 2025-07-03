pub mod error;
pub mod ndarray;
pub mod ops;

pub use error::NDArrayError;
pub use ndarray::NDArray;
pub use ops::*;

#[cfg(test)]
mod integration_tests {
    use crate::ndarray::NDArray;
    use crate::ops::{ReductionOps, SliceOps};

    #[test]
    fn test_sum_on_slice() {
        let array = NDArray::new((0..16).collect::<Vec<i32>>(), vec![4, 4]);
        let sliced = array.slice(&[1..3, 1..3]); // A 2x2 view of the center

        // The elements in the slice are 5, 6, 9, 10
        let expected_sum = 5 + 6 + 9 + 10;

        assert_eq!(sliced.sum(), expected_sum);
    }

    #[test]
    fn test_sum_axis_rows_on_slice() {
        let array = NDArray::new((0..12).collect::<Vec<i32>>(), vec![3, 4]);
        // Slice to a 2x3 view:
        // Original:
        // [[0, 1, 2, 3],
        //  [4, 5, 6, 7],
        //  [8, 9, 10, 11]]
        // Sliced view:
        // [[1, 2, 3],
        //  [5, 6, 7]]
        let sliced = array.slice(&[0..2, 1..4]);
        let result = sliced.sum_axis(1); // Sum rows

        // Expected sums: [1+2+3, 5+6+7] = [6, 18]
        let expected_array = NDArray::new(vec![6, 18], vec![2]);

        assert_eq!(result.dims(), expected_array.dims());
        assert_eq!(result.iter().collect::<Vec<_>>(), expected_array.iter().collect::<Vec<_>>());
    }

    #[test]
    fn test_sum_axis_cols_on_slice() {
        let array = NDArray::new((0..12).collect::<Vec<i32>>(), vec![3, 4]);
        // Sliced view:
        // [[1, 2, 3],
        //  [5, 6, 7]]
        let sliced = array.slice(&[0..2, 1..4]);
        let result = sliced.sum_axis(0); // Sum columns

        // Expected sums: [1+5, 2+6, 3+7] = [6, 8, 10]
        let expected_array = NDArray::new(vec![6, 8, 10], vec![3]);
        
        assert_eq!(result.dims(), expected_array.dims());
        assert_eq!(result.iter().collect::<Vec<_>>(), expected_array.iter().collect::<Vec<_>>());
    }
}
