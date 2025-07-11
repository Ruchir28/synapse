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
    fn test_broadcast_on_two_slices() {
        // A: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        let matrix_a = NDArray::new((0..12).collect::<Vec<i32>>(), vec![4, 3]);
        // B: [[100, 200, 300]]
        let matrix_b = NDArray::new(vec![100, 200, 300], vec![1, 3]);

        // slice_a is a column vector view: [[1], [4], [7], [10]] (shape [4, 1])
        let slice_a = matrix_a.slice(&[0..4, 1..2]);
        // slice_b is a row vector view: [[100, 200, 300]] (shape [1, 3])
        let slice_b = matrix_b.slice(&[0..1, 0..3]);

        // Broadcast add [4, 1] + [1, 3] -> [4, 3]
        let result = &slice_a + &slice_b;

        let expected_data = vec![
            101, 201, 301,
            104, 204, 304,
            107, 207, 307,
            110, 210, 310,
        ];
        let expected_array = NDArray::new(expected_data, vec![4, 3]);

        assert_eq!(result.dims(), expected_array.dims());
        assert_eq!(result.iter().collect::<Vec<_>>(), expected_array.iter().collect::<Vec<_>>());
    }

    #[test]
    fn test_broadcast_on_slice() {
        // Create a 3x3 matrix
        let matrix = NDArray::new((0..9).collect::<Vec<i32>>(), vec![3, 3]);
        // Slice to get the inner 2x2 matrix: [[4, 5], [7, 8]]
        let sliced_matrix = matrix.slice(&[1..3, 1..3]);

        // Create a vector to broadcast
        let vector: NDArray<i32> = NDArray::new(vec![10, 20], vec![2]);

        // Add the vector to each row of the sliced matrix
        let result = &sliced_matrix + &vector;

        // Expected result: [[4+10, 5+20], [7+10, 8+20]] = [[14, 25], [17, 28]]
        let expected_data = vec![14, 25, 17, 28];
        let expected_array = NDArray::new(expected_data, vec![2, 2]);

        assert_eq!(result.dims(), expected_array.dims());
        assert_eq!(result.iter().collect::<Vec<_>>(), expected_array.iter().collect::<Vec<_>>());
    }

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
