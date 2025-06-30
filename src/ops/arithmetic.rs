use crate::NDArray;
use std::ops::Add;

impl<'a, 'b, T> Add<&'b NDArray<T>> for &'a NDArray<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = NDArray<T>;

    fn add(self, rhs: &'b NDArray<T>) -> Self::Output {
        if self.dims() != rhs.dims() {
            panic!("Dimension mismatch for addition");
        }

        let result_data: Vec<T> = self.iter().zip(rhs.iter()).map(|(a, b)| *a + *b).collect();

        NDArray::new(result_data, self.dims().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use crate::{ops::TransformOps, NDArray};

    #[test]
    fn test_add_arrays() {
        let array1 = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
        let array2 = NDArray::new(vec![5, 6, 7, 8], vec![2, 2]);

        let result = &array1 + &array2;

        let expected_data = vec![6, 8, 10, 12];
        assert_eq!(result.data().as_ref(), &expected_data);
        assert_eq!(result.dims(), &[2, 2]);
    }

    #[test]
    fn test_add_with_transpose() {
        let array1 = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
        let array2 = NDArray::new(vec![5, 6, 7, 8], vec![2, 2]);

        let transposed_array2 = array2.permute_axis(&[1, 0]);

        let result = &array1 + &transposed_array2;

        let expected_data = vec![6, 9, 9, 12];
        assert_eq!(result.data().as_ref(), &expected_data);
        assert_eq!(result.dims(), &[2, 2]);
    }
}
