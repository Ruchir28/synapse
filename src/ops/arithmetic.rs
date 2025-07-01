use crate::NDArray;
use std::ops::{Add, Div, Mul, Sub};

impl<'a, 'b, T> Add<&'b NDArray<T>> for &'a NDArray<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = NDArray<T>;

    fn add(self, rhs: &'b NDArray<T>) -> Self::Output {
        self.try_add(rhs).unwrap()
    }
}

impl<'a, 'b, T> Sub<&'b NDArray<T>> for &'a NDArray<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = NDArray<T>;

    fn sub(self, rhs: &'b NDArray<T>) -> Self::Output {
        self.try_sub(rhs).unwrap()
    }
}


impl<'a, 'b, T> Mul<&'b NDArray<T>> for &'a NDArray<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = NDArray<T>;

    fn mul(self, rhs: &'b NDArray<T>) -> Self::Output {
        self.try_mul(rhs).unwrap()
    }
}


impl<'a, 'b, T> Div<&'b NDArray<T>> for &'a NDArray<T>
where
    T: Div<Output = T> + Copy,
{
    type Output = NDArray<T>;

    fn div(self, rhs: &'b NDArray<T>) -> Self::Output {
        self.try_div(rhs).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{ops::TransformOps, NDArray, NDArrayError};

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

        #[test]
    fn test_sub_arrays() {
        let array1 = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
        let array2 = NDArray::new(vec![5, 6, 7, 8], vec![2, 2]);

        let result: NDArray<i32> = &array1 - &array2;

        let expected_data = vec![-4, -4, -4, -4];
        assert_eq!(result.data().as_ref(), &expected_data);
        assert_eq!(result.dims(), &[2, 2]);
    }

    #[test]
    fn test_sub_with_transpose() {
        let array1 = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
        let array2 = NDArray::new(vec![5, 6, 7, 8], vec![2, 2]);

        let transposed_array2 = array2.permute_axis(&[1, 0]);

        let result = &array1 - &transposed_array2;

        let expected_data = vec![-4, -5, -3, -4];
        assert_eq!(result.data().as_ref(), &expected_data);
        assert_eq!(result.dims(), &[2, 2]);
    }

    #[test]
    fn test_mul() {
        let array1 = NDArray::new(vec![1,2,3,4],vec![2,2]);
        let array2 = NDArray::new(vec![5,6,7,8],vec![2,2]);

        let result = &array1 * &array2;

        let expected_data = vec![5,12,21,32];

        assert_eq!(result.data().as_ref(),&expected_data);
        assert_eq!(result.dims(),&[2,2]);
    }


    #[test]
    fn test_mul_with_transpose() {
        let array1 = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
        let array2 = NDArray::new(vec![5, 6, 7, 8], vec![2, 2]);

        let transposed_array2 = array2.permute_axis(&[1, 0]);

        let result = &array1 * &transposed_array2;

        let expected_data = vec![5,14, 18, 32];
        assert_eq!(result.data().as_ref(), &expected_data);
        assert_eq!(result.dims(), &[2, 2]);
    }

    #[test]
    fn test_div_arrays() {
        let array1 = NDArray::new(vec![10, 20, 30, 40], vec![2, 2]);
        let array2 = NDArray::new(vec![2, 4, 6, 8], vec![2, 2]);

        let result = &array1 / &array2;

        let expected_data = vec![5, 5, 5, 5];
        assert_eq!(result.data().as_ref(), &expected_data);
        assert_eq!(result.dims(), &[2, 2]);
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let array1 = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
        let array2 = NDArray::new(vec![5, 6, 7, 8], vec![4, 1]);
        let result = array1.try_add(&array2);
        assert_eq!(
            result.unwrap_err(),
            NDArrayError::DimensionMismatch {
                expected: vec![2, 2],
                found: vec![4, 1],
            }
        );
    }

    #[test]
    fn test_sub_dimension_mismatch() {
        let array1 = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
        let array2 = NDArray::new(vec![5, 6, 7, 8], vec![4, 1]);
        let result = array1.try_sub(&array2);
        assert_eq!(
            result.unwrap_err(),
            NDArrayError::DimensionMismatch {
                expected: vec![2, 2],
                found: vec![4, 1],
            }
        );
    }

    #[test]
    fn test_mul_dimension_mismatch() {
        let array1 = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
        let array2 = NDArray::new(vec![5, 6, 7, 8], vec![4, 1]);
        let result = array1.try_mul(&array2);
        assert_eq!(
            result.unwrap_err(),
            NDArrayError::DimensionMismatch {
                expected: vec![2, 2],
                found: vec![4, 1],
            }
        );
    }

    #[test]
    fn test_div_dimension_mismatch() {
        let array1 = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
        let array2 = NDArray::new(vec![5, 6, 7, 8], vec![4, 1]);
        let result = array1.try_div(&array2);
        assert_eq!(
            result.unwrap_err(),
            NDArrayError::DimensionMismatch {
                expected: vec![2, 2],
                found: vec![4, 1],
            }
        );
    }
}
