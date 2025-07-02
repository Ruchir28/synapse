use crate::NDArray;
use std::ops::Add;

pub trait ReductionOps<T> {
    fn sum(&self) -> T;
    fn sum_axis(&self, axis: usize) -> NDArray<T>;
}

impl<T> ReductionOps<T> for NDArray<T>
where
    T: Copy + Clone + Add<Output = T> + Default,
{
    fn sum(&self) -> T {
        self.iter().fold(T::default(), |acc, &x| acc + x)
    }

    fn sum_axis(&self, axis: usize) -> NDArray<T> {
        let mut output_dims = self.dims().to_vec();
        if axis >= output_dims.len() {
            panic!("Axis out of bounds");
        }
        output_dims.remove(axis);

        let mut result = NDArray::new(
            vec![T::default(); output_dims.iter().product()],
            output_dims,
        );

        for (mut input_index, value) in self.indexed_iter() {
            input_index.remove(axis);
            let output_index = &input_index;

            if let Some(val_to_update) = result.get_mut(output_index) {
                *val_to_update = *val_to_update + *value;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NDArray;

    #[test]
    fn test_sum() {
        let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert_eq!(array.sum(), 21);
    }

    #[test]
    fn test_sum_axis_0() {
        let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let result = array.sum_axis(0);
        assert_eq!(result.dims(), &[3]);
        assert_eq!(result.data().as_ref(), &vec![5, 7, 9]);
    }

    #[test]
    fn test_sum_axis_1() {
        let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let result = array.sum_axis(1);
        assert_eq!(result.dims(), &[2]);
        assert_eq!(result.data().as_ref(), &vec![6, 15]);
    }

    #[test]
    #[should_panic]
    fn test_sum_axis_out_of_bounds() {
        let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        array.sum_axis(2);
    }
}
