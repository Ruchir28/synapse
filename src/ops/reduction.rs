use crate::NDArray;
use std::{ops::Add, vec};
use num_traits::{Zero, ToPrimitive};


pub trait ReductionOps<T> {

    type MeanOutput;

    fn sum(&self) -> T;
    fn sum_axis(&self, axis: usize) -> NDArray<T>;
    fn mean(&self) -> Self::MeanOutput;
    fn mean_axis(&self, axis: usize) -> NDArray<Self::MeanOutput>;
}

impl<T> ReductionOps<T> for NDArray<T>
where
    T: Copy + Clone + Add<Output = T> + Default + ToPrimitive + Zero,
{

    type MeanOutput = f64;

    fn sum(&self) -> T {
        self.iter().fold(T::default(), |acc, &x| acc + x)
    }

    fn sum_axis(&self, axis: usize) -> NDArray<T> {
        let mut output_dims: Vec<usize> = self.dims().to_vec();
        if axis >= output_dims.len() {
            panic!("Axis out of bounds");
        }
        output_dims.remove(axis);

        let mut result = NDArray::new(
            vec![T::zero(); output_dims.iter().product()],
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

    fn mean(&self) -> Self::MeanOutput {
        let sum: f64 = self.iter().map(|&x| x.to_f64().expect("mean undefined for this dtype")).sum();
        let product = self.dims().iter().product::<usize>() as f64;
        return sum / product;
    }

    fn mean_axis(&self, axis: usize) -> NDArray<Self::MeanOutput> {

        let axis_sum = self.sum_axis(axis);

        let mean_axis_len = self.dims()[axis] as f64;

        let mean_data = axis_sum.iter().map(|x| x.to_f64()
        .expect("mean undefined for this dtype") / mean_axis_len).collect();

        let mut updated_dims = self.dims().to_vec();

        updated_dims.remove(axis);

        return NDArray::new(mean_data,updated_dims);
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

    #[test]
    fn test_mean() {
        let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert_eq!(array.mean(),21.0/6.0);
    }

    #[test]
    fn test_mean_axis_0() {
        let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let result = array.mean_axis(0);
        assert_eq!(result.dims(), &[3]);
        assert_eq!(result.data().as_ref(), &vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_mean_axis_1() {
        let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let result = array.mean_axis(1);
        assert_eq!(result.dims(), &[2]);
        assert_eq!(result.data().as_ref(), &vec![2.0, 5.0]);
    }

    #[test]
    #[should_panic]
    fn test_mean_axis_out_of_bounds() {
        let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        array.mean_axis(2);
    }

    #[test]
    fn test_mean_u8() {
        let array: NDArray<u8> = NDArray::new(vec![1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8], vec![2, 3]);
        assert_eq!(array.mean(), 21.0/6.0);
    }

    #[test]
    fn test_mean_axis_0_u8() {
        let array: NDArray<u8> = NDArray::new(vec![1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8], vec![2, 3]);
        let result = array.mean_axis(0);
        assert_eq!(result.dims(), &[3]);
        assert_eq!(result.data().as_ref(), &vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_mean_axis_1_u8() {
        let array: NDArray<u8> = NDArray::new(vec![1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8], vec![2, 3]);
        let result = array.mean_axis(1);
        assert_eq!(result.dims(), &[2]);
        assert_eq!(result.data().as_ref(), &vec![2.0, 5.0]);
    }
    
}
