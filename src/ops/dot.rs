use std::iter::Sum;

use crate::{NDArray, TransformOps};

pub trait DotOps<T> {
    fn dot_2d(&self,other: &NDArray<T>) -> NDArray<T>;
}


impl<T> DotOps<T> for NDArray<T>
where
    T: Copy + Clone + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Default + Sum<T>,
{
    fn dot_2d(&self,other: &NDArray<T>) -> NDArray<T> {

        let m = self.dims()[0];
        let k1 = self.dims()[1];

        let k2 = other.dims()[0];
        let n = other.dims()[1];


        if k1 != k2 {
            panic!("Dimensions for matrix multiplication are not compatible");
        }

        let a_contig = self.to_contiguous();

        // transposing the other to make things faster 
        let b_transposed = other.permute_axis(&[1,0]).to_contiguous();

        let a_data = a_contig.data();

        let b_data = b_transposed.data();

        
        let result_data = vec![T::default(); m * n];
        let mut result = NDArray::new(result_data, vec![m, n]);

        for row_a_index in 0..m {
            for row_b_index in 0..n {

                let row_a_start = row_a_index * k1;
                let row_b_start = row_b_index * k2;

                let row_a_slice = &a_data[row_a_start..row_a_start + k1];
                let row_b_slice = &b_data[row_b_start..row_b_start + k2];

                let dot_product = row_a_slice.iter()
                .zip(row_b_slice.iter())
                .map(|(a,b)| *a * *b)
                .sum();

                if let Some(val) = result.get_mut(&[row_a_index,row_b_index]) {
                    *val = dot_product;
                }
                
            }
        }

        result

    }
}

#[cfg(test)]
mod tests {
    use crate::{dot::DotOps, NDArray};

    #[test]
    fn test_2d_mul() {
        
        let array_1 = NDArray::new(vec![1,2,3,4], vec![2,2]);
        
        let array_2 = NDArray::new(vec![1,2,3,4,5,6],vec![2,3]);


        let dot_product = array_1.dot_2d(&array_2);
        
        let expected_result = vec![9,12,15,19,26,33];

        assert!(dot_product.dims() == vec![2,3]);
        assert!(dot_product.data().as_ref() == &expected_result);

    }
}