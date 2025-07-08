use core::panic;
use std::iter::Sum;

use crate::{NDArray, TransformOps};

pub trait DotOps<T> {
    fn dot_2d(&self,other: &NDArray<T>) -> NDArray<T>;
    fn dot(&self,other: &NDArray<T>) -> NDArray<T>;
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

    fn dot(&self, other: &NDArray<T>) -> NDArray<T> {

        if self.dims().len() == 2 {
            return self.dot_2d(other);
        }

        let self_contig = self.to_contiguous();
        let other_contig = other.to_contiguous();

        let self_batch_dims = &self_contig.dims()[0..self_contig.dims().len() - 2];
        let other_batch_dims = &other_contig.dims()[0..other_contig.dims().len() - 2];

        if self_batch_dims != other_batch_dims {
            panic!("Batch dimensions should be same in dot");
        }

        let self_2d_dims = &self_contig.dims()[self.dims().len() - 2..self_contig.dims().len()];
        let other_2d_dims = &other_contig.dims()[other.dims().len() - 2..other_contig.dims().len()];

        let self_2d_size = self_2d_dims[0] * self_2d_dims[1];
        let other_2d_size = other_2d_dims[0] * other_2d_dims[1];

        if self_2d_dims[1] != other_2d_dims[0] {
            panic!("Dimensions for matrix multiplication are not compatible");
        }
    
        let result_2d_dims = &[self_2d_dims[0],other_2d_dims[1]];
        let result_2d_size = result_2d_dims[0] * result_2d_dims[1];

        let final_result_dims: Vec<usize>  = self_batch_dims.iter().cloned().chain(result_2d_dims.iter().cloned()).collect();

        let data_vec_len: usize = final_result_dims.iter().product();

        let final_result_data = vec![T::default(); data_vec_len];

        let mut final_result_array = NDArray::new(final_result_data, final_result_dims);

        let num_batches = self_batch_dims.iter().product();

        for i in 0..num_batches {
            
            let self_slice_start = i * self_2d_size;
            
            let other_slice_start = i * other_2d_size;

            let self_2d_data_slice = &self_contig.data()[self_slice_start..self_slice_start + self_2d_size];
            let other_2d_data_slice = &other_contig.data()[other_slice_start..other_slice_start + other_2d_size];

            let temp_self_2d = NDArray::new(self_2d_data_slice.to_vec(),self_2d_dims.to_vec());
            let temp_other_2d = NDArray::new(other_2d_data_slice.to_vec(),other_2d_dims.to_vec());

            let result_2d = temp_self_2d.dot_2d(&temp_other_2d);

            let result_slice_start = i * result_2d_size;

            final_result_array.data_mut()[result_slice_start .. result_slice_start + result_2d_size]
            .copy_from_slice(result_2d.data());

        }
        
        final_result_array


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

        assert_eq!(dot_product.dims(), &vec![2,3]);
        assert_eq!(dot_product.data().as_ref(), &expected_result);

    }

    #[test]
    fn test_3d_mul() {
        // a: shape [2, 2, 3]
        // b: shape [2, 3, 4]
        // result: shape [2, 2, 4]
    
        let a = NDArray::new(
            (1..=12).map(|x| x as i32).collect(), // Use a specific type like i32
            vec![2, 2, 3]
        );
    
        let b = NDArray::new(
            (1..=24).map(|x| x as i32).collect(),
            vec![2, 3, 4]
        );
    
        let dot_product = a.dot(&b);
    
        assert_eq!(dot_product.dims(), &vec![2, 2, 4]);
    
        // I've calculated the expected result for i32
        let expected_data: Vec<i32> = vec![
            // Batch 0 results
            38, 44, 50, 56,
            83, 98, 113, 128,
            // Batch 1 results
            416, 440, 464, 488,
            569, 602, 635, 668
        ];
    
        assert_eq!(dot_product.data().as_ref(), &expected_data);
    }
}
