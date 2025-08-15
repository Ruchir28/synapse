use core::panic;
use std::{any::TypeId, iter::Sum};

use crate::{ops::arch, NDArray, TransformOps};
use rayon::prelude::*;

pub trait DotOps<T> {
    fn dot_2d(&self, other: &NDArray<T>) -> NDArray<T>;
    fn dot(&self, other: &NDArray<T>) -> NDArray<T>;
    fn fallback_dot(&self, other: &NDArray<T>) -> NDArray<T>;
    fn dot_2d_tiled(&self, other: &NDArray<T>) -> NDArray<T>;
    fn dot_2d_tiled_kblocked(&self, other: &NDArray<T>) -> NDArray<T>;
}

impl<T> DotOps<T> for NDArray<T>
where
    T: Copy
        + Clone
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + Default
        + Sum<T>
        + 'static
        + Send
        + Sync
{
    fn dot_2d(&self, other: &NDArray<T>) -> NDArray<T> {
        let m = self.dims()[0];
        let k1 = self.dims()[1];

        let k2 = other.dims()[0];
        let n = other.dims()[1];

        if k1 != k2 {
            panic!("Dimensions for matrix multiplication are not compatible");
        }

        let a_contig = self.to_contiguous();

        // transposing the other to make things faster
        let b_transposed = other.permute_axis(&[1, 0]).to_contiguous();

        let a_data = a_contig.data();

        let b_data = b_transposed.data();

        let result_data = vec![T::default(); m * n];
        let mut result = NDArray::new(result_data, vec![m, n]);

        let type_is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();

        for row_a_index in 0..m {
            for row_b_index in 0..n {
                let row_a_start = row_a_index * k1;
                let row_b_start = row_b_index * k2;

                let row_a_slice = &a_data[row_a_start..row_a_start + k1];
                let row_b_slice = &b_data[row_b_start..row_b_start + k2];

                let mut dot_product = T::default();

                if type_is_f32 {
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") {
                            unsafe {
                                let a_slice = &*(row_a_slice as *const [T] as *const [f32]);
                                let b_slice = &*(row_b_slice as *const [T] as *const [f32]);
                                dot_product =
                                    *(&arch::aarch64::dot_f32_neon(a_slice, b_slice) as *const f32
                                        as *const T);
                            }
                        } else {
                            dot_product = row_a_slice.iter().zip(row_b_slice.iter()).map(|(a, b)| *a * *b).sum();
                        }
                    }

                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        dot_product = row_a_slice.iter().zip(row_b_slice.iter()).map(|(a, b)| *a * *b).sum();
                    }
                } else {
                    dot_product = row_a_slice.iter().zip(row_b_slice.iter()).map(|(a, b)| *a * *b).sum();
                }

                if let Some(val) = result.get_mut(&[row_a_index, row_b_index]) {
                    *val = dot_product;
                }
            }
        }
        result
    }

    fn dot_2d_tiled(&self, other: &NDArray<T>) -> NDArray<T>
     {
        const TILE_M: usize = 64;
        const TILE_N: usize = 64;

        let m = self.dims()[0];
        let k1 = self.dims()[1];

        let k2 = other.dims()[0];
        let n = other.dims()[1];

        if k1 != k2 {
            panic!("Dimensions for matrix multiplication are not compatible");
        }

        let a = self.to_contiguous();
        let b = other.permute_axis(&[1,0]).to_contiguous(); // TRANSPOSED B

        let a_data: &[T] = a.data();
        let b_data: &[T] = b.data();

        let mut result = NDArray::new(vec![T::default(); m * n], vec![m, n]);

        let type_is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();

        let c_base = result.data_mut().as_mut_ptr() as usize;

        let m_tiles = (m + TILE_M - 1) / TILE_M;
        let n_tiles = (n + TILE_N - 1) / TILE_N;

        (0..m_tiles * n_tiles).into_par_iter().for_each(|tile_id| {

            let c_ptr = c_base as *mut T;

            let tile_r: usize = tile_id / n_tiles;
            let tile_c = tile_id % n_tiles;

            let i0 = tile_r * TILE_M;
            let i1 = (i0 + TILE_M).min(m);

            let j0 = tile_c * TILE_N;
            let j1 = (j0 + TILE_N).min(n);

            for i in i0..i1 {
                for j in j0..j1 {

                    let row_a_slice = &a_data[i * k1 .. i * k1 + k1];
                    let row_b_slice = &b_data[j * k2 .. j * k2 + k2];
                    
                    let mut dot_product = T::default();
                    
                    if type_is_f32 {
                        #[cfg(target_arch = "aarch64")]
                        {
                            if std::arch::is_aarch64_feature_detected!("neon") {
                                unsafe {
                                    let a_slice = &*(row_a_slice as *const [T] as *const [f32]);
                                    let b_slice = &*(row_b_slice as *const [T] as *const [f32]);
                                    let dp = arch::aarch64::dot_f32_neon(a_slice, b_slice);
                                    dot_product = *(&dp as *const f32 as *const T);
                                }
                            } else {
                                dot_product = row_a_slice.iter().zip(row_b_slice.iter()).map(|(a,b)| *a * *b).sum();
                            }
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            dot_product = row_a_slice.iter().zip(row_b_slice.iter()).map(|(a,b)| *a * *b).sum();
                        }
                    } else {
                        dot_product = row_a_slice.iter().zip(row_b_slice.iter()).map(|(a,b)| *a * *b).sum();
                    }
                    
                    unsafe { *c_ptr.add(i * n + j) = dot_product; }               
                }
            }
        });


        result
    }

    
    fn dot_2d_tiled_kblocked(&self, other: &NDArray<T>) -> NDArray<T> {
        const TILE_M: usize = 64;
        const TILE_N: usize = 64;
        const TILE_K: usize = 64;

        let m = self.dims()[0];
        let k1 = self.dims()[1];

        let k2 = other.dims()[0];
        let n = other.dims()[1];

        if k1 != k2 {
            panic!("Dimensions for matrix multiplication are not compatible");
        }

        let a = self.to_contiguous();
        let b = other.permute_axis(&[1,0]).to_contiguous(); // TRANSPOSED B

        let a_data: &[T] = a.data();
        let b_data: &[T] = b.data();

        let mut result = NDArray::new(vec![T::default(); m * n], vec![m, n]);

        let type_is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();

        let c_base = result.data_mut().as_mut_ptr() as usize;

        let m_tiles = (m + TILE_M - 1) / TILE_M;
        let n_tiles = (n + TILE_N - 1) / TILE_N;

        (0..m_tiles * n_tiles).into_par_iter().for_each(|tile_id| {

            let c_ptr = c_base as *mut T;

            let tile_r: usize = tile_id / n_tiles;
            let tile_c = tile_id % n_tiles;

            let i0 = tile_r * TILE_M;
            let i1 = (i0 + TILE_M).min(m);

            let j0 = tile_c * TILE_N;
            let j1 = (j0 + TILE_N).min(n);

            let tile_h = i1 - i0;
            let tile_w = j1 - j0;

            const MR: usize = 4;  // Process 4 rows of C at once
            const NR: usize = 8;  // Process 8 cols of C at once

            let mut c_tile = vec![T::default(); tile_h * tile_w];

            let mut k_block_start = 0;
            while k_block_start < k1 {
                let k_block_end = (k_block_start + TILE_K).min(k1);
                let _k_block_size = k_block_end - k_block_start;

                for i_micro in (0..tile_h).step_by(MR) {
                    for j_micro in (0..tile_w).step_by(NR) {
                        let i_end = (i_micro + MR).min(tile_h);
                        let j_end = (j_micro + NR).min(tile_w);

                        let mut a_micro_rows: [&[T]; MR] = [&[]; MR];
                        let mut b_micro_rows: [&[T]; NR] = [&[]; NR];

                        for i_local in 0..(i_end - i_micro) {
                            let global_row = i0 + i_micro + i_local;
                            let a_row_slice = &a_data[global_row * k1 + k_block_start..global_row * k1 + k_block_end];
                            a_micro_rows[i_local] = a_row_slice;
                        }

                        for j_local in 0..(j_end - j_micro) {
                            let global_col = j0 + j_micro + j_local;
                            let b_row_slice = &b_data[global_col * k1 + k_block_start..global_col * k1 + k_block_end];
                            b_micro_rows[j_local] = b_row_slice;
                        }

                        let actual_mr = i_end - i_micro;
                        let actual_nr = j_end - j_micro;
                        
                        for i_local in 0..actual_mr {
                            for j_local in 0..actual_nr {
                                let a_row = a_micro_rows[i_local];
                                let b_row = b_micro_rows[j_local];
                                
                                let mut dot_product = T::default();
                                
                                if type_is_f32 {
                                    #[cfg(target_arch = "aarch64")]
                                    {
                                        if std::arch::is_aarch64_feature_detected!("neon") {
                                            unsafe {
                                                let a_slice = &*(a_row as *const [T] as *const [f32]);
                                                let b_slice = &*(b_row as *const [T] as *const [f32]);
                                                let dp = arch::aarch64::dot_f32_neon(a_slice, b_slice);
                                                dot_product = *(&dp as *const f32 as *const T);
                                            }
                                        } else {
                                            dot_product = a_row.iter().zip(b_row.iter()).map(|(a, b)| *a * *b).sum();
                                        }
                                    }
                                    #[cfg(not(target_arch = "aarch64"))]
                                    {
                                        dot_product = a_row.iter().zip(b_row.iter()).map(|(a, b)| *a * *b).sum();
                                    }
                                } else {
                                    dot_product = a_row.iter().zip(b_row.iter()).map(|(a, b)| *a * *b).sum();
                                }

                                // Update c_tile with correct indexing
                                let c_tile_row = i_micro + i_local;
                                let c_tile_col = j_micro + j_local;
                                let c_idx = c_tile_row * tile_w + c_tile_col;
                                c_tile[c_idx] = c_tile[c_idx] + dot_product;
                            }
                        }
                        
                    }
                }
                k_block_start = k_block_end;
            }

            unsafe  {
                for r_local in 0..tile_h {
                    let src = &c_tile[r_local * tile_w .. (r_local + 1) * tile_w];
                    let dst_row = i0 + r_local;
                    let dst = c_ptr.add(dst_row * n + j0);
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, tile_w);
                }

            }
        });

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

    fn fallback_dot(&self, other: &NDArray<T>) -> NDArray<T> {
        let m = self.dims()[0];
        let k1 = self.dims()[1];
        let k2 = other.dims()[0];
        let n = other.dims()[1];

        if k1 != k2 {
            panic!("Dimensions for matrix multiplication are not compatible");
        }

        let a_contig = self.to_contiguous();
        let b_transposed = other.permute_axis(&[1, 0]).to_contiguous();
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

                let dot_product = row_a_slice
                    .iter()
                    .zip(row_b_slice.iter())
                    .map(|(a, b)| *a * *b)
                    .sum();

                if let Some(val) = result.get_mut(&[row_a_index, row_b_index]) {
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

        assert_eq!(dot_product.dims(), &vec![2,3]);
        assert_eq!(dot_product.data().as_ref(), &expected_result);

    }

    #[test]
    fn test_2d_mul_tiled_kblocked_small_int() {
        use crate::ops::dot::DotOps;

        // 2x2 · 2x3 (ints)
        let a = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
        let b = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

        let baseline = a.dot_2d(&b);
        let tiled_k = a.dot_2d_tiled_kblocked(&b);

        assert_eq!(tiled_k.dims(), baseline.dims());
        assert_eq!(tiled_k.data(), baseline.data());
    }

    #[test]
    fn test_2d_mul_tiled_kblocked_rect_f32() {
        use crate::ops::dot::DotOps;

        // Non-multiple tile sizes to exercise edge paths
        let m = 7; let k = 5; let n = 9;
        let a = NDArray::new((0..m*k).map(|x| (x as f32) * 0.5 + 1.0).collect(), vec![m, k]);
        let b = NDArray::new((0..k*n).map(|x| (x as f32) * 0.25 - 0.75).collect(), vec![k, n]);

        // Use scalar fallback as a stable reference
        let baseline = a.fallback_dot(&b);
        let tiled_k = a.dot_2d_tiled_kblocked(&b);

        assert_eq!(tiled_k.dims(), baseline.dims());

        // Allow small FP differences due to K-block accumulation order
        let left = tiled_k.data();
        let right = baseline.data();
        let eps: f32 = 1e-5;
        for (idx, (l, r)) in left.iter().zip(right.iter()).enumerate() {
            let diff = (l - r).abs();
            let tol = eps * (1.0 + l.abs().max(r.abs()));
            assert!(diff <= tol, "mismatch at {}: left={}, right={}, diff={}, tol={}", idx, l, r, diff, tol);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_2d_mul_tiled_kblocked_f32_neon_matches() {
        use crate::ops::dot::DotOps;

        if std::arch::is_aarch64_feature_detected!("neon") {
            let m = 31; let k = 37; let n = 23; // odd sizes
            let a = NDArray::new((0..m*k).map(|x| (x as f32).sin()).collect(), vec![m,k]);
            let b = NDArray::new((0..k*n).map(|x| (x as f32).cos()).collect(), vec![k,n]);

            let baseline = a.fallback_dot(&b); // pure scalar
            let tiled_k = a.dot_2d_tiled_kblocked(&b);

            assert_eq!(tiled_k.dims(), baseline.dims());
            // Allow small FP differences between different reduction orders
            let left = tiled_k.data();
            let right = baseline.data();
            let eps: f32 = 1e-5;
            for (idx, (l, r)) in left.iter().zip(right.iter()).enumerate() {
                let diff = (l - r).abs();
                let tol = eps * (1.0 + l.abs().max(r.abs()));
                assert!(diff <= tol, "mismatch at {}: left={}, right={}, diff={}, tol={}", idx, l, r, diff, tol);
            }

        }
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

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_2d_mul_f32_neon() {
        if std::arch::is_aarch64_feature_detected!("neon") {
            let array_1 = NDArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            let array_2 = NDArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

            let dot_product = array_1.dot_2d(&array_2);

            let expected_result = vec![9.0, 12.0, 15.0, 19.0, 26.0, 33.0];

            assert_eq!(dot_product.dims(), &vec![2, 3]);
            assert_eq!(dot_product.data().as_ref(), &expected_result);
        }
    }


    #[test]
    fn test_2d_mul_tiled_small_int() {
        use crate::ops::dot::DotOps;

        // 2x2 · 2x3 (ints)
        let a = NDArray::new(vec![1,2,3,4], vec![2,2]);
        let b = NDArray::new(vec![1,2,3,4,5,6], vec![2,3]);

        let baseline = a.dot_2d(&b);
        let tiled = a.dot_2d_tiled(&b);

        assert_eq!(tiled.dims(), baseline.dims());
        assert_eq!(tiled.data(), baseline.data());
    }

    #[test]
    fn test_2d_mul_tiled_rect_f32() {
        use crate::ops::dot::DotOps;

        // 7x5 · 5x9 (f32), checks non-multiple of tile sizes
        let m = 7; let k = 5; let n = 9;
        let a = NDArray::new((0..m*k).map(|x| (x as f32) * 0.5 + 1.0).collect(), vec![m,k]);
        let b = NDArray::new((0..k*n).map(|x| (x as f32) * 0.25 - 0.75).collect(), vec![k,n]);

        let baseline = a.dot_2d(&b);
        let tiled = a.dot_2d_tiled(&b);

        assert_eq!(tiled.dims(), baseline.dims());
        assert_eq!(tiled.data(), baseline.data());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_2d_mul_tiled_f32_neon_matches() {
        use crate::ops::dot::DotOps;

        if std::arch::is_aarch64_feature_detected!("neon") {
            let m = 31; let k = 37; let n = 23; // odd sizes
            let a = NDArray::new((0..m*k).map(|x| (x as f32).sin()).collect(), vec![m,k]);
            let b = NDArray::new((0..k*n).map(|x| (x as f32).cos()).collect(), vec![k,n]);

            let baseline = a.fallback_dot(&b); // pure scalar
            let tiled = a.dot_2d_tiled(&b);

            assert_eq!(tiled.dims(), baseline.dims());
            // Allow small FP differences between SIMD and scalar reductions
            let left = tiled.data();
            let right = baseline.data();
            let eps: f32 = 1e-5;
            for (idx, (l, r)) in left.iter().zip(right.iter()).enumerate() {
                let diff = (l - r).abs();
                let tol = eps * (1.0 + l.abs().max(r.abs()));
                assert!(diff <= tol, "mismatch at {}: left={}, right={}, diff={}, tol={}", idx, l, r, diff, tol);
            }
        }
    }
}
