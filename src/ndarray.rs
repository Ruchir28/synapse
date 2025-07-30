use crate::{ops::broadcast::broadcast_shapes, NDArrayError};
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;
use std::borrow::Cow;

#[derive(Debug)]
pub struct NDArray<T> {
    data: Arc<Vec<T>>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

// The Clone implementation for NDArray is cheap because the data is stored in an Arc.
// It just creates a new view pointing to the same underlying data.
impl<T> Clone for NDArray<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}

impl<T> NDArray<T> {
    pub fn new(data: Vec<T>, dims: Vec<usize>) -> Self {
        if data.len() != dims.iter().product::<usize>() {
            panic!("Data length does not match dimensions");
        }

        let strides = Self::calculate_strides(&dims);

        Self {
            data: Arc::new(data),
            dims,
            strides,
            offset: 0
        }
    }

    fn calculate_strides(dims: &[usize]) -> Vec<isize> {
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i+1] * dims[i+1] as isize;
        }
        strides
    }

    pub fn get(&self, index: &[usize]) -> Option<&T> {
        
        if index.len() != self.dims.len() {
            return None;
        }

        for i in 0..self.dims.len() {
            if index[i] >= self.dims[i] {
                return None;
            }
        }

        let mut index_offset: isize = 0;

        for i in 0..self.dims.len() {
            index_offset += self.strides[i] * (index[i] as isize);
        }

        let data_index = (self.offset as isize + index_offset) as usize;

        if data_index > self.data.len() {
            return None;
        }

        return self.data.get(data_index);
    }
    
    /// Returns a reference to the dimensions of the array.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns a reference to the strides of the array.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Returns the offset of the array.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns a reference to the data of the array.
    pub fn data(&self) -> &Arc<Vec<T>> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut Vec<T>  {
        Arc::get_mut(&mut self.data).expect("Data is not mutable")
    }

    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        if index.len() != self.dims.len() {
            return None;
        }

        for i in 0..self.dims.len() {
            if index[i] >= self.dims[i] {
                return None;
            }
        }

        let mut index_offset: isize = 0;

        for i in 0..self.dims.len() {
            index_offset += self.strides[i] * (index[i] as isize);
        }

        let data_index = (self.offset as isize + index_offset) as usize;

        if data_index >= self.data.len() {
            return None;
        }

        let data_ref = Arc::get_mut(&mut self.data).expect("Data is not mutable");
        
        data_ref.get_mut(data_index)
    }

    pub fn iter(&self) -> NdArrayIter<T> {
        NdArrayIter { array: self, current_index: vec![0;self.dims.len()], finished: self.dims.iter().product::<usize>() == 0}
    }

    pub fn indexed_iter(&self) -> NdArrayIndexedIter<T> {
        NdArrayIndexedIter { array: self, current_index: vec![0;self.dims().len()], finished: self.dims().iter().product::<usize>() == 0}
    }

    pub(crate) fn from_parts(
        data: Arc<Vec<T>>,
        dims: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
    ) -> Self {
        Self {
            data,
            dims,
            strides,
            offset,
        }
    }
    
    pub fn is_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        if self.dims.iter().product::<usize>() != self.data.len() {
            return false;
        }
        let standard_strides = Self::calculate_strides(&self.dims);
        self.strides == standard_strides
    }

    pub fn to_contiguous(&self) -> Self
    where
        T: Clone,
    {
        if self.is_contiguous() {
            return self.clone();
        }

        let new_data: Vec<T> = self.iter().cloned().collect();

        NDArray::new(new_data, self.dims.clone())
    }

    pub fn try_add(&self, rhs: &NDArray<T>) -> Result<NDArray<T>, NDArrayError>
    where
        T: Add<Output = T> + Copy + Default + 'static,
    {
        // SIMD fast path for aarch64
        #[cfg(target_arch = "aarch64")]
        if self.dims == rhs.dims && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let self_contig = if self.is_contiguous() { Cow::Borrowed(self) } else { Cow::Owned(self.to_contiguous()) };
            let other_contig = if rhs.is_contiguous() { Cow::Borrowed(rhs) } else { Cow::Owned(rhs.to_contiguous()) };

            let num_elements = self.dims().iter().product();
            let mut result = NDArray::new(vec![T::default(); num_elements], self.dims.clone());

            unsafe {
                let a_slice: &[f32] = std::mem::transmute(self_contig.data().as_ref() as &[T]);
                let b_slice: &[f32] = std::mem::transmute(other_contig.data().as_ref() as &[T]);
                let r_slice: &mut [f32] = std::mem::transmute(result.data_mut() as &mut [T]);

                crate::ops::arch::aarch64::add_f32_neon(a_slice, b_slice, r_slice);
            }

            return Ok(result);
        }

        self.fallback_add(rhs)
    }

    pub fn fallback_add(&self, rhs: &NDArray<T>) -> Result<NDArray<T>, NDArrayError>
    where
       T: Add<Output = T> + Copy + Default + 'static,
    {
        // Fallback for other architectures or types
        let info = broadcast_shapes(self.dims(), rhs.dims(), self.strides(), rhs.strides())?;

        let a_view = NDArray::from_parts(
            self.data.clone(),
            info.result_shape.clone(),
            info.a_strides,
            self.offset,
        );

        let b_view = NDArray::from_parts(
            rhs.data.clone(),
            info.result_shape.clone(),
            info.b_strides,
            rhs.offset,
        );

        let result_data: Vec<T> = a_view.iter().zip(b_view.iter()).map(|(a, b)| *a + *b).collect();
        Ok(NDArray::new(result_data, info.result_shape))
    }

    pub fn try_sub(&self, rhs: &NDArray<T>) -> Result<NDArray<T>, NDArrayError>
    where
        T: Sub<Output = T> + Copy + Default + 'static,
    {
        #[cfg(target_arch = "aarch64")]
        if self.dims == rhs.dims && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let self_contig = if self.is_contiguous() {Cow::Borrowed(self)} else {Cow::Owned(self.to_contiguous())};
            let rhs_contig = if rhs.is_contiguous() {Cow::Borrowed(rhs)} else {Cow::Owned(rhs.to_contiguous())};

            let num_elements = self.dims().iter().product();

            let mut result = NDArray::new(vec![T::default();num_elements], self.dims.clone());

            unsafe {
                let a_slice: &[f32] = std::mem::transmute(self_contig.data().as_ref() as &[T]);
                let b_slice: &[f32] = std::mem::transmute(rhs_contig.data().as_ref() as &[T]);
                let r_slice: &mut [f32] = std::mem::transmute(result.data_mut() as &mut [T]);

                crate::ops::arch::aarch64::sub_f32_neon(a_slice, b_slice, r_slice);
            }

            return Ok(result);
        } 

        self.fallback_sub(rhs)

    }

    pub fn fallback_sub(&self, rhs: &NDArray<T>) -> Result<NDArray<T>,NDArrayError> 
    where 
        T: Sub<Output = T> + Copy + Default + 'static,
    {
        let info = broadcast_shapes(self.dims(), rhs.dims(), self.strides(), rhs.strides())?;

        let a_view = NDArray::from_parts(
            self.data.clone(),
            info.result_shape.clone(),
            info.a_strides,
            self.offset,
        );

        let b_view = NDArray::from_parts(
            rhs.data.clone(),
            info.result_shape.clone(),
            info.b_strides,
            rhs.offset,
        );

        let result_data: Vec<T> = a_view.iter().zip(b_view.iter()).map(|(a, b)| *a - *b).collect();
        Ok(NDArray::new(result_data, info.result_shape))
    }

    pub fn try_mul(&self, rhs: &NDArray<T>) -> Result<NDArray<T>, NDArrayError>
    where
        T: Mul<Output = T> + Copy + Default + 'static,
    {

        #[cfg(target_arch = "aarch64")]
        if self.dims == rhs.dims && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let self_contig = if self.is_contiguous() { Cow::Borrowed(self) } else { Cow::Owned(self.to_contiguous()) };
            let other_contig = if rhs.is_contiguous() { Cow::Borrowed(rhs) } else { Cow::Owned(rhs.to_contiguous()) };

            let num_elements = self.dims().iter().product();
            let mut result = NDArray::new(vec![T::default(); num_elements], self.dims.clone());

            unsafe {
                let a_slice: &[f32] = std::mem::transmute(self_contig.data().as_ref() as &[T]);
                let b_slice: &[f32] = std::mem::transmute(other_contig.data().as_ref() as &[T]);
                let r_slice: &mut [f32] = std::mem::transmute(result.data_mut() as &mut [T]);

                crate::ops::arch::aarch64::multiply_f32_neon(a_slice, b_slice, r_slice);
            }

            return Ok(result);
        }

        self.fallback_mul(rhs)
    }

    pub fn fallback_mul(&self,rhs: &NDArray<T>) -> Result<NDArray<T>, NDArrayError>
    where
        T : Mul<Output = T> + Copy
    {
        let info = broadcast_shapes(self.dims(), rhs.dims(), self.strides(), rhs.strides())?;

        let a_view = NDArray::from_parts(
            self.data.clone(),
            info.result_shape.clone(),
            info.a_strides,
            self.offset,
        );

        let b_view = NDArray::from_parts(
            rhs.data.clone(),
            info.result_shape.clone(),
            info.b_strides,
            rhs.offset,
        );

        let result_data: Vec<T> = a_view.iter().zip(b_view.iter()).map(|(a, b)| *a * *b).collect();
        Ok(NDArray::new(result_data, info.result_shape))
    }

    pub fn try_div(&self, rhs: &NDArray<T>) -> Result<NDArray<T>, NDArrayError>
    where
        T: Div<Output = T> + Copy + Default + 'static,
    {

        #[cfg(target_arch = "aarch64")]
        if self.dims == rhs.dims && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let self_contig = if self.is_contiguous() { Cow::Borrowed(self) } else { Cow::Owned(self.to_contiguous()) };
            let other_contig = if rhs.is_contiguous() { Cow::Borrowed(rhs) } else { Cow::Owned(rhs.to_contiguous()) };

            let num_elements = self.dims().iter().product();
            let mut result = NDArray::new(vec![T::default(); num_elements], self.dims.clone());

            unsafe {
                let a_slice: &[f32] = std::mem::transmute(self_contig.data().as_ref() as &[T]);
                let b_slice: &[f32] = std::mem::transmute(other_contig.data().as_ref() as &[T]);
                let r_slice: &mut [f32] = std::mem::transmute(result.data_mut() as &mut [T]);

                crate::ops::arch::aarch64::divide_f32_neon(a_slice, b_slice, r_slice);
            }

            return Ok(result);
        }

        self.fallback_div(rhs)
    }

    pub fn fallback_div(&self, rhs: &NDArray<T>) -> Result<NDArray<T>, NDArrayError>
    where
        T: Div<Output = T> + Copy + Default + 'static,
    {
        let info = broadcast_shapes(self.dims(), rhs.dims(), self.strides(), rhs.strides())?;

        let a_view = NDArray::from_parts(
            self.data.clone(),
            info.result_shape.clone(),
            info.a_strides,
            self.offset,
        );

        let b_view = NDArray::from_parts(
            rhs.data.clone(),
            info.result_shape.clone(),
            info.b_strides,
            rhs.offset,
        );

        let result_data: Vec<T> = a_view.iter().zip(b_view.iter()).map(|(a, b)| *a / *b).collect();
        Ok(NDArray::new(result_data, info.result_shape))
    }

}

#[derive(Debug)]
pub struct NdArrayIter<'a, T> {
    array: &'a NDArray<T>,
    current_index: Vec<usize>,
    finished: bool,
}

impl<'a, T> Iterator for NdArrayIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let item = self.array.get(&self.current_index).unwrap();

        for dim_index in (0..self.current_index.len()).rev() {

            self.current_index[dim_index] += 1;
                        
            if self.current_index[dim_index] <  self.array.dims()[dim_index] {
                return Some(item);            
            }
            
            self.current_index[dim_index] = 0;            
        }

        self.finished = true;

        return Some(item);
    }
}

#[derive(Debug)]
pub struct NdArrayIndexedIter<'a, T> {
    array: &'a NDArray<T>,
    current_index: Vec<usize>,
    finished: bool
}

impl<'a, T> Iterator for NdArrayIndexedIter<'a,T> {

    type Item = (Vec<usize>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        
        if self.finished {
            return None;
        }

        let index_to_return = self.current_index.clone();
        let item = self.array.get(&self.current_index).unwrap();

        // The "odometer" logic to increment the index remains the same
        for dim_index in (0..self.current_index.len()).rev() {
            self.current_index[dim_index] += 1;
            if self.current_index[dim_index] < self.array.dims()[dim_index] {
                // Return the index we saved earlier, along with the item
                return Some((index_to_return, item));
            }
            self.current_index[dim_index] = 0;
        }

        self.finished = true;

        // Return the last item
        Some((index_to_return, item))

    }

}

#[cfg(test)]
mod tests {
    use crate::ops::{SliceOps, TransformOps};
    use super::*;

    #[test]
    fn test_is_contiguous() {
            let a = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
            assert!(a.is_contiguous());

            let b = a.permute_axis(&[1, 0]); // Transpose
            assert!(!b.is_contiguous());

            let c = a.slice(&[0..1, 0..2]); // Slice
            assert!(!c.is_contiguous());
        }

    #[test]
    fn test_to_contiguous_from_view() {
        let a = NDArray::new((0..9).collect(), vec![3, 3]);
        let b = a.permute_axis(&[1, 0]); // Transposed view
        assert!(!b.is_contiguous());

        // Get a value from the view to make sure it's correct before copy
        assert_eq!(b.get(&[0, 1]), Some(&3)); // B[0,1] is A[1,0]

        let c = b.to_contiguous();
        assert!(c.is_contiguous());

        // Check that the data in C is the transposed data
        let expected_data: Vec<i32> = vec![0, 3, 6, 1, 4, 7, 2, 5, 8];
        assert_eq!(c.data().as_ref(), &expected_data);
        assert_eq!(c.dims(), &[3, 3]);

        // Check that the original array is untouched
        assert_eq!(a.data().as_ref(), &(0..9).collect::<Vec<i32>>());
    }

    #[test]
    fn test_to_contiguous_from_slice() {
            let array = NDArray::new((0..25).collect(), vec![5, 5]);
            let view = array.slice(&[1..4, 1..4]); // 3x3 view, offset 6
            assert!(!view.is_contiguous());

            assert_eq!(view.get(&[0, 0]), Some(&6));
            assert_eq!(view.get(&[2, 2]), Some(&18));

            let contiguous_view = view.to_contiguous();
            assert!(contiguous_view.is_contiguous());
            assert_eq!(contiguous_view.dims(), &[3, 3]);

            // Check the data is now dense
            let expected_data: Vec<i32> = vec![
                6, 7, 8,
                11, 12, 13,
                16, 17, 18
            ];
            assert_eq!(contiguous_view.data().as_ref(), &expected_data);

            // Check that getting an element from the new array works
            assert_eq!(contiguous_view.get(&[0, 0]), Some(&6));
            assert_eq!(contiguous_view.get(&[2, 2]), Some(&18));
        }

    #[test]
    fn test_add_f32_simd_path() {
        // This test is designed to trigger the SIMD fast path.
        // It uses f32, identical shapes, and on a supported CPU, it should use NEON.
        let a = NDArray::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        let b = NDArray::new(vec![9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], vec![3, 3]);

        // We need to use try_add to match the implementation location
        let result = a.try_add(&b).unwrap();

        let expected_data: Vec<f32> = vec![10.0; 9];
        assert_eq!(result.data().as_ref(), &expected_data);
        assert_eq!(result.dims(), &[3, 3]);
    }
}
