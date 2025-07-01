use crate::NDArrayError;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

#[derive(Debug)]
pub struct NDArray<T> {
    data: Arc<Vec<T>>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

impl<T> NDArray<T> {
    pub fn new(data: Vec<T>, dims: Vec<usize>) -> Self {
        if data.len() != dims.iter().product() {
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

    pub fn try_add(&self, rhs: &NDArray<T>) -> Result<NDArray<T>, NDArrayError>
    where
        T: Add<Output = T> + Copy,
    {
        if self.dims() != rhs.dims() {
            return Err(NDArrayError::DimensionMismatch {
                expected: self.dims().to_vec(),
                found: rhs.dims().to_vec(),
            });
        }
        let result_data: Vec<T> = self.iter().zip(rhs.iter()).map(|(a, b)| *a + *b).collect();
        Ok(NDArray::new(result_data, self.dims().to_vec()))
    }

    pub fn try_sub(&self, rhs: &NDArray<T>) -> Result<NDArray<T>, NDArrayError>
    where
        T: Sub<Output = T> + Copy,
    {
        if self.dims() != rhs.dims() {
            return Err(NDArrayError::DimensionMismatch {
                expected: self.dims().to_vec(),
                found: rhs.dims().to_vec(),
            });
        }
        let result_data: Vec<T> = self.iter().zip(rhs.iter()).map(|(a, b)| *a - *b).collect();
        Ok(NDArray::new(result_data, self.dims().to_vec()))
    }

    pub fn try_mul(&self, rhs: &NDArray<T>) -> Result<NDArray<T>, NDArrayError>
    where
        T: Mul<Output = T> + Copy,
    {
        if self.dims() != rhs.dims() {
            return Err(NDArrayError::DimensionMismatch {
                expected: self.dims().to_vec(),
                found: rhs.dims().to_vec(),
            });
        }
        let result_data: Vec<T> = self.iter().zip(rhs.iter()).map(|(a, b)| *a * *b).collect();
        Ok(NDArray::new(result_data, self.dims().to_vec()))
    }

    pub fn try_div(&self, rhs: &NDArray<T>) -> Result<NDArray<T>, NDArrayError>
    where
        T: Div<Output = T> + Copy,
    {
        if self.dims() != rhs.dims() {
            return Err(NDArrayError::DimensionMismatch {
                expected: self.dims().to_vec(),
                found: rhs.dims().to_vec(),
            });
        }
        let result_data: Vec<T> = self.iter().zip(rhs.iter()).map(|(a, b)| *a / *b).collect();
        Ok(NDArray::new(result_data, self.dims().to_vec()))
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
