use crate::NDArray;
use std::{ops::Range, sync::Arc};

pub trait SliceOps<T> {
    fn slice(&self,ranges: &[Range<usize>]) -> NDArray<T>;
}


impl<T> SliceOps<T> for NDArray<T> {
    fn slice(&self, ranges: &[Range<usize>]) -> NDArray<T> {
        assert_eq!(ranges.len(), self.dims().len(), "Number of ranges must match number of dimensions");

        let mut new_start = vec![0 as usize;ranges.len()];

        let mut new_dims = vec![0 as usize;ranges.len()];

        for (dim, range) in ranges.iter().enumerate() {
            let dim_len = self.dims()[dim];
            assert!(range.start <= range.end, "Range start must be <= end");
            assert!(range.end <= dim_len, "Range end {} out of bounds for dimension {} of length {}", range.end, dim, dim_len);
            new_start[dim] = range.start;
            new_dims[dim] = range.end - range.start;
        }


        let mut index_offset: isize = 0;

        for i in 0..self.dims().len() {
            index_offset += self.strides()[i] * (new_start[i] as isize);
        }

        let new_offset = (self.offset() as isize + index_offset) as usize;

        NDArray::from_parts(Arc::clone(&self.data()), new_dims,self.strides().to_vec(),new_offset)

    }
}



#[cfg(test)]
mod tests {
    use super::{SliceOps};
    use crate::NDArray;

    #[test]
    fn test_slice() {
        
        let array = NDArray::new(vec![0,1,2,3,4,5], vec![2,3]);

        let sliced_array = array.slice(&[1..2, 0..2]);

        assert_eq!(sliced_array.dims(),&vec![1,2]);

        assert_eq!(sliced_array.get(&[0,0]),Some(&3));
        assert_eq!(sliced_array.get(&[0,1]),Some(&4));

    }

    #[test]
    fn test_slice_of_slice() {
        let array = NDArray::new((0..25).collect(), vec![5, 5]); // 5x5 matrix
        let view1 = array.slice(&[1..4, 1..4]); // -> 3x3 view, offset should be 6
        assert_eq!(view1.get(&[0, 0]), Some(&6)); 
        
        let view2 = view1.slice(&[1..3, 0..2]); // -> 2x2 view from the 3x3 view
        assert_eq!(view2.dims(), &vec![2, 2]);
        
        assert_eq!(view2.get(&[0, 0]), Some(&11)); // Original index [2, 1]
        assert_eq!(view2.get(&[0, 1]), Some(&12)); // Original index [2, 2]
        assert_eq!(view2.get(&[1, 0]), Some(&16)); // Original index [3, 1]
        assert_eq!(view2.get(&[1, 1]), Some(&17)); // Original index [3, 2]
    }

    #[test]
    #[should_panic]
    fn test_slice_out_of_bounds() {
        let array = NDArray::new(vec![0, 1, 2, 3], vec![2, 2]);
        // This range goes past the end of dimension 1 (length 2)
        array.slice(&[0..1, 0..3]); 
    }

    #[test]
    fn test_1d_slice() {
        let array = NDArray::new((0..10).collect(), vec![10]);
        let sliced = array.slice(&[2..5]);
        assert_eq!(sliced.dims(), &vec![3]);
        assert_eq!(sliced.get(&[0]), Some(&2));
        assert_eq!(sliced.get(&[1]), Some(&3));
        assert_eq!(sliced.get(&[2]), Some(&4));
    }
}
