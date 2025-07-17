use crate::{dot::DotOps, ops::ReductionOps, NDArray};
use pyo3::{prelude::*, types::PyFloat};


enum DataType {
    F64(NDArray<f64>),
    F32(NDArray<f32>),
}

#[pyclass]
struct NDArrayPy(DataType);


#[pymethods]
impl NDArrayPy {
    #[new]
    #[pyo3(signature = (obj, shape, dtype=None))]
    #[pyo3(text_signature = "(obj, shape, dtype=None)")]
    /// Create a new NDArray from a Python object.
    /// 
    /// Parameters
    /// ----------
    /// obj : list or array-like
    ///     The input data as a Python list or array-like object
    /// shape : list of int
    ///     The desired shape of the array
    /// dtype : str, optional
    ///     Data type ('f64' or 'f32'). If None, defaults to 'f64'
    /// 
    /// Returns
    /// -------
    /// NDArray
    ///     A new NDArray instance
    fn new(obj: Bound<'_, PyAny>, shape: Vec<usize>, dtype: Option<&str>) -> Self {

        match dtype {
            Some("f64") => {
                let rust_data: Vec<f64> = obj.extract().unwrap() ;
                let ndarray:NDArray<f64>  = NDArray::new(rust_data,shape);
                NDArrayPy(DataType::F64(ndarray))
            }
            Some("f32") => {
                let rust_data: Vec<f32> = obj.extract().unwrap() ;
                let ndarray:NDArray<f32>  = NDArray::new(rust_data,shape);
                NDArrayPy(DataType::F32(ndarray))
            }
            None => {
                // Default to f64 if dtype is not provided
                let rust_data: Vec<f64> = obj.extract().unwrap() ;
                let ndarray:NDArray<f64>  = NDArray::new(rust_data,shape);
                NDArrayPy(DataType::F64(ndarray))
            }
            Some(other) => panic!("Unsupported dtype: {}", other),
        }
    }

    /// Element-wise addition of two arrays.
    /// 
    /// Parameters
    /// ----------
    /// other : NDArray
    ///     The array to add to this array
    /// 
    /// Returns
    /// -------
    /// NDArray
    ///     A new array containing the element-wise sum
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If arrays have incompatible shapes or data types
    fn __add__(&self, other: &NDArrayPy) -> PyResult<NDArrayPy> {
        match (&self.0, &other.0) {
            
            (DataType::F64(a), DataType::F64(b)) => {
                let result = a.try_add(b).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
                Ok(NDArrayPy(DataType::F64(result)))
            }
            
            (DataType::F32(a), DataType::F32(b)) => {
                let result = a.try_add(b).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
                Ok(NDArrayPy(DataType::F32(result)))
            }

            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Mismatched dtypes for addition")),
        }
    }

    /// Element-wise multiplication of two arrays.
    /// 
    /// Parameters
    /// ----------
    /// other : NDArray
    ///     The array to multiply with this array
    /// 
    /// Returns
    /// -------
    /// NDArray
    ///     A new array containing the element-wise product
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If arrays have incompatible shapes or data types
    fn __mul__(&self, other: &NDArrayPy) -> PyResult<NDArrayPy> {
        
        match (&self.0, &other.0) {
            (DataType::F64(a), DataType::F64(b)) => {
                let result = a.try_mul(b).map_err(|e|PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
                Ok(NDArrayPy(DataType::F64(result)))
            }

            (DataType::F32(a), DataType::F32(b)) => {
                let result = a.try_mul(b).map_err(|e|PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
                Ok(NDArrayPy(DataType::F32(result)))
            }

           _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Mismatched dtypes for multiplication"))
        }
    }

    /// Element-wise division of two arrays.
    /// 
    /// Parameters
    /// ----------
    /// other : NDArray
    ///     The array to divide this array by
    /// 
    /// Returns
    /// -------
    /// NDArray
    ///     A new array containing the element-wise quotient
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If arrays have incompatible shapes or data types
    /// ZeroDivisionError
    ///     If division by zero occurs
    fn __truediv__(&self, other: &NDArrayPy) -> PyResult<NDArrayPy> {
        match (&self.0, &other.0) {
            (DataType::F64(a), DataType::F64(b)) => {
                let result = a.try_div(b).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
                Ok(NDArrayPy(DataType::F64(result)))
            }

            (DataType::F32(a), DataType::F32(b)) => {
                let result = a.try_div(b).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
                Ok(NDArrayPy(DataType::F32(result)))
            }

            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Mismatched dtypes for division")),
        }
    }

    #[pyo3(text_signature = "($self, other)")]
    /// Compute the dot product of two arrays.
    /// 
    /// Parameters
    /// ----------
    /// other : NDArray
    ///     The array to compute dot product with
    /// 
    /// Returns
    /// -------
    /// NDArray
    ///     The dot product result
    /// 
    /// Raises
    /// ------
    /// TypeError
    ///     If arrays have incompatible data types
    /// ValueError
    ///     If arrays have incompatible shapes for dot product
    fn dot(&self,other: &NDArrayPy) -> PyResult<NDArrayPy> {
        match (&self.0, &other.0) {
            
            (DataType::F64(a), DataType::F64(b)) => {
                let result = a.dot(b);
                Ok(NDArrayPy(DataType::F64(result)))
            }

            (DataType::F32(a), DataType::F32(b)) => {
                let result = a.dot(b);
                Ok(NDArrayPy(DataType::F32(result)))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Mismatched dtypes for dot product")),
        }
    }

    #[pyo3(text_signature = "($self, axis)")]
    /// Sum array elements over a given axis.
    /// 
    /// Parameters
    /// ----------
    /// axis : int
    ///     Axis along which the sum is performed
    /// 
    /// Returns
    /// -------
    /// NDArray
    ///     Array with the specified axis removed and elements summed
    /// 
    /// Raises
    /// ------
    /// IndexError
    ///     If axis is out of bounds
    fn sum_axis(&self, axis: usize) -> PyResult<NDArrayPy> {
        match &self.0 {
            DataType::F64(arr) => {
                let result = arr.sum_axis(axis);
                Ok(NDArrayPy(DataType::F64(result)))
            }
            DataType::F32(arr) => {
                let result = arr.sum_axis(axis);
                Ok(NDArrayPy(DataType::F32(result)))
            }
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Return the sum of all elements in the array.
    /// 
    /// Returns
    /// -------
    /// float
    ///     The sum of all array elements
    fn sum<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let obj: Py<PyAny> = match &self.0 {
            DataType::F64(arr) => PyFloat::new(py, arr.sum()).into(),
            DataType::F32(arr) => PyFloat::new(py, arr.sum() as f64).into(),
        };
        Ok(obj)
    }

    /// Return a string representation of the array.
    /// 
    /// Returns
    /// -------
    /// str
    ///     String representation showing the array shape
    fn __repr__(&self) -> String{
        match &self.0 {
            DataType::F64(array) => format!("NDArray(shape={:?})", array.dims()),
            DataType::F32(array) => format!("NDArray(shape={:?})", array.dims())
        }
    }

}

#[pymodule]
/// A fast n-dimensional array library for Python implemented in Rust.
fn _synapse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NDArrayPy>()?;
    Ok(())
}
