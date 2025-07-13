use crate::{NDArray, ops::ReductionOps};
use pyo3::{prelude::*, types::PyFloat};

enum DataType {
    F64(NDArray<f64>),
    F32(NDArray<f32>),
}

#[pyclass(name = "NDArray")]
struct NDArrayPy(DataType);


#[pymethods]
impl NDArrayPy {
    #[new]
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

    fn sum<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let obj: Py<PyAny> = match &self.0 {
            DataType::F64(arr) => PyFloat::new(py, arr.sum()).into(),
            DataType::F32(arr) => PyFloat::new(py, arr.sum() as f64).into(),
        };
        Ok(obj)
    }


    fn __repr__(&self) -> String{
        match &self.0 {
            DataType::F64(array) => format!("NDArray(shape={:?})", array.dims()),
            DataType::F32(array) => format!("NDArray(shape={:?})", array.dims())
        }
    }


    
    
}

#[pymodule]
fn synapse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NDArrayPy>()?;
    Ok(())
}
