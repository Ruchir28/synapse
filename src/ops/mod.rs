pub mod arithmetic;
pub mod dot;
pub mod reduction;
pub mod transform;
pub mod slice;
pub mod broadcast;
pub mod arch;
pub mod shapes;

// Re-export the traits
pub use reduction::ReductionOps;
pub use transform::TransformOps;
pub use slice::SliceOps;
pub use shapes::ShapeOps;
