use pyo3::prelude::*;

pub mod coords;
pub mod event;
pub mod mesh;
pub mod message;
pub mod pe;
pub mod profiling;
pub mod route;

/// Returns the version of the mesh runtime.
#[pyfunction]
fn runtime_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// meshflow._mesh_runtime — Rust core for the Meshflow spatial inference engine.
#[pymodule]
fn _mesh_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(runtime_version, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_version() {
        assert_eq!(runtime_version(), "0.1.0");
    }
}
