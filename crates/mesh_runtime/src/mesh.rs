use crate::coords::Coord;
use crate::pe::PE;

/// A 2D mesh of processing elements.
///
/// PEs are stored flat, indexed by `y * width + x`. All access should go
/// through the accessor methods, not by indexing `pes` directly.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub width: u32,
    pub height: u32,
    pes: Vec<PE>,
}

impl Mesh {
    /// Create a new mesh with the given dimensions. All PEs are initialized
    /// with empty SRAM, no tasks, and zero counters.
    pub fn new(width: u32, height: u32) -> Self {
        let mut pes = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                pes.push(PE::new(Coord::new(x, y)));
            }
        }
        Self { width, height, pes }
    }

    fn index(&self, coord: Coord) -> usize {
        assert!(
            coord.x < self.width && coord.y < self.height,
            "Coord {} out of bounds for {}x{} mesh",
            coord,
            self.width,
            self.height
        );
        (coord.y * self.width + coord.x) as usize
    }

    pub fn pe(&self, coord: Coord) -> &PE {
        let i = self.index(coord);
        &self.pes[i]
    }

    pub fn pe_mut(&mut self, coord: Coord) -> &mut PE {
        let i = self.index(coord);
        &mut self.pes[i]
    }

    /// Returns true if the coordinate is within mesh bounds.
    pub fn contains(&self, coord: Coord) -> bool {
        coord.x < self.width && coord.y < self.height
    }

    /// Iterate over all PEs in the mesh.
    pub fn iter_pes(&self) -> impl Iterator<Item = &PE> {
        self.pes.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_mesh_has_correct_dimensions() {
        let mesh = Mesh::new(3, 4);
        assert_eq!(mesh.width, 3);
        assert_eq!(mesh.height, 4);
    }

    #[test]
    fn all_pes_initialized_with_correct_coords() {
        let mesh = Mesh::new(3, 2);
        for y in 0..2 {
            for x in 0..3 {
                let coord = Coord::new(x, y);
                assert_eq!(mesh.pe(coord).coord, coord);
            }
        }
    }

    #[test]
    fn pe_mut_allows_modification() {
        let mut mesh = Mesh::new(2, 2);
        let coord = Coord::new(1, 1);
        mesh.pe_mut(coord).write_slot(0, vec![42.0]);
        assert_eq!(mesh.pe(coord).read_slot(0), &vec![42.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn pe_out_of_bounds_panics() {
        let mesh = Mesh::new(2, 2);
        mesh.pe(Coord::new(5, 0));
    }

    #[test]
    fn contains() {
        let mesh = Mesh::new(3, 4);
        assert!(mesh.contains(Coord::new(0, 0)));
        assert!(mesh.contains(Coord::new(2, 3)));
        assert!(!mesh.contains(Coord::new(3, 0)));
        assert!(!mesh.contains(Coord::new(0, 4)));
    }

    #[test]
    fn iter_pes_covers_all() {
        let mesh = Mesh::new(3, 2);
        assert_eq!(mesh.iter_pes().count(), 6);
    }
}
