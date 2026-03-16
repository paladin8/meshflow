use std::fmt;

/// A 2D coordinate on the PE mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord {
    pub x: u32,
    pub y: u32,
}

impl Coord {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    /// Step in a direction, returning None if the result would be out of bounds
    /// for a mesh of the given dimensions.
    pub fn step(self, dir: Direction, width: u32, height: u32) -> Option<Coord> {
        match dir {
            Direction::East => {
                if self.x + 1 < width {
                    Some(Coord::new(self.x + 1, self.y))
                } else {
                    None
                }
            }
            Direction::West => {
                if self.x > 0 {
                    Some(Coord::new(self.x - 1, self.y))
                } else {
                    None
                }
            }
            Direction::North => {
                if self.y + 1 < height {
                    Some(Coord::new(self.x, self.y + 1))
                } else {
                    None
                }
            }
            Direction::South => {
                if self.y > 0 {
                    Some(Coord::new(self.x, self.y - 1))
                } else {
                    None
                }
            }
        }
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

/// Cardinal direction for mesh traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_east() {
        let c = Coord::new(1, 1);
        assert_eq!(c.step(Direction::East, 4, 4), Some(Coord::new(2, 1)));
    }

    #[test]
    fn step_west() {
        let c = Coord::new(1, 1);
        assert_eq!(c.step(Direction::West, 4, 4), Some(Coord::new(0, 1)));
    }

    #[test]
    fn step_north() {
        let c = Coord::new(1, 1);
        assert_eq!(c.step(Direction::North, 4, 4), Some(Coord::new(1, 2)));
    }

    #[test]
    fn step_south() {
        let c = Coord::new(1, 1);
        assert_eq!(c.step(Direction::South, 4, 4), Some(Coord::new(1, 0)));
    }

    #[test]
    fn step_east_at_boundary() {
        let c = Coord::new(3, 1);
        assert_eq!(c.step(Direction::East, 4, 4), None);
    }

    #[test]
    fn step_west_at_boundary() {
        let c = Coord::new(0, 1);
        assert_eq!(c.step(Direction::West, 4, 4), None);
    }

    #[test]
    fn step_north_at_boundary() {
        let c = Coord::new(1, 3);
        assert_eq!(c.step(Direction::North, 4, 4), None);
    }

    #[test]
    fn step_south_at_boundary() {
        let c = Coord::new(1, 0);
        assert_eq!(c.step(Direction::South, 4, 4), None);
    }

    #[test]
    fn step_origin_south_and_west() {
        let c = Coord::new(0, 0);
        assert_eq!(c.step(Direction::South, 4, 4), None);
        assert_eq!(c.step(Direction::West, 4, 4), None);
        assert_eq!(c.step(Direction::East, 4, 4), Some(Coord::new(1, 0)));
        assert_eq!(c.step(Direction::North, 4, 4), Some(Coord::new(0, 1)));
    }
}
