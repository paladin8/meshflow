use crate::coords::{Coord, Direction};

/// Generate a hop list using dimension-ordered (XY) routing.
///
/// Moves along the X axis first (East/West), then the Y axis (North/South).
/// Returns an empty list if source and destination are the same PE.
///
/// This is the only route generator in M1. The runtime consumes hop lists,
/// not this function — swapping to a different algorithm later means replacing
/// this function call where hop lists are built, not changing the runtime.
pub fn generate_route_xy(from: Coord, to: Coord) -> Vec<Direction> {
    let mut hops = Vec::new();

    // X axis first
    let x_dir = if to.x > from.x {
        Some(Direction::East)
    } else if to.x < from.x {
        Some(Direction::West)
    } else {
        None
    };
    if let Some(dir) = x_dir {
        for _ in 0..to.x.abs_diff(from.x) {
            hops.push(dir);
        }
    }

    // Y axis second
    let y_dir = if to.y > from.y {
        Some(Direction::North)
    } else if to.y < from.y {
        Some(Direction::South)
    } else {
        None
    };
    if let Some(dir) = y_dir {
        for _ in 0..to.y.abs_diff(from.y) {
            hops.push(dir);
        }
    }

    hops
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_pe_returns_empty() {
        assert!(generate_route_xy(Coord::new(1, 1), Coord::new(1, 1)).is_empty());
    }

    #[test]
    fn adjacent_east() {
        let hops = generate_route_xy(Coord::new(0, 0), Coord::new(1, 0));
        assert_eq!(hops, vec![Direction::East]);
    }

    #[test]
    fn adjacent_west() {
        let hops = generate_route_xy(Coord::new(1, 0), Coord::new(0, 0));
        assert_eq!(hops, vec![Direction::West]);
    }

    #[test]
    fn adjacent_north() {
        let hops = generate_route_xy(Coord::new(0, 0), Coord::new(0, 1));
        assert_eq!(hops, vec![Direction::North]);
    }

    #[test]
    fn adjacent_south() {
        let hops = generate_route_xy(Coord::new(0, 1), Coord::new(0, 0));
        assert_eq!(hops, vec![Direction::South]);
    }

    #[test]
    fn multi_hop_xy() {
        let hops = generate_route_xy(Coord::new(0, 0), Coord::new(3, 2));
        assert_eq!(
            hops,
            vec![
                Direction::East,
                Direction::East,
                Direction::East,
                Direction::North,
                Direction::North
            ]
        );
    }

    #[test]
    fn reverse_direction() {
        let hops = generate_route_xy(Coord::new(3, 2), Coord::new(0, 0));
        assert_eq!(
            hops,
            vec![
                Direction::West,
                Direction::West,
                Direction::West,
                Direction::South,
                Direction::South
            ]
        );
    }

    #[test]
    fn x_only() {
        let hops = generate_route_xy(Coord::new(0, 3), Coord::new(4, 3));
        assert_eq!(hops.len(), 4);
        assert!(hops.iter().all(|d| *d == Direction::East));
    }

    #[test]
    fn y_only() {
        let hops = generate_route_xy(Coord::new(2, 0), Coord::new(2, 3));
        assert_eq!(hops.len(), 3);
        assert!(hops.iter().all(|d| *d == Direction::North));
    }

    #[test]
    fn x_before_y() {
        // Verify dimension ordering: all X hops come before any Y hops
        let hops = generate_route_xy(Coord::new(0, 0), Coord::new(2, 3));
        let first_y = hops.iter().position(|d| *d == Direction::North).unwrap();
        let last_x = hops.iter().rposition(|d| *d == Direction::East).unwrap();
        assert!(last_x < first_y, "X hops must precede Y hops");
    }
}
