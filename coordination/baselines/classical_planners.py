"""
Classical Multi-Robot Planning Baselines

Implements Voronoi coverage and frontier exploration for multi-UAV SAR.
These are the "proper baselines" the reviewer requested.
"""

import numpy as np
from scipy.spatial import Voronoi
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


def point_in_nfz(point, nfzs, buffer=10.0):
    """Check if a 2D point is inside any NFZ (with buffer)."""
    for nfz in (nfzs or []):
        c = nfz["center"] if isinstance(nfz["center"], (list, tuple)) else nfz["center"]
        he = nfz["half_extents"] if isinstance(nfz["half_extents"], (list, tuple)) else nfz["half_extents"]
        if (abs(point[0] - c[0]) < he[0] + buffer and
            abs(point[1] - c[1]) < he[1] + buffer):
            return True
    return False


@dataclass
class Waypoint:
    position: np.ndarray  # [x, y, z]
    action: str           # navigate, inspect, return_home
    params: dict


class VoronoiCoveragePlanner:
    """
    Voronoi-based area partitioning with lawnmower sweep.
    
    Standard multi-UAV SAR baseline from Cortes et al. (2004).
    Each drone is assigned a Voronoi cell based on starting positions,
    then executes a lawnmower pattern within its cell.
    
    Reference: Cortes et al., "Coverage control for mobile sensing 
    networks," IEEE T-RA, 2004.
    """

    def __init__(self, arena_size, sweep_spacing=40.0, altitude=30.0,
                 inspect_altitude=20.0, inspect_passes=3):
        self.arena_w, self.arena_h = arena_size
        self.sweep_spacing = sweep_spacing
        self.altitude = altitude
        self.inspect_altitude = inspect_altitude
        self.inspect_passes = inspect_passes

    def generate_plan(self, drone_positions: List[np.ndarray],
                      nfzs: List[dict] = None) -> Dict[int, List[Waypoint]]:
        """
        Generate Voronoi-partitioned sweep plan.
        
        Args:
            drone_positions: list of [x, y, z] for each drone
            nfzs: list of {"center": [x,y], "half_extents": [w,h]}
            
        Returns:
            dict mapping drone_id -> list of Waypoint
        """
        n_drones = len(drone_positions)
        nfzs = nfzs or []

        # Compute Voronoi partition using drone starting positions (2D)
        generators = np.array([p[:2] for p in drone_positions])

        # Create grid of sample points to assign to nearest drone
        grid_res = 10.0  # meters
        xs = np.arange(0, self.arena_w, grid_res)
        ys = np.arange(0, self.arena_h, grid_res)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        # Assign each grid point to nearest drone
        cell_assignments = {}
        for i in range(n_drones):
            cell_assignments[i] = []

        for pt in grid_points:
            # Skip if in NFZ
            in_nfz = False
            for nfz in nfzs:
                c = np.array(nfz["center"])
                he = np.array(nfz["half_extents"])
                if abs(pt[0] - c[0]) < he[0] and abs(pt[1] - c[1]) < he[1]:
                    in_nfz = True
                    break
            if in_nfz:
                continue

            dists = np.linalg.norm(generators - pt, axis=1)
            nearest = np.argmin(dists)
            cell_assignments[nearest].append(pt)

        # Generate lawnmower pattern for each cell
        plans = {}
        for drone_id in range(n_drones):
            points = cell_assignments[drone_id]
            if not points:
                plans[drone_id] = [
                    Waypoint(drone_positions[drone_id].copy(),
                             "return_home", {})
                ]
                continue

            points = np.array(points)
            waypoints = self._lawnmower_sweep(points, drone_id)
            plans[drone_id] = waypoints

        return plans

    def _lawnmower_sweep(self, cell_points: np.ndarray,
                         drone_id: int) -> List[Waypoint]:
        """Generate boustrophedon (lawnmower) pattern through a Voronoi cell."""
        x_min, y_min = cell_points.min(axis=0)
        x_max, y_max = cell_points.max(axis=0)

        waypoints = []
        y_positions = np.arange(y_min, y_max + self.sweep_spacing,
                                self.sweep_spacing)

        for i, y in enumerate(y_positions):
            if i % 2 == 0:
                # Left to right
                x_start, x_end = x_min, x_max
            else:
                # Right to left
                x_start, x_end = x_max, x_min

            # Navigate to start of sweep line
            waypoints.append(Waypoint(
                position=np.array([x_start, y, self.altitude]),
                action="navigate",
                params={"max_steps": 500},
            ))

            # Navigate to end of sweep line
            waypoints.append(Waypoint(
                position=np.array([x_end, y, self.altitude]),
                action="navigate",
                params={"max_steps": 500},
            ))

            # Inspect at midpoint
            mid_x = (x_start + x_end) / 2
            waypoints.append(Waypoint(
                position=np.array([mid_x, y, self.inspect_altitude]),
                action="inspect",
                params={
                    "altitude": self.inspect_altitude,
                    "n_passes": self.inspect_passes,
                },
            ))

        # Return home
        waypoints.append(Waypoint(
            position=np.array([0, 0, self.altitude]),  # placeholder
            action="return_home",
            params={},
        ))

        return waypoints


class FrontierExplorationPlanner:
    """
    Information-gain frontier exploration adapted for multi-agent aerial SAR.
    
    Each agent maintains a belief map of unexplored areas and greedily
    selects the nearest high-information frontier.
    
    Reference: Yamauchi, "A frontier-based approach for autonomous 
    exploration," CIRA 1997.
    """

    def __init__(self, arena_size, cell_size=20.0, altitude=30.0,
                 sensor_range=50.0, inspect_altitude=20.0):
        self.arena_w, self.arena_h = arena_size
        self.cell_size = cell_size
        self.altitude = altitude
        self.sensor_range = sensor_range
        self.inspect_altitude = inspect_altitude

        # Discretize arena into exploration grid
        self.nx = int(self.arena_w / cell_size)
        self.ny = int(self.arena_h / cell_size)
        self.explored = None  # reset per trial

    def reset(self, nfzs: List[dict] = None):
        """Reset exploration state."""
        self.explored = np.zeros((self.nx, self.ny), dtype=bool)
        # Mark NFZs as explored (don't visit)
        for nfz in (nfzs or []):
            c = nfz["center"]
            he = nfz["half_extents"]
            for ix in range(self.nx):
                for iy in range(self.ny):
                    cx = (ix + 0.5) * self.cell_size
                    cy = (iy + 0.5) * self.cell_size
                    if (abs(cx - c[0]) < he[0] + self.cell_size and
                        abs(cy - c[1]) < he[1] + self.cell_size):
                        self.explored[ix, iy] = True

    def get_next_waypoint(self, drone_pos: np.ndarray,
                          other_targets: List[np.ndarray] = None
                          ) -> Optional[Waypoint]:
        """
        Get next frontier waypoint for a drone.
        
        Selects the nearest unexplored cell, with penalty for cells
        that other drones are already heading toward.
        """
        if self.explored is None:
            raise RuntimeError("Call reset() first")

        other_targets = other_targets or []

        # Mark cells near current position as explored
        ix = int(np.clip(drone_pos[0] / self.cell_size, 0, self.nx - 1))
        iy = int(np.clip(drone_pos[1] / self.cell_size, 0, self.ny - 1))
        r = int(self.sensor_range / self.cell_size)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nix, niy = ix + dx, iy + dy
                if (0 <= nix < self.nx and 0 <= niy < self.ny):
                    dist = np.sqrt(dx**2 + dy**2) * self.cell_size
                    if dist <= self.sensor_range:
                        self.explored[nix, niy] = True

        # Find unexplored frontier cells
        frontiers = []
        for fx in range(self.nx):
            for fy in range(self.ny):
                if not self.explored[fx, fy]:
                    cell_center = np.array([
                        (fx + 0.5) * self.cell_size,
                        (fy + 0.5) * self.cell_size,
                    ])
                    dist = np.linalg.norm(drone_pos[:2] - cell_center)

                    # Penalty for cells other drones are targeting
                    peer_penalty = 0
                    for ot in other_targets:
                        if np.linalg.norm(ot[:2] - cell_center) < self.sensor_range:
                            peer_penalty += 50.0  # discourage overlap

                    cost = dist + peer_penalty
                    frontiers.append((cost, cell_center))

        if not frontiers:
            return None  # fully explored

        frontiers.sort(key=lambda x: x[0])
        target_2d = frontiers[0][1]

        return Waypoint(
            position=np.array([target_2d[0], target_2d[1], self.altitude]),
            action="navigate",
            params={"max_steps": 500},
        )

    def get_exploration_fraction(self) -> float:
        """Fraction of explorable cells that have been explored."""
        if self.explored is None:
            return 0.0
        return float(self.explored.sum()) / self.explored.size


class CBFShield:
    """
    Discrete-time Control Barrier Function shield for waypoint modification.
    
    Modifies LLM-generated waypoints to enforce:
    - Minimum separation distance between drones
    - No-fly zone avoidance
    
    This represents the "formal safety methods" baseline requested by reviewer.
    Note: operates on waypoints, not continuous control — a simplification
    appropriate for the high-level planning context.
    
    Reference: Wang et al., "Safety barrier certificates for 
    collision-free multirobot systems," IEEE T-RO, 2017.
    """

    def __init__(self, config: dict):
        self.d_min = config.get("min_inter_drone_dist", 5.0)
        self.arena = config.get("arena_size", [500.0, 500.0])
        self.gamma = 0.3  # CBF decay rate

    def modify_waypoint(self, target: np.ndarray,
                        drone_pos: np.ndarray,
                        other_drones: List[np.ndarray],
                        nfzs: List[dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Apply CBF-inspired corrections to a target waypoint.
        
        Returns:
            (modified_target, list of modifications applied)
        """
        modified = target.copy().astype(float)
        mods = []

        # --- Separation barrier ---
        for i, other in enumerate(other_drones):
            diff = modified[:2] - other[:2]
            dist = np.linalg.norm(diff)
            if dist < self.d_min * 2:  # within barrier region
                # Push away: barrier function h(x) = ||x_i - x_j||^2 - d_min^2
                # Gradient direction: away from other drone
                if dist > 0.01:
                    push_dir = diff / dist
                else:
                    push_dir = np.random.randn(2)
                    push_dir /= np.linalg.norm(push_dir)
                push_mag = (self.d_min * 2 - dist) * (1 + self.gamma)
                modified[:2] += push_dir * push_mag
                mods.append(f"separation_push_drone_{i}")

        # --- NFZ barrier ---
        for nfz in nfzs:
            c = np.array(nfz["center"])
            he = np.array(nfz["half_extents"]) + self.d_min  # buffer
            dx = modified[0] - c[0]
            dy = modified[1] - c[1]
            if abs(dx) < he[0] and abs(dy) < he[1]:
                # Inside barrier — push to nearest edge
                push_x = np.sign(dx) * (he[0] - abs(dx) + 5.0)
                push_y = np.sign(dy) * (he[1] - abs(dy) + 5.0)
                if abs(push_x) < abs(push_y):
                    modified[0] += push_x
                else:
                    modified[1] += push_y
                mods.append(f"nfz_push_{nfz.get('id', '?')}")

        # --- Arena boundary ---
        modified[0] = np.clip(modified[0], 5, self.arena[0] - 5)
        modified[1] = np.clip(modified[1], 5, self.arena[1] - 5)
        if not np.allclose(modified[:2], target[:2]):
            if "boundary_clip" not in [m.split("_")[0] for m in mods]:
                mods.append("boundary_clip")

        return modified, mods

    def verify(self, target: np.ndarray, drone_pos: np.ndarray,
               other_drones: List[np.ndarray], nfzs: List[dict]) -> bool:
        """Check if waypoint satisfies all CBF constraints without modification."""
        mod, changes = self.modify_waypoint(target, drone_pos, other_drones, nfzs)
        return len(changes) == 0


# ================================================================
# Quick Tests
# ================================================================

if __name__ == "__main__":
    print("Testing classical baselines...\n")

    arena = [500.0, 500.0]
    drone_pos = [
        np.array([100, 10, 30]),
        np.array([200, 10, 30]),
        np.array([300, 10, 30]),
        np.array([400, 10, 30]),
    ]
    nfzs = [{"center": [250, 250], "half_extents": [30, 30], "id": 0}]

    # --- Voronoi ---
    print("Voronoi Coverage Planner:")
    vor = VoronoiCoveragePlanner(arena)
    plans = vor.generate_plan(drone_pos, nfzs)
    for did, wps in plans.items():
        print(f"  Drone {did}: {len(wps)} waypoints")
        for wp in wps[:3]:
            print(f"    {wp.action} -> [{wp.position[0]:.0f}, {wp.position[1]:.0f}, {wp.position[2]:.0f}]")
        if len(wps) > 3:
            print(f"    ... ({len(wps)-3} more)")

    # --- Frontier ---
    print("\nFrontier Exploration:")
    fe = FrontierExplorationPlanner(arena)
    fe.reset(nfzs)
    for i in range(3):
        wp = fe.get_next_waypoint(drone_pos[0])
        if wp:
            print(f"  Step {i}: {wp.action} -> [{wp.position[0]:.0f}, {wp.position[1]:.0f}]")
    print(f"  Explored: {fe.get_exploration_fraction():.1%}")

    # --- CBF ---
    print("\nCBF Shield:")
    cbf = CBFShield({"min_inter_drone_dist": 5.0, "arena_size": arena})
    # Test: target near another drone
    target = np.array([201, 11, 30])
    mod, changes = cbf.modify_waypoint(target, drone_pos[0],
                                        [p for p in drone_pos[1:]], nfzs)
    print(f"  Target: [{target[0]:.0f}, {target[1]:.0f}]")
    print(f"  Modified: [{mod[0]:.0f}, {mod[1]:.0f}]")
    print(f"  Changes: {changes}")

    # Test: target in NFZ
    target_nfz = np.array([250, 250, 30])
    mod2, changes2 = cbf.modify_waypoint(target_nfz, drone_pos[0],
                                          [p for p in drone_pos[1:]], nfzs)
    print(f"\n  NFZ target: [{target_nfz[0]:.0f}, {target_nfz[1]:.0f}]")
    print(f"  Modified: [{mod2[0]:.0f}, {mod2[1]:.0f}]")
    print(f"  Changes: {changes2}")

    print("\n✓ All baselines operational")
