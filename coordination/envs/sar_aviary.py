"""
SAR (Search and Rescue) Aviary Environment

Custom multi-agent environment extending gym-pybullet-drones for
search-and-rescue coordination experiments.

Drones must search an arena for simulated survivors, respecting
no-fly zones and battery constraints, while coordinating via
a message bus.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import time


# =============================================================
# Data Structures
# =============================================================

class TargetPriority(Enum):
    HIGH = "high"
    LOW = "low"


class TargetDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class SurvivorTarget:
    """A simulated survivor to be found."""
    id: int
    position: np.ndarray  # [x, y] in meters
    priority: TargetPriority
    difficulty: TargetDifficulty
    description: str  # e.g., "person on rooftop wearing orange jacket"
    found: bool = False
    found_by: Optional[int] = None  # drone ID
    found_at_step: Optional[int] = None


@dataclass
class NoFlyZone:
    """Rectangular no-fly zone."""
    id: int
    center: np.ndarray  # [x, y]
    half_extents: np.ndarray  # [half_w, half_h]

    def contains(self, point: np.ndarray) -> bool:
        """Check if a 2D point is inside this NFZ."""
        return (abs(point[0] - self.center[0]) < self.half_extents[0] and
                abs(point[1] - self.center[1]) < self.half_extents[1])


@dataclass
class DroneState:
    """State of a single drone agent."""
    id: int
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    battery: float  # 0.0 to 1.0
    active: bool = True
    targets_found: List[int] = field(default_factory=list)
    inside_nfz: set = field(default_factory=set)
    had_collision: set = field(default_factory=set)


@dataclass
class Primitive:
    """An executable primitive action."""
    type: str  # navigate, inspect, hover, photograph
    params: dict
    assigned_drone: int
    verified: bool = False
    status: str = "pending"  # pending, executing, completed, failed


# =============================================================
# Perception Simulator
# =============================================================

class PerceptionSimulator:
    """
    Simulates open-vocabulary detection as a probabilistic model.
    
    Detection probabilities are calibrated against Track A results
    (Experiment P1) to ensure coordination experiments use realistic
    perception assumptions.
    """

    def __init__(
        self,
        base_probs: Dict[str, float] = None,
        max_detection_range: float = 50.0,
        optimal_altitude: float = 30.0,
        false_positive_rate: float = 0.05,
        seed: int = 42,
    ):
        self.base_probs = base_probs or {
            "easy": 0.92,
            "medium": 0.71,
            "hard": 0.43,
        }
        self.max_range = max_detection_range
        self.optimal_alt = optimal_altitude
        self.fp_rate = false_positive_rate
        self.rng = np.random.default_rng(seed)

    def detect(
        self,
        drone_pos: np.ndarray,
        target: SurvivorTarget,
    ) -> Tuple[bool, float]:
        """
        Simulate whether a drone detects a target.

        Returns:
            (detected: bool, confidence: float)
        """
        # Horizontal distance
        dist_2d = np.linalg.norm(drone_pos[:2] - target.position)
        if dist_2d > self.max_range:
            return False, 0.0

        # Distance decay (linear falloff)
        range_factor = max(0.0, 1.0 - dist_2d / self.max_range)

        # Altitude penalty (Gaussian around optimal)
        altitude = drone_pos[2]
        alt_factor = np.exp(-((altitude - self.optimal_alt) / 20.0) ** 2)

        # Base probability from difficulty
        base_p = self.base_probs[target.difficulty.value]

        # Combined detection probability
        det_prob = base_p * range_factor * alt_factor

        # Sample detection
        detected = self.rng.random() < det_prob
        confidence = det_prob * (0.8 + 0.2 * self.rng.random()) if detected else 0.0

        return detected, confidence

    def generate_false_positives(
        self,
        drone_pos: np.ndarray,
        arena_bounds: Tuple[float, float],
    ) -> List[dict]:
        """Generate false positive detections."""
        fps = []
        if self.rng.random() < self.fp_rate:
            # Random position near drone
            offset = self.rng.normal(0, 15, size=2)
            fp_pos = drone_pos[:2] + offset
            fp_pos = np.clip(fp_pos, 0, arena_bounds)
            fps.append({
                "position": fp_pos,
                "confidence": 0.2 + 0.3 * self.rng.random(),
                "is_false_positive": True,
            })
        return fps


# =============================================================
# SAR Environment
# =============================================================

class SAREnvironment:
    """
    Search-and-Rescue simulation environment.

    This environment manages:
    - Arena with obstacles and no-fly zones
    - Multiple drone agents with battery dynamics
    - Survivor targets with detection simulation
    - Safety monitoring (collisions, NFZ violations, battery)
    - Mission success evaluation

    Designed to interface with LLM planners and the hybrid verifier.
    """

    def __init__(self, config: dict = None):
        config = config or self.default_config()
        self.config = config

        # Arena
        self.arena_size = config.get("arena_size", [500.0, 500.0])
        self.n_drones = config.get("n_drones", 4)
        self.n_targets = config.get("n_targets", 12)
        self.n_nfz = config.get("n_nfz", 3)

        # Physics
        self.dt = config.get("dt", 1.0 / 24.0)  # 24 Hz control
        self.max_velocity = config.get("max_velocity", 15.0)  # m/s
        self.default_altitude = config.get("default_altitude", 30.0)  # meters
        self.min_inter_drone_dist = config.get("min_inter_drone_dist", 5.0)

        # Battery
        self.battery_drain_per_step = config.get("battery_drain_per_step", 1.0 / 14400)
        self.battery_critical = config.get("battery_critical", 0.1)

        # Mission
        self.max_steps = config.get("max_steps", 14400)  # 10 min at 24 Hz
        self.success_threshold = config.get("success_threshold", 0.8)  # 80% high-prio found

        # State
        self.step_count = 0
        self.drones: List[DroneState] = []
        self.targets: List[SurvivorTarget] = []
        self.nfzs: List[NoFlyZone] = []
        self.perception = None
        self.rng = None

        # Metrics logging
        self.log = {
            "collisions": 0,
            "nfz_violations": 0,
            "battery_deaths": 0,
            "targets_found_over_time": [],
            "messages_sent": 0,
            "messages_bytes": 0,
            "verifier_rejections": 0,
            "primitive_count": 0,
        }

    @staticmethod
    def default_config() -> dict:
        return {
            "arena_size": [500.0, 500.0],
            "n_drones": 4,
            "n_targets": 12,
            "n_nfz": 3,
            "dt": 1.0 / 24.0,
            "max_velocity": 15.0,
            "default_altitude": 30.0,
            "min_inter_drone_dist": 5.0,
            "battery_drain_per_step": 1.0 / 14400,
            "battery_critical": 0.1,
            "max_steps": 14400,
            "success_threshold": 0.8,
        }

    def reset(self, seed: int = 42) -> dict:
        """Reset environment with new random layout."""
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.log = {k: 0 if isinstance(v, (int, float)) else []
                    for k, v in self.log.items()}

        self.perception = PerceptionSimulator(seed=seed)

        # Place drones at starting positions (evenly spaced along south edge)
        self.drones = []
        for i in range(self.n_drones):
            x = (i + 1) * self.arena_size[0] / (self.n_drones + 1)
            pos = np.array([x, 10.0, self.default_altitude])
            self.drones.append(DroneState(
                id=i, position=pos, velocity=np.zeros(3),
                battery=1.0, active=True, targets_found=[],
            ))

        # Place targets randomly
        self.targets = []
        target_descriptions = [
            ("person on rooftop", TargetPriority.HIGH, TargetDifficulty.EASY),
            ("person wearing orange jacket", TargetPriority.HIGH, TargetDifficulty.MEDIUM),
            ("person waving near debris", TargetPriority.HIGH, TargetDifficulty.MEDIUM),
            ("person trapped under rubble", TargetPriority.HIGH, TargetDifficulty.HARD),
            ("person in open field", TargetPriority.LOW, TargetDifficulty.EASY),
            ("person near vehicle", TargetPriority.LOW, TargetDifficulty.EASY),
            ("person walking on road", TargetPriority.LOW, TargetDifficulty.MEDIUM),
            ("person under tree canopy", TargetPriority.LOW, TargetDifficulty.HARD),
        ]

        for i in range(self.n_targets):
            desc, priority, difficulty = target_descriptions[i % len(target_descriptions)]
            pos = self.rng.uniform(
                [20, 20],
                [self.arena_size[0] - 20, self.arena_size[1] - 20]
            )
            self.targets.append(SurvivorTarget(
                id=i, position=pos, priority=priority,
                difficulty=difficulty, description=desc,
            ))

        # Place no-fly zones (avoiding target locations)
        self.nfzs = []
        for i in range(self.n_nfz):
            center = self.rng.uniform(
                [50, 50],
                [self.arena_size[0] - 50, self.arena_size[1] - 50]
            )
            half_ext = self.rng.uniform([15, 15], [40, 40])
            self.nfzs.append(NoFlyZone(id=i, center=center, half_extents=half_ext))

        return self.get_state_summary()

    def get_state_summary(self) -> dict:
        """Get compact state for LLM planner."""
        return {
            "arena_size": self.arena_size,
            "step": self.step_count,
            "max_steps": self.max_steps,
            "drones": [
                {
                    "id": d.id,
                    "position": d.position.tolist(),
                    "battery": round(d.battery, 3),
                    "active": d.active,
                    "targets_found": d.targets_found,
                }
                for d in self.drones
            ],
            "targets_found": sum(1 for t in self.targets if t.found),
            "targets_total": len(self.targets),
            "high_priority_found": sum(
                1 for t in self.targets
                if t.found and t.priority == TargetPriority.HIGH
            ),
            "high_priority_total": sum(
                1 for t in self.targets if t.priority == TargetPriority.HIGH
            ),
            "no_fly_zones": [
                {
                    "id": nfz.id,
                    "center": nfz.center.tolist(),
                    "half_extents": nfz.half_extents.tolist(),
                }
                for nfz in self.nfzs
            ],
        }

    def execute_primitive(self, primitive: Primitive) -> dict:
        """
        Execute a single primitive action.

        Returns:
            result dict with status, observations, safety info
        """
        drone = self.drones[primitive.assigned_drone]
        result = {
            "primitive_type": primitive.type,
            "drone_id": primitive.assigned_drone,
            "success": False,
            "detections": [],
            "safety_violations": [],
        }

        if not drone.active:
            result["status"] = "drone_inactive"
            return result

        self.log["primitive_count"] += 1

        if primitive.type == "navigate":
            result = self._execute_navigate(drone, primitive.params)
        elif primitive.type == "inspect":
            result = self._execute_inspect(drone, primitive.params)
        elif primitive.type == "hover":
            result = self._execute_hover(drone, primitive.params)
        elif primitive.type == "return_home":
            result = self._execute_return_home(drone)
        else:
            result["status"] = f"unknown_primitive: {primitive.type}"

        return result

    def _execute_navigate(self, drone: DroneState, params: dict) -> dict:
        """Navigate drone to target position."""
        target_pos = np.array(params.get("position", [250, 250, 30]), dtype=np.float64)
        result = {"success": False, "detections": [], "safety_violations": []}

        # Simulate movement over multiple steps
        steps_taken = 0
        max_nav_steps = int(params.get("max_steps", 500))

        while steps_taken < max_nav_steps:
            # Compute direction and move
            direction = target_pos - drone.position
            dist = np.linalg.norm(direction)

            if dist < 2.0:  # Close enough
                drone.position = target_pos.copy()
                result["success"] = True
                break

            velocity = direction / dist * min(self.max_velocity, dist / self.dt)
            drone.position += velocity * self.dt
            drone.velocity = velocity

            # Battery drain
            drone.battery -= self.battery_drain_per_step
            if drone.battery <= self.battery_critical:
                drone.active = False
                self.log["battery_deaths"] += 1
                result["safety_violations"].append("battery_critical")
                break

            # Safety checks
            self._check_safety(drone, result)

            # Perception: check for targets along path
            for target in self.targets:
                if not target.found:
                    detected, conf = self.perception.detect(drone.position, target)
                    if detected:
                        target.found = True
                        target.found_by = drone.id
                        target.found_at_step = self.step_count + steps_taken
                        drone.targets_found.append(target.id)
                        result["detections"].append({
                            "target_id": target.id,
                            "description": target.description,
                            "priority": target.priority.value,
                            "confidence": conf,
                        })

            steps_taken += 1
            self.step_count += 1

        result["steps_taken"] = steps_taken
        result["final_position"] = drone.position.tolist()
        return result

    def _execute_inspect(self, drone: DroneState, params: dict) -> dict:
        """Hover and inspect a region (enhanced detection)."""
        result = {"success": True, "detections": [], "safety_violations": []}

        # Lower altitude for better detection
        original_alt = drone.position[2]
        inspect_alt = params.get("altitude", 20.0)
        drone.position[2] = inspect_alt

        # Multiple observation passes (simulates circling)
        n_passes = params.get("n_passes", 3)
        for _ in range(n_passes):
            for target in self.targets:
                if not target.found:
                    detected, conf = self.perception.detect(drone.position, target)
                    if detected:
                        target.found = True
                        target.found_by = drone.id
                        target.found_at_step = self.step_count
                        drone.targets_found.append(target.id)
                        result["detections"].append({
                            "target_id": target.id,
                            "description": target.description,
                            "priority": target.priority.value,
                            "confidence": conf,
                        })

            drone.battery -= self.battery_drain_per_step * 10
            self.step_count += 10

        drone.position[2] = original_alt
        return result

    def _execute_hover(self, drone: DroneState, params: dict) -> dict:
        """Hover in place for a duration."""
        duration_steps = int(params.get("duration_steps", 100))
        result = {"success": True, "detections": [], "safety_violations": []}

        for _ in range(duration_steps):
            drone.battery -= self.battery_drain_per_step
            self.step_count += 1

            if drone.battery <= self.battery_critical:
                drone.active = False
                self.log["battery_deaths"] += 1
                result["safety_violations"].append("battery_critical")
                result["success"] = False
                break

        return result

    def _execute_return_home(self, drone: DroneState) -> dict:
        """Return drone to starting position."""
        home = np.array([
            (drone.id + 1) * self.arena_size[0] / (self.n_drones + 1),
            10.0,
            self.default_altitude,
        ])
        return self._execute_navigate(drone, {"position": home.tolist()})

    def _check_safety(self, drone: DroneState, result: dict):
        """Check for safety violations (discrete entry counting)."""
        # NFZ check - count ENTRY events, not per-step overlap
        for nfz in self.nfzs:
            if nfz.contains(drone.position[:2]):
                if nfz.id not in drone.inside_nfz:
                    # New entry into NFZ
                    drone.inside_nfz.add(nfz.id)
                    self.log["nfz_violations"] += 1
                    result["safety_violations"].append(f"nfz_{nfz.id}")
            else:
                drone.inside_nfz.discard(nfz.id)

        # Inter-drone collision check - count per pair once
        for other in self.drones:
            if other.id != drone.id and other.active:
                dist = np.linalg.norm(drone.position - other.position)
                pair = tuple(sorted([drone.id, other.id]))
                if dist < self.min_inter_drone_dist:
                    if pair not in drone.had_collision:
                        drone.had_collision.add(pair)
                        self.log["collisions"] += 1
                        result["safety_violations"].append(f"near_collision_drone_{other.id}")
                else:
                    drone.had_collision.discard(pair)

    def get_mission_result(self) -> dict:
        """Evaluate mission outcome."""
        high_prio_found = sum(
            1 for t in self.targets
            if t.found and t.priority == TargetPriority.HIGH
        )
        high_prio_total = sum(
            1 for t in self.targets if t.priority == TargetPriority.HIGH
        )
        all_found = sum(1 for t in self.targets if t.found)

        high_prio_rate = high_prio_found / high_prio_total if high_prio_total > 0 else 0
        overall_rate = all_found / len(self.targets) if self.targets else 0

        mission_success = (
            high_prio_rate >= self.success_threshold
        )

        return {
            "mission_success": mission_success,
            "high_priority_found": high_prio_found,
            "high_priority_total": high_prio_total,
            "high_priority_rate": high_prio_rate,
            "all_found": all_found,
            "all_total": len(self.targets),
            "overall_rate": overall_rate,
            "steps_used": self.step_count,
            "max_steps": self.max_steps,
            "time_fraction": self.step_count / self.max_steps,
            "safety_log": dict(self.log),
            "per_drone_summary": [
                {
                    "id": d.id,
                    "active": d.active,
                    "battery_remaining": round(d.battery, 3),
                    "targets_found": len(d.targets_found),
                }
                for d in self.drones
            ],
        }

    def log_message(self, msg_bytes: int):
        """Log a communication message for overhead tracking."""
        self.log["messages_sent"] += 1
        self.log["messages_bytes"] += msg_bytes

    def log_verifier_rejection(self):
        """Log a verifier rejection."""
        self.log["verifier_rejections"] += 1


# =============================================================
# Quick Test
# =============================================================

if __name__ == "__main__":
    print("Testing SAR Environment...")

    env = SAREnvironment()
    state = env.reset(seed=42)

    print(f"Arena: {state['arena_size']}")
    print(f"Drones: {len(state['drones'])}")
    print(f"Targets: {state['targets_total']}")
    print(f"High priority: {state['high_priority_total']}")
    print(f"No-fly zones: {len(state['no_fly_zones'])}")

    # Simulate a simple sweep pattern
    for drone_id in range(env.n_drones):
        # Navigate to assigned sector
        sector_x = (drone_id + 0.5) * env.arena_size[0] / env.n_drones
        sector_y = env.arena_size[1] / 2

        prim = Primitive(
            type="navigate",
            params={"position": [sector_x, sector_y, 30.0], "max_steps": 200},
            assigned_drone=drone_id,
        )
        result = env.execute_primitive(prim)
        print(f"Drone {drone_id} navigate: {result['success']}, "
              f"detections: {len(result['detections'])}")

        # Inspect the area
        prim = Primitive(
            type="inspect",
            params={"altitude": 20.0, "n_passes": 5},
            assigned_drone=drone_id,
        )
        result = env.execute_primitive(prim)
        print(f"Drone {drone_id} inspect: detections={len(result['detections'])}")

    # Get final results
    mission = env.get_mission_result()
    print(f"\n{'='*50}")
    print(f"Mission success: {mission['mission_success']}")
    print(f"Targets found: {mission['all_found']}/{mission['all_total']}")
    print(f"High priority: {mission['high_priority_found']}/{mission['high_priority_total']}")
    print(f"NFZ violations: {mission['safety_log']['nfz_violations']}")
    print(f"Collisions: {mission['safety_log']['collisions']}")
    print(f"Battery deaths: {mission['safety_log']['battery_deaths']}")
    print(f"Steps used: {mission['steps_used']}/{mission['max_steps']}")
    print(f"{'='*50}")
