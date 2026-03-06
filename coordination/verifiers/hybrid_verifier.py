"""
Hybrid Verifier: Rule-Based Safety Checks + Learned Feasibility Classifier

Evaluates primitives before execution to reject unsafe or infeasible actions.
This is a core contribution of the paper — the verifier reduces unsafe LLM
proposals by an order of magnitude.

Components:
    1. Rule checker: deterministic safety constraints (always enforced)
    2. Learned classifier: MLP predicting P(success) from simulated rollouts
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of primitive verification."""
    accepted: bool
    rule_violations: List[str]
    feasibility_score: float
    rejection_reason: Optional[str] = None
    suggested_correction: Optional[dict] = None


class RuleChecker:
    """
    Deterministic safety rule checks.
    
    These are hard constraints that cannot be overridden:
    - No-fly zone intersection
    - Minimum inter-drone distance
    - Battery budget feasibility
    - Altitude limits
    - Arena boundary
    """

    def __init__(self, config: dict):
        self.min_drone_dist = config.get("min_inter_drone_dist", 5.0)
        self.min_altitude = config.get("min_altitude", 5.0)
        self.max_altitude = config.get("max_altitude", 120.0)  # FAA Part 107: 400ft ≈ 122m
        self.arena_bounds = config.get("arena_size", [500.0, 500.0])
        self.battery_reserve = config.get("battery_reserve", 0.15)  # 15% reserve for RTH

    def check(
        self,
        primitive: dict,
        drone_state: dict,
        other_drones: List[dict],
        no_fly_zones: List[dict],
    ) -> Tuple[bool, List[str]]:
        """
        Check all deterministic rules.

        Returns:
            (passed: bool, violations: list of violation descriptions)
        """
        violations = []

        # ---- Rule 1: Arena boundary ----
        if primitive["type"] == "navigate":
            target = primitive["params"].get("position", [0, 0, 30])
            if (target[0] < 0 or target[0] > self.arena_bounds[0] or
                target[1] < 0 or target[1] > self.arena_bounds[1]):
                violations.append(
                    f"OUT_OF_BOUNDS: target ({target[0]:.0f}, {target[1]:.0f}) "
                    f"outside arena {self.arena_bounds}"
                )

        # ---- Rule 2: Altitude limits ----
        if primitive["type"] in ["navigate", "inspect"]:
            alt = primitive["params"].get("position", [0, 0, 30])[2] \
                if primitive["type"] == "navigate" \
                else primitive["params"].get("altitude", 30)
            if alt < self.min_altitude:
                violations.append(f"ALTITUDE_LOW: {alt:.1f}m < min {self.min_altitude}m")
            if alt > self.max_altitude:
                violations.append(f"ALTITUDE_HIGH: {alt:.1f}m > max {self.max_altitude}m")

        # ---- Rule 3: No-fly zone intersection ----
        if primitive["type"] == "navigate":
            target = primitive["params"].get("position", [0, 0, 30])
            for nfz in no_fly_zones:
                center = np.array(nfz["center"])
                half_ext = np.array(nfz["half_extents"])
                if (abs(target[0] - center[0]) < half_ext[0] and
                    abs(target[1] - center[1]) < half_ext[1]):
                    violations.append(
                        f"NFZ_VIOLATION: target in no-fly zone {nfz['id']}"
                    )

            # Check path intersection at multiple points along trajectory
            current = np.array(drone_state["position"][:2])
            target_2d = np.array(target[:2])
            for t in [0.25, 0.5, 0.75]:
                check_pt = current + t * (target_2d - current)
                for nfz in no_fly_zones:
                    center = np.array(nfz["center"])
                    half_ext = np.array(nfz["half_extents"])
                    if (abs(check_pt[0] - center[0]) < half_ext[0] and
                        abs(check_pt[1] - center[1]) < half_ext[1]):
                        violations.append(
                            f"NFZ_PATH: path crosses no-fly zone {nfz['id']} at t={t}"
                        )
                        break  # one path violation is enough

        # ---- Rule 4: Battery feasibility ----
        if primitive["type"] == "navigate":
            target = np.array(primitive["params"].get("position", [0, 0, 30]))
            current = np.array(drone_state["position"])
            dist = np.linalg.norm(target - current)

            # Estimate battery cost: distance / speed * drain_rate
            speed = 15.0  # m/s
            steps_needed = dist / speed / (1.0 / 24.0)
            battery_cost = steps_needed * (1.0 / 14400)

            # Also need to return home
            home_dist = np.linalg.norm(target[:2] - np.array([
                drone_state["position"][0], 10.0  # approximate home
            ]))
            home_cost = (home_dist / speed / (1.0 / 24.0)) * (1.0 / 14400)

            total_needed = battery_cost + home_cost + self.battery_reserve
            if total_needed > drone_state["battery"]:
                violations.append(
                    f"BATTERY_INSUFFICIENT: need {total_needed:.3f}, "
                    f"have {drone_state['battery']:.3f}"
                )

        # ---- Rule 5: Inter-drone distance ----
        if primitive["type"] == "navigate":
            target = primitive["params"].get("position", [0, 0, 30])
            for other in other_drones:
                if other.get("active", True):
                    dist = np.linalg.norm(
                        np.array(target) - np.array(other["position"])
                    )
                    if dist < self.min_drone_dist:
                        violations.append(
                            f"COLLISION_RISK: {dist:.1f}m from drone {other['id']} "
                            f"(min: {self.min_drone_dist}m)"
                        )

        passed = len(violations) == 0
        return passed, violations


class LearnedFeasibilityClassifier:
    """
    MLP classifier predicting P(success) for a primitive.
    
    Trained on simulated rollout outcomes (Experiment C3).
    Input: primitive features + state features
    Output: success probability
    """

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.6):
        self.threshold = threshold
        self.model = None
        if model_path is None:
            model_path = "coordination/models/feas_mlp.joblib"

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            # Use a simple heuristic model until trained model is available
            self.model = None
            print("LearnedFeasibilityClassifier: using heuristic fallback")

    def _load_model(self, path: str):
        """Load trained sklearn model (joblib)."""
        import joblib
        self.model = joblib.load(path)
        print(f"Loaded feasibility classifier from {path}")

    def _featurize(self, primitive: dict, drone_state: dict) -> np.ndarray:
        """Convert primitive + state to feature vector."""
        # Primitive type one-hot
        prim_types = ["navigate", "inspect", "hover", "photograph", "return_home"]
        type_onehot = [1.0 if primitive["type"] == pt else 0.0 for pt in prim_types]

        # Navigation distance (if applicable)
        if primitive["type"] == "navigate":
            target = np.array(primitive["params"].get("position", [0, 0, 30]))
            current = np.array(drone_state["position"])
            nav_dist = np.linalg.norm(target - current)
        else:
            nav_dist = 0.0

        # State features
        battery = drone_state.get("battery", 1.0)
        altitude = drone_state["position"][2]
        velocity_mag = np.linalg.norm(drone_state.get("velocity", [0, 0, 0]))

        features = np.array(
            type_onehot +
            [nav_dist / 500.0, battery, altitude / 120.0, velocity_mag / 15.0]
        )

        return features

    def predict(self, primitive: dict, drone_state: dict) -> float:
        """Predict success probability."""
        features = self._featurize(primitive, drone_state)

        if self.model is not None:
            # Use trained model
            prob = self.model.predict_proba(features.reshape(1, -1))[0, 1]
        else:
            # Heuristic fallback
            prob = self._heuristic_predict(primitive, drone_state)

        return float(prob)

    def _heuristic_predict(self, primitive: dict, drone_state: dict) -> float:
        """Feasibility heuristic based on battery budget analysis."""
        battery = drone_state.get("battery", 1.0)

        if primitive["type"] == "navigate":
            target = np.array(primitive["params"].get("position", [0, 0, 30]))
            current = np.array(drone_state["position"])
            dist = np.linalg.norm(target - current)
            
            # Estimate battery needed for this action + return home
            speed = 15.0
            steps_needed = dist / speed / (1.0 / 24.0)
            battery_cost = steps_needed * (1.0 / 14400)
            
            # Rough return home cost
            home_dist = np.linalg.norm(target[:2])  # distance from origin
            home_cost = (home_dist / speed / (1.0 / 24.0)) * (1.0 / 14400)
            
            total_needed = battery_cost + home_cost + 0.15  # 15% reserve
            
            if total_needed > battery:
                return 0.2  # Will run out of battery
            
            margin = battery - total_needed
            if margin > 0.3:
                return 0.95
            elif margin > 0.1:
                return 0.8
            else:
                return 0.6

        if primitive["type"] == "inspect":
            if battery < 0.15:
                return 0.3
            return 0.9

        if battery < 0.1:
            return 0.2
        
        return 0.9


class HybridVerifier:
    """
    Combined verifier: rule checks + learned feasibility.

    Pipeline:
    1. Rule checks (hard constraints) → immediate reject if violated
    2. Learned feasibility → reject if P(success) < threshold
    3. If rejected, optionally suggest corrections

    This is a central contribution of the paper.
    """

    def __init__(
        self,
        config: dict,
        model_path: Optional[str] = None,
        feasibility_threshold: float = 0.6,
        use_learned: bool = True,
    ):
        self.rule_checker = RuleChecker(config)
        self.use_learned = use_learned
        if use_learned:
            self.classifier = LearnedFeasibilityClassifier(
                model_path=model_path,
                threshold=feasibility_threshold,
            )
        self.feasibility_threshold = feasibility_threshold

        # Statistics
        self.stats = {
            "total_checked": 0,
            "rule_rejected": 0,
            "feasibility_rejected": 0,
            "accepted": 0,
        }

    def verify(
        self,
        primitive: dict,
        drone_state: dict,
        other_drones: List[dict],
        no_fly_zones: List[dict],
    ) -> VerificationResult:
        """
        Verify a primitive action.

        Args:
            primitive: {"type": str, "params": dict, "assigned_drone": int}
            drone_state: current state of assigned drone
            other_drones: states of all other drones
            no_fly_zones: list of NFZ definitions

        Returns:
            VerificationResult with accept/reject decision
        """
        self.stats["total_checked"] += 1

        # Step 1: Rule checks
        rule_passed, violations = self.rule_checker.check(
            primitive, drone_state, other_drones, no_fly_zones
        )

        if not rule_passed:
            self.stats["rule_rejected"] += 1
            correction = self._suggest_correction(primitive, violations)
            return VerificationResult(
                accepted=False,
                rule_violations=violations,
                feasibility_score=0.0,
                rejection_reason=f"RULE_VIOLATION: {'; '.join(violations)}",
                suggested_correction=correction,
            )

        # Step 2: Learned feasibility
        if self.use_learned:
            score = self.classifier.predict(primitive, drone_state)
            if score < self.feasibility_threshold:
                self.stats["feasibility_rejected"] += 1
                return VerificationResult(
                    accepted=False,
                    rule_violations=[],
                    feasibility_score=score,
                    rejection_reason=f"LOW_FEASIBILITY: {score:.3f} < {self.feasibility_threshold}",
                )
        else:
            score = 1.0

        # Accepted
        self.stats["accepted"] += 1
        return VerificationResult(
            accepted=True,
            rule_violations=[],
            feasibility_score=score,
        )

    def _suggest_correction(
        self, primitive: dict, violations: List[str]
    ) -> Optional[dict]:
        """Suggest parameter corrections for common violations."""
        corrections = {}

        for v in violations:
            if "NFZ_VIOLATION" in v or "NFZ_PATH" in v:
                corrections["suggestion"] = "Reroute around no-fly zone"
            elif "BATTERY_INSUFFICIENT" in v:
                corrections["suggestion"] = "Return to home or reduce mission scope"
            elif "ALTITUDE" in v:
                corrections["suggestion"] = "Adjust altitude to within [5, 120] meters"
            elif "OUT_OF_BOUNDS" in v:
                corrections["suggestion"] = "Clamp target to arena boundaries"
            elif "COLLISION_RISK" in v:
                corrections["suggestion"] = "Add offset to avoid other drone"

        return corrections if corrections else None

    def get_stats(self) -> dict:
        """Return verification statistics."""
        return dict(self.stats)

    def reset_stats(self):
        """Reset statistics for new trial."""
        self.stats = {k: 0 for k in self.stats}


# =============================================================
# Quick Test
# =============================================================

if __name__ == "__main__":
    print("Testing Hybrid Verifier...\n")

    config = {
        "arena_size": [500.0, 500.0],
        "min_inter_drone_dist": 5.0,
        "min_altitude": 5.0,
        "max_altitude": 120.0,
        "battery_reserve": 0.15,
    }

    verifier = HybridVerifier(config, use_learned=True)

    drone_state = {
        "id": 0,
        "position": [100, 100, 30],
        "velocity": [0, 0, 0],
        "battery": 0.8,
        "active": True,
    }

    other_drones = [
        {"id": 1, "position": [200, 200, 30], "active": True},
        {"id": 2, "position": [300, 100, 30], "active": True},
    ]

    nfzs = [
        {"id": 0, "center": [250, 250], "half_extents": [30, 30]},
    ]

    # Test 1: Valid navigate
    prim1 = {"type": "navigate", "params": {"position": [150, 150, 30]}, "assigned_drone": 0}
    r1 = verifier.verify(prim1, drone_state, other_drones, nfzs)
    print(f"Test 1 (valid navigate):   accepted={r1.accepted}, score={r1.feasibility_score:.2f}")

    # Test 2: Navigate into NFZ
    prim2 = {"type": "navigate", "params": {"position": [250, 250, 30]}, "assigned_drone": 0}
    r2 = verifier.verify(prim2, drone_state, other_drones, nfzs)
    print(f"Test 2 (NFZ violation):    accepted={r2.accepted}, reason={r2.rejection_reason}")

    # Test 3: Navigate out of bounds
    prim3 = {"type": "navigate", "params": {"position": [600, 300, 30]}, "assigned_drone": 0}
    r3 = verifier.verify(prim3, drone_state, other_drones, nfzs)
    print(f"Test 3 (out of bounds):    accepted={r3.accepted}, reason={r3.rejection_reason}")

    # Test 4: Low battery
    low_batt_drone = {**drone_state, "battery": 0.05}
    prim4 = {"type": "navigate", "params": {"position": [400, 400, 30]}, "assigned_drone": 0}
    r4 = verifier.verify(prim4, low_batt_drone, other_drones, nfzs)
    print(f"Test 4 (low battery):      accepted={r4.accepted}, reason={r4.rejection_reason}")

    # Test 5: Too close to another drone
    prim5 = {"type": "navigate", "params": {"position": [201, 201, 30]}, "assigned_drone": 0}
    r5 = verifier.verify(prim5, drone_state, other_drones, nfzs)
    print(f"Test 5 (collision risk):   accepted={r5.accepted}, reason={r5.rejection_reason}")

    print(f"\nVerifier stats: {verifier.get_stats()}")
