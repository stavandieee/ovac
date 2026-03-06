import os, random
import numpy as np

from coordination.envs.sar_aviary import SAREnvironment
from coordination.verifiers.hybrid_verifier import LearnedFeasibilityClassifier, HybridVerifier

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def main(out_path="data/feas_dataset.npz", n_episodes=2000, steps_per_ep=50, seed=0):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    set_seed(seed)

    env = SAREnvironment()
    clf = LearnedFeasibilityClassifier(model_path=None, threshold=0.6)  # heuristic fallback for scoring
    X, y = [], []

    for ep in range(n_episodes):
        state = env.reset(seed=seed + ep)
        # Build verifier config (keys expected by HybridVerifier)
        config = {
            "min_inter_drone_dist": 5.0,
            "min_altitude": 5.0,
            "max_altitude": 120.0,
            "arena_size": state.get("arena_size", [500.0, 500.0]),
            "battery_reserve": 0.15,
        }
        verifier = HybridVerifier(config=config, model_path=None, feasibility_threshold=0.6, use_learned=True)

        for t in range(steps_per_ep):
            # Candidate action (adjust to match your verifier's expected schema)
            cand = {
                "type": "navigate",
                "drone_id": int(np.random.randint(0, getattr(env, "n_drones", 4))),
                "params": {
                    "position": [float(np.random.uniform(0, 500)), float(np.random.uniform(0, 500)), 30.0],
                    "speed": float(np.random.uniform(3.0, 12.0)),
                },
            }

            # Inject hard negatives to ensure non-trivial labels
            r = np.random.rand()
            nfz = state["no_fly_zones"][int(np.random.randint(0, len(state["no_fly_zones"])))]
            cx, cy = nfz["center"]

            if r < 0.25:
                # Put target inside an NFZ (should be rejected by rules)
                cand["params"]["position"] = [float(cx), float(cy), 30.0]
            elif r < 0.40:
                # Force low battery (should be rejected by battery feasibility rule)
                cand_low_batt = True
            else:
                cand_low_batt = False


            # Featurize EXACTLY like the feasibility classifier does
            drone_state = state["drones"][cand["drone_id"]]
            # Apply the low-battery hard negative if requested above
            if 'cand_low_batt' in locals() and cand_low_batt:
                drone_state = dict(drone_state)
                drone_state['battery'] = 0.02

            feats = clf._featurize(cand, drone_state)

            # Label: use current heuristic feasibility proxy as teacher label (works now, then you can replace with rollout labels)
            res = verifier.verify(cand, drone_state, state["drones"], state["no_fly_zones"])
            label = 1 if res.accepted else 0

            X.append(feats)
            y.append(label)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    np.savez_compressed(out_path, X=X, y=y)
    print(f"Saved {len(y)} samples to {out_path}. X shape={X.shape}, pos_rate={y.mean():.3f}")

if __name__ == "__main__":
    main()
