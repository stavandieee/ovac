"""Generate Table 6: LLM Plan Quality Comparison"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, ".")

from coordination.planners.llm_planner import LLMPlanner, SAR_MISSIONS
from coordination.envs.sar_aviary import SAREnvironment

N_PLANS = 30  # 30 plans, 3 missions x 10 seeds
VALID_TYPES = {"navigate", "inspect", "hover", "return_home"}

def analyze_plan(plan, env_state):
    """Analyze a single plan for quality metrics."""
    result = {
        "json_parse": True,
        "has_actions": len(plan.get("actions", [])) > 0,
        "n_actions": len(plan.get("actions", [])),
        "valid_types": 0,
        "invalid_types": 0,
        "has_position": 0,
        "missing_position": 0,
        "valid_drone_ids": 0,
        "invalid_drone_ids": 0,
        "nfz_aware": False,
        "actions_per_drone": {},
    }
    
    n_drones = len(env_state["drones"])
    nfzs = env_state.get("no_fly_zones", [])
    
    for a in plan.get("actions", []):
        # Type validity
        if a.get("type") in VALID_TYPES:
            result["valid_types"] += 1
        else:
            result["invalid_types"] += 1
        
        # Drone ID validity
        did = a.get("drone_id")
        if isinstance(did, int) and 0 <= did < n_drones:
            result["valid_drone_ids"] += 1
            result["actions_per_drone"][did] = result["actions_per_drone"].get(did, 0) + 1
        else:
            result["invalid_drone_ids"] += 1
        
        # Position params for navigate
        if a.get("type") == "navigate":
            pos = a.get("params", {}).get("position")
            if pos and len(pos) >= 2:
                result["has_position"] += 1
                # Check if position avoids NFZs
                for nfz in nfzs:
                    c = nfz["center"]
                    he = nfz["half_extents"]
                    if abs(pos[0] - c[0]) < he[0] and abs(pos[1] - c[1]) < he[1]:
                        pass  # In NFZ
                    else:
                        result["nfz_aware"] = True
            else:
                result["missing_position"] += 1
    
    # NFZ awareness: check if any waypoint is inside NFZ
    nfz_violations = 0
    for a in plan.get("actions", []):
        if a.get("type") == "navigate":
            pos = a.get("params", {}).get("position", [0,0,0])
            for nfz in nfzs:
                c = nfz["center"]
                he = nfz["half_extents"]
                if abs(pos[0] - c[0]) < he[0] and abs(pos[1] - c[1]) < he[1]:
                    nfz_violations += 1
    result["nfz_violations_in_plan"] = nfz_violations
    result["nfz_aware"] = nfz_violations == 0
    
    return result

print("="*60)
print("  TABLE 6: LLM Plan Quality (Claude claude-sonnet-4-5-20250929)")
print("="*60)

all_results = []
parse_failures = 0
total_attempts = 0

for seed in range(10):
    for mission in SAR_MISSIONS:
        total_attempts += 1
        env = SAREnvironment()
        state = env.reset(seed=seed)
        
        planner = LLMPlanner(backend="anthropic")
        plan = planner.generate_plan(mission["text"], state)
        
        if "error" in plan:
            parse_failures += 1
            all_results.append({
                "json_parse": False,
                "has_actions": False,
                "n_actions": 0,
                "error": plan["error"],
            })
            print(f"  Seed {seed} / {mission['id']}: PARSE FAIL - {plan['error'][:60]}")
        else:
            analysis = analyze_plan(plan, state)
            all_results.append(analysis)
            print(f"  Seed {seed} / {mission['id']}: {analysis['n_actions']} actions, "
                  f"NFZ-aware: {analysis['nfz_aware']}, "
                  f"valid types: {analysis['valid_types']}/{analysis['n_actions']}")

# Compute summary stats
parsed = [r for r in all_results if r.get("json_parse", False)]
with_actions = [r for r in parsed if r.get("has_actions", False)]

parse_rate = len(parsed) / total_attempts * 100
valid_plan_rate = len(with_actions) / total_attempts * 100
avg_actions = np.mean([r["n_actions"] for r in with_actions]) if with_actions else 0
nfz_aware_rate = np.mean([r["nfz_aware"] for r in with_actions]) * 100 if with_actions else 0
type_valid_rate = np.mean([r["valid_types"] / max(r["n_actions"], 1) for r in with_actions]) * 100 if with_actions else 0
drone_valid_rate = np.mean([r["valid_drone_ids"] / max(r["n_actions"], 1) for r in with_actions]) * 100 if with_actions else 0

print(f"\n{'='*60}")
print(f"  RESULTS: Plan Quality (n={total_attempts})")
print(f"{'='*60}")
print(f"  JSON Parse Rate:     {parse_rate:.1f}%")
print(f"  Valid Plans:         {valid_plan_rate:.1f}%")
print(f"  Avg Actions/Plan:    {avg_actions:.1f}")
print(f"  Valid Action Types:  {type_valid_rate:.1f}%")
print(f"  Valid Drone IDs:     {drone_valid_rate:.1f}%")
print(f"  NFZ-Aware Plans:     {nfz_aware_rate:.1f}%")

# Save
os.makedirs("results/table6", exist_ok=True)
with open("results/table6/plan_quality.json", "w") as f:
    json.dump({
        "model": "claude-sonnet-4-5-20250929",
        "n_plans": total_attempts,
        "parse_rate": parse_rate,
        "valid_plan_rate": valid_plan_rate,
        "avg_actions": avg_actions,
        "valid_type_rate": type_valid_rate,
        "valid_drone_rate": drone_valid_rate,
        "nfz_aware_rate": nfz_aware_rate,
        "per_plan": all_results,
    }, f, indent=2, default=str)

print(f"\n  ✓ Saved to results/table6/plan_quality.json")
