"""
LLM Mission Planner

Translates natural-language SAR missions into structured plans
using GPT-4o, Claude, or LLaMA-3.

The planner receives:
  - Mission description (natural language)
  - Agent capabilities manifest
  - Environment state summary

And outputs:
  - Structured JSON plan with high-level actions
  - Each action has: agent assignment, target, completion criteria
"""

import json
import os
from typing import Dict, List, Optional


# =============================================================
# Prompt Template
# =============================================================

SAR_SYSTEM_PROMPT = """You are a multi-drone mission planner for search-and-rescue operations. 
You receive a mission description, the current state of all drones, and environment information.
You must output a JSON plan that decomposes the mission into executable actions.

RULES:
1. Each action must be assigned to a specific drone by ID
2. Respect no-fly zones — never route through them
3. Respect battery constraints — drones with <20% battery should return home
4. Distribute search areas evenly across available drones
5. Prioritize high-priority targets as described in the mission
6. Include "return_home" as the final action for each drone
7. CRITICAL: Generate 6 navigate+inspect pairs per drone for thorough coverage
8. Space waypoints 50-60m apart in a grid or lawnmower pattern within each drone sector
9. The arena is large (500x500m) — 3 waypoints per drone is NOT enough. Detection range is only 50m
10. Each drone should cover its assigned sector systematically. Output ONLY valid JSON, no markdown

OUTPUT FORMAT (JSON only, no markdown):
{
  "plan_id": "unique_string",
  "strategy": "brief description of approach",
  "actions": [
    {
      "action_id": 1,
      "drone_id": 0,
      "type": "navigate",
      "params": {"position": [x, y, z]},
      "purpose": "why this action"
    },
    {
      "action_id": 2,
      "drone_id": 0,
      "type": "inspect",
      "params": {"altitude": 20, "n_passes": 3},
      "purpose": "search sector for survivors"
    }
  ]
}

AVAILABLE PRIMITIVE TYPES:
- navigate: move to position [x, y, z]
- inspect: lower altitude and do multi-pass search at current location
- hover: hold position for observation (params: duration_steps)
- return_home: return to starting position

Respond with ONLY the JSON plan. No explanations."""


def build_mission_prompt(mission_text: str, env_state: dict) -> str:
    """Build the user prompt with mission and state."""
    prompt = f"""MISSION: {mission_text}

ENVIRONMENT STATE:
- Arena size: {env_state['arena_size'][0]}m x {env_state['arena_size'][1]}m
- Time remaining: {env_state['max_steps'] - env_state['step']} steps out of {env_state['max_steps']}

DRONES:
"""
    for d in env_state["drones"]:
        prompt += (
            f"  Drone {d['id']}: position=({d['position'][0]:.0f}, "
            f"{d['position'][1]:.0f}, {d['position'][2]:.0f}), "
            f"battery={d['battery']:.1%}, active={d['active']}, "
            f"targets_found={len(d['targets_found'])}\n"
        )

    prompt += f"\nTARGET STATUS: {env_state['targets_found']}/{env_state['targets_total']} found"
    prompt += f" ({env_state['high_priority_found']}/{env_state['high_priority_total']} high-priority)\n"

    prompt += "\nNO-FLY ZONES:\n"
    for nfz in env_state["no_fly_zones"]:
        c = nfz["center"]
        he = nfz["half_extents"]
        prompt += (
            f"  NFZ {nfz['id']}: center=({c[0]:.0f}, {c[1]:.0f}), "
            f"size=({he[0]*2:.0f}x{he[1]*2:.0f}m)\n"
        )

    prompt += "\nGenerate the mission plan as JSON."
    return prompt


# =============================================================
# LLM Backends
# =============================================================

class LLMPlanner:
    """
    Base class for LLM-based mission planning.
    
    Supports: OpenAI (GPT-4o), Anthropic (Claude), local (LLaMA via vLLM).
    """

    def __init__(
        self,
        backend: str = "anthropic",
        model: str = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens

        if backend == "openai":
            self.model = model or "gpt-4o"
            self.client = self._init_openai()
        elif backend == "anthropic":
            self.model = model or "claude-sonnet-4-5-20250929"
            self.client = self._init_anthropic()
        elif backend == "local":
            self.model = model or "meta-llama/Llama-3-70b-chat-hf"
            self.client = None  # TODO: vLLM integration
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_openai(self):
        from openai import OpenAI
        return OpenAI()  # uses OPENAI_API_KEY env var

    def _init_anthropic(self):
        import anthropic
        return anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    def generate_plan(
        self,
        mission_text: str,
        env_state: dict,
    ) -> dict:
        """
        Generate a structured mission plan from natural language.

        Returns:
            Parsed JSON plan dict, or error dict if parsing fails
        """
        user_prompt = build_mission_prompt(mission_text, env_state)

        try:
            raw_response = self._call_llm(user_prompt)
            plan = self._parse_plan(raw_response)
            plan["_raw_response"] = raw_response
            plan["_model"] = self.model
            plan["_backend"] = self.backend
            return plan
        except Exception as e:
            return {
                "error": str(e),
                "actions": [],
                "_raw_response": raw_response if 'raw_response' in dir() else "",
                "_model": self.model,
            }

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM API and return raw text response."""

        if self.backend == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SAR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        elif self.backend == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                system=SAR_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.content[0].text

        elif self.backend == "local":
            raise NotImplementedError("Local LLM backend not yet implemented")

    def _parse_plan(self, raw: str) -> dict:
        """Parse JSON plan from LLM response."""
        # Strip markdown code blocks if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        plan = json.loads(text)

        # Validate minimal structure
        if "actions" not in plan:
            raise ValueError("Plan missing 'actions' key")

        for action in plan["actions"]:
            if "type" not in action:
                raise ValueError(f"Action missing 'type': {action}")
            if "drone_id" not in action:
                raise ValueError(f"Action missing 'drone_id': {action}")

        return plan


class SymbolicPlanner:
    """
    Baseline: simple rule-based planner (no LLM).
    
    Divides arena into sectors and assigns each drone to sweep its sector.
    """

    def generate_plan(self, mission_text: str, env_state: dict) -> dict:
        """Generate a simple grid-sweep plan."""
        n_drones = len(env_state["drones"])
        arena_w, arena_h = env_state["arena_size"]
        altitude = 30.0

        actions = []
        action_id = 1

        for i, drone in enumerate(env_state["drones"]):
            if not drone["active"] or drone["battery"] < 0.2:
                continue

            # Assign sector (vertical strips)
            x_start = i * arena_w / n_drones + 20
            x_end = (i + 1) * arena_w / n_drones - 20
            x_mid = (x_start + x_end) / 2

            # Lawnmower pattern: 3 waypoints per sector
            waypoints = [
                [x_mid, arena_h * 0.25, altitude],
                [x_mid, arena_h * 0.50, altitude],
                [x_mid, arena_h * 0.75, altitude],
            ]

            for wp in waypoints:
                actions.append({
                    "action_id": action_id,
                    "drone_id": drone["id"],
                    "type": "navigate",
                    "params": {"position": wp},
                    "purpose": f"sweep sector {i}",
                })
                action_id += 1

                actions.append({
                    "action_id": action_id,
                    "drone_id": drone["id"],
                    "type": "inspect",
                    "params": {"altitude": 20, "n_passes": 3},
                    "purpose": f"inspect waypoint in sector {i}",
                })
                action_id += 1

            # Return home
            actions.append({
                "action_id": action_id,
                "drone_id": drone["id"],
                "type": "return_home",
                "params": {},
                "purpose": "return to base",
            })
            action_id += 1

        return {
            "plan_id": "symbolic_sweep",
            "strategy": "Grid sweep with lawnmower pattern",
            "actions": actions,
            "_model": "symbolic",
            "_backend": "rule_based",
        }


# =============================================================
# Mission Definitions
# =============================================================

SAR_MISSIONS = [
    {
        "id": "sar_basic",
        "text": (
            "Search the disaster zone for survivors. Prioritize people on "
            "elevated surfaces and those wearing high-visibility clothing. "
            "Avoid restricted areas. Complete search within battery limits. "
            "Report all confirmed sightings with coordinates."
        ),
    },
    {
        "id": "sar_flood",
        "text": (
            "A flash flood has hit the eastern sector. Search for stranded "
            "individuals, especially near rooftops and elevated terrain. "
            "Weather is deteriorating — complete the search quickly and "
            "return all drones before battery reaches 20%."
        ),
    },
    {
        "id": "sar_building_collapse",
        "text": (
            "A building has collapsed in the center of the area. Search for "
            "survivors in the rubble zone and surrounding streets. Be careful "
            "of unstable structures — maintain minimum altitude of 25 meters "
            "over the collapse zone. Coordinate drones to avoid overlap."
        ),
    },
]


# =============================================================
# Quick Test
# =============================================================

if __name__ == "__main__":
    print("Testing Planners...\n")

    # Test symbolic planner
    env_state = {
        "arena_size": [500.0, 500.0],
        "step": 0,
        "max_steps": 14400,
        "drones": [
            {"id": i, "position": [(i+1)*125, 10, 30],
             "battery": 1.0, "active": True, "targets_found": []}
            for i in range(4)
        ],
        "targets_found": 0,
        "targets_total": 12,
        "high_priority_found": 0,
        "high_priority_total": 5,
        "no_fly_zones": [
            {"id": 0, "center": [250, 250], "half_extents": [30, 30]},
        ],
    }

    sym = SymbolicPlanner()
    plan = sym.generate_plan(SAR_MISSIONS[0]["text"], env_state)

    print(f"Symbolic plan: {plan['strategy']}")
    print(f"Actions: {len(plan['actions'])}")
    for a in plan["actions"][:5]:
        print(f"  {a['action_id']}: drone {a['drone_id']} -> {a['type']} | {a['purpose']}")
    print(f"  ... ({len(plan['actions']) - 5} more)")

    # LLM planner test (only if API key is set)
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\nTesting LLM planner (Anthropic)...")
        llm = LLMPlanner(backend="anthropic")
        plan = llm.generate_plan(SAR_MISSIONS[0]["text"], env_state)
        if "error" not in plan:
            print(f"LLM plan: {plan.get('strategy', 'N/A')}")
            print(f"Actions: {len(plan['actions'])}")
        else:
            print(f"LLM error: {plan['error']}")
    else:
        print("\nSkipping LLM test (no ANTHROPIC_API_KEY set)")
