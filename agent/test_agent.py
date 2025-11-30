"""Test the design space exploration agent."""

import sys
sys.path.insert(0, '.')

from src.agent.graph import graph

# Define the initial state with requirements and constraints
initial_state = {
    "requirements": {
        "sampling_interval": "5 ± 0.2 min",
        "alert_thresholds": {"low": 70, "high": 180},
        "battery_life": ">=24h"
    },
    "constraints": {
        "adc_bits": "{10,12,14,16}",
        "ble_range": "≤5m"
    },
    "messages": [],
    "candidates": [],
    "initial_candidates": [],
    "evaluations": [],
    "iteration": 0,
    "max_iterations": 2,  # Number of refinement cycles
    "top_pick": None,
    "selection_reasoning": "",
    "formatted_output": ""
}

print("Starting Design Space Exploration Agent...")
print("=" * 60)
print("This will:")
print("  1. Generate 5 initial candidate designs")
print("  2. Evaluate each candidate on accuracy, power, cost, reliability")
print("  3. Refine top candidates (2 iterations)")
print("  4. Select the best final design")
print("=" * 60)
print()

# Run the agent
result = graph.invoke(initial_state)

# The formatted output is already printed by the formatter node
# Optionally show the raw top pick data
# print("\n[Debug] Top Pick Data:")
# print(result.get("top_pick"))
