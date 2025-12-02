"""Node functions for the design space exploration agent."""

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment!")


def get_llm():
    """Get a configured LLM instance."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )


def seed_generator_node(state: dict) -> dict:
    """Generate initial candidate component designs."""
    requirements = state.get("requirements")
    constraints = state.get("constraints")
    
    if not requirements or not constraints:
        raise ValueError("Missing requirements or constraints in state")
    
    # Extract fields
    sampling_interval = requirements.get("sampling_interval")
    alert_thresholds = requirements.get("alert_thresholds")
    battery_life = requirements.get("battery_life")
    adc_bits = constraints.get("adc_bits")
    ble_range = constraints.get("ble_range")

    # Format alert threshold string
    if isinstance(alert_thresholds, dict):
        alert_str = f"<{alert_thresholds['low']} / >{alert_thresholds['high']} mg/dL"
    else:
        alert_str = str(alert_thresholds)

    # Unified upgraded prompt
    prompt = f"""
    You are a CPS design assistant performing Design Space Exploration (DSE)
    for a continuous glucose monitoring (CGM) system. Generate 5 complete
    candidate hardware configurations that respect ALL objectives and constraints.

    ===========================
            SYSTEM OBJECTIVES
    ===========================
    1. Minimize alert latency (must detect & alert <10s)
    2. Minimize power consumption (≥ 24h battery life)
    3. Optimize measurement accuracy (MARD ≤ 10%)
    4. Maximize BLE reliability (≥99% successful transmissions)

    ===========================
              CONSTRAINTS
    ===========================
    - Sampling interval: {sampling_interval}
    - Alert thresholds: {alert_str}
    - Battery life: {battery_life}
    - ADC bits ∈ {adc_bits}
    - BLE range: {ble_range}
    - Sampling stability: ±0.2 min max drift
    - AFE ↔ sensor compatibility required
    - MCU ADC must match AFE output range
    - BLE module must match MCU interface (UART/SPI/I2C)
    - Must fit into wearable patch form factor

    ===========================
        DESIGN SPACE PARAMETERS
    ===========================
    1. Sensor & AFE:
       - sensor type, bias voltage, response time τ
       - AFE gain, filtering, ADC resolution

    2. MCU:
       - family (Cortex-M0/M4, Nordic, TI MSP…)
       - clock modes, ADC sampling, duty cycle

    3. BLE Subsystem:
       - tx power, advertising interval, payload size,
       - retry/ACK configuration

    4. Battery & Power:
       - capacity, chemistry, regulator efficiency
       - power gating strategy

    5. System Timing:
       - sampling interval & filtering window
       - BLE message bundling strategy

    ===========================
           REQUIRED OUTPUT
    ===========================
    Return your response as a JSON array with exactly 5 candidates.

    Each candidate must contain:
      - id (1–5)
      - microcontroller
      - sensor_type
      - sensor_bias_voltage
      - sensor_tau
      - afe_gain_filter
      - adc_bits
      - ble_module_tx_interval
      - battery
      - power_gating
      - sampling_interval
      - ble_bundling
      - rationale

    Rationale must link design choices to:
      - latency
      - power
      - accuracy
      - BLE reliability

    Return ONLY valid JSON. NO markdown, NO explanations outside JSON.
    """

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="You are a technical design assistant. Always respond with valid JSON only."),
        HumanMessage(content=prompt)
    ])
    
    try:
        candidates = json.loads(response.content)
        if isinstance(candidates, dict) and "candidates" in candidates:
            candidates = candidates["candidates"]
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\[[\s\S]*\]', response.content)
        if json_match:
            candidates = json.loads(json_match.group())
        else:
            candidates = []
    
    # Display initial candidates
    print("\n" + "=" * 80)
    print("                         INITIAL CANDIDATES GENERATED")
    print("=" * 80)
    for c in candidates:
        print(f"\n  #{c.get('id')}: {c.get('microcontroller')}")
        print(f"      Sensor: {c.get('sensor_type')} | ADC: {c.get('adc_bits')}-bit | BLE: {c.get('ble_module_tx_interval')}")
        print(f"      Battery: {c.get('battery')}")
    print("\n" + "-" * 80)
    
    return {
        "candidates": candidates,
        "initial_candidates": candidates,
        "messages": [HumanMessage(content=f"Generated {len(candidates)} initial candidates")]
    }

def evaluator_node(state: dict) -> dict:
    """Evaluate each candidate design and assign scores."""
    candidates = state.get("candidates", [])
    requirements = state.get("requirements", {})
    constraints = state.get("constraints", {})
    iteration = state.get("iteration", 0)
    
    if not candidates:
        return {"evaluations": [], "messages": [HumanMessage(content="No candidates to evaluate")]}
    
    # Show progress
    eval_label = "INITIAL" if iteration == 0 else f"ROUND {iteration}"
    print(f"\n   EVALUATING {len(candidates)} candidates ({eval_label})")
    
    prompt = f"""Evaluate each of these glucose monitoring system candidates based on:
1. Accuracy (how well the ADC and sensor meet precision needs)
2. Power efficiency (battery life optimization)
3. Cost (component affordability)
4. Reliability (proven components, robustness)

Requirements context:
- Sampling interval: {requirements.get('sampling_interval')}
- Alert thresholds: {requirements.get('alert_thresholds')}
- Battery life: {requirements.get('battery_life')}

Candidates to evaluate:
{json.dumps(candidates, indent=2)}

Return a JSON array with an evaluation for each candidate:
- candidate_id
- accuracy_score (0-10)
- power_score (0-10)
- cost_score (0-10)
- reliability_score (0-10)
- overall_score (weighted average)
- feedback (brief improvement suggestions)

Return ONLY valid JSON, no markdown or explanation."""

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="You are a technical evaluator. Always respond with valid JSON only."),
        HumanMessage(content=prompt)
    ])
    
    try:
        evaluations = json.loads(response.content)
        if isinstance(evaluations, dict) and "evaluations" in evaluations:
            evaluations = evaluations["evaluations"]
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\[[\s\S]*\]', response.content)
        if json_match:
            evaluations = json.loads(json_match.group())
        else:
            evaluations = []
    
    # Display evaluation results
    print(f"   Scores: ", end="")
    for e in evaluations:
        print(f"#{e.get('candidate_id')}={e.get('overall_score', 'N/A')} ", end="")
    print()
    
    return {
        "evaluations": evaluations,
        "messages": [HumanMessage(content=f"Evaluated {len(evaluations)} candidates")]
    }


def refinement_node(state: dict) -> dict:
    """Refine candidates based on evaluation feedback."""
    candidates = state.get("candidates", [])
    evaluations = state.get("evaluations", [])
    iteration = state.get("iteration", 0)
    requirements = state.get("requirements", {})
    constraints = state.get("constraints", {})
    
    # Get top 3 candidates by overall score
    sorted_evals = sorted(evaluations, key=lambda x: x.get("overall_score", 0), reverse=True)
    top_ids = [e["candidate_id"] for e in sorted_evals[:3]]
    top_candidates = [c for c in candidates if c.get("id") in top_ids]
    top_feedback = [e for e in evaluations if e.get("candidate_id") in top_ids]
    
    # Show progress
    print(f"\n   REFINING top 3 candidates (iteration {iteration + 1})")
    print(f"   Selected: {', '.join([f'#{cid}' for cid in top_ids])}")
    
    prompt = f"""Refine these top 3 glucose monitoring system candidates based on evaluation feedback.

Current top candidates:
{json.dumps(top_candidates, indent=2)}

Evaluation feedback:
{json.dumps(top_feedback, indent=2)}

Requirements:
- Sampling interval: {requirements.get('sampling_interval')}
- Alert thresholds: {requirements.get('alert_thresholds')}
- Battery life: {requirements.get('battery_life')}

Constraints:
- ADC_bits ∈ {constraints.get('adc_bits')}
- BLE range {constraints.get('ble_range')}

Improve each candidate based on its feedback while respecting constraints.
Return a JSON array with the 3 refined candidates (same structure as input).
Return ONLY valid JSON, no markdown or explanation."""

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="You are a design optimization expert. Always respond with valid JSON only."),
        HumanMessage(content=prompt)
    ])
    
    try:
        refined = json.loads(response.content)
        if isinstance(refined, dict) and "candidates" in refined:
            refined = refined["candidates"]
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\[[\s\S]*\]', response.content)
        if json_match:
            refined = json.loads(json_match.group())
        else:
            refined = top_candidates
    
    return {
        "candidates": refined,
        "iteration": iteration + 1,
        "messages": [HumanMessage(content=f"Refined to {len(refined)} candidates (iteration {iteration + 1})")]
    }


def selector_node(state: dict) -> dict:
    """Select the final top pick with detailed reasoning."""
    candidates = state.get("candidates", [])
    evaluations = state.get("evaluations", [])
    requirements = state.get("requirements", {})
    
    prompt = f"""Select the BEST candidate from these refined glucose monitoring system designs.

Final candidates:
{json.dumps(candidates, indent=2)}

Latest evaluations:
{json.dumps(evaluations, indent=2)}

Requirements:
{json.dumps(requirements, indent=2)}

Return a JSON object with:
- top_pick: the complete selected candidate object
- selection_reasoning: a single STRING (not nested object) with 2-3 paragraphs explaining why this is the best choice, considering accuracy, power efficiency, cost, and reliability trade-offs.

Example format:
{{
  "top_pick": {{...candidate object...}},
  "selection_reasoning": "This candidate offers the best balance of... The nRF52840 microcontroller provides... Additionally, the electrochemical sensor..."
}}

Return ONLY valid JSON, no markdown or explanation."""

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="You are a senior design architect. Always respond with valid JSON. The selection_reasoning must be a plain string, not a nested object."),
        HumanMessage(content=prompt)
    ])
    
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"top_pick": candidates[0] if candidates else None, "selection_reasoning": "Unable to parse selection"}
    
    # Ensure selection_reasoning is a string
    reasoning = result.get("selection_reasoning", "")
    if isinstance(reasoning, dict):
        # Convert dict to readable paragraphs
        parts = []
        for key, value in reasoning.items():
            parts.append(str(value))
        reasoning = " ".join(parts)
    
    return {
        "top_pick": result.get("top_pick"),
        "selection_reasoning": reasoning,
        "messages": [HumanMessage(content="Final selection complete")]
    }
    
    return {
        "top_pick": result.get("top_pick"),
        "selection_reasoning": result.get("selection_reasoning", ""),
        "messages": [HumanMessage(content="Final selection complete")]
    }


def formatter_node(state: dict) -> dict:
    """Format the final output for display."""
    candidates = state.get("candidates", [])
    evaluations = state.get("evaluations", [])
    top_pick = state.get("top_pick")
    selection_reasoning = state.get("selection_reasoning", "")
    iteration = state.get("iteration", 0)
    
    # Ensure selection_reasoning is a string
    if not isinstance(selection_reasoning, str):
        selection_reasoning = str(selection_reasoning)
    
    output_lines = []
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("              GLUCOSE MONITORING SYSTEM - DESIGN EXPLORATION")
    output_lines.append("=" * 80)
    output_lines.append(f"\n  Iterations completed: {iteration}")
    output_lines.append(f"  Final candidates: {len(candidates)}")
    output_lines.append("")
    
    # Show final candidates with scores
    output_lines.append("-" * 80)
    output_lines.append("  FINAL CANDIDATES")
    output_lines.append("-" * 80)
    
    for candidate in candidates:
        cid = candidate.get("id")
        eval_data = next((e for e in evaluations if e.get("candidate_id") == cid), {})
        
        output_lines.append(f"\n  #{cid}: {candidate.get('microcontroller', 'N/A')}")
        output_lines.append(f"      * Sensor: {candidate.get('sensor_type', 'N/A')}")
        output_lines.append(f"      * ADC: {candidate.get('adc_bits', 'N/A')}-bit")
        output_lines.append(f"      * BLE: {candidate.get('ble_module', 'N/A')}")
        output_lines.append(f"      * Battery: {candidate.get('battery', 'N/A')}")
        if eval_data:
            output_lines.append(f"      * Score: {eval_data.get('overall_score', 'N/A')}/10")
    
    # Show top pick
    if top_pick:
        top_pick_id = top_pick.get("id")
        top_pick_eval = next((e for e in evaluations if e.get("candidate_id") == top_pick_id), {})
        
        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append("                                   TOP PICK ")
        output_lines.append("=" * 80)
        output_lines.append("")
        output_lines.append(f"  {top_pick.get('microcontroller', 'N/A')}")
        output_lines.append(f"      * Sensor Type: {top_pick.get('sensor_type', 'N/A')}")
        output_lines.append(f"      * ADC Bits: {top_pick.get('adc_bits', 'N/A')}")
        output_lines.append(f"      * BLE Module: {top_pick.get('ble_module', 'N/A')}")
        output_lines.append(f"      * Battery: {top_pick.get('battery', 'N/A')}")
        
        # Show score breakdown
        if top_pick_eval:
            output_lines.append("")
            output_lines.append("  Score Breakdown:")
            output_lines.append(f"      * Accuracy:    {top_pick_eval.get('accuracy_score', 'N/A')}/10")
            output_lines.append(f"      * Power:       {top_pick_eval.get('power_score', 'N/A')}/10")
            output_lines.append(f"      * Cost:        {top_pick_eval.get('cost_score', 'N/A')}/10")
            output_lines.append(f"      * Reliability: {top_pick_eval.get('reliability_score', 'N/A')}/10")
            output_lines.append(f"      ─────────────────────")
            output_lines.append(f"      * Overall:     {top_pick_eval.get('overall_score', 'N/A')}/10")
        
        output_lines.append("")
        output_lines.append("  Why this is the best choice:")
        # Word wrap the reasoning
        words = selection_reasoning.split()
        line = "    "
        for word in words:
            if len(line) + len(word) + 1 > 76:
                output_lines.append(line)
                line = "    " + word
            else:
                line += " " + word if line.strip() else word
        if line.strip():
            output_lines.append(line)
    
    output_lines.append("")
    output_lines.append("=" * 80)
    
    formatted = "\n".join(output_lines)
    print(formatted)
    
    return {"formatted_output": formatted}


def should_continue_refining(state: dict) -> str:
    """Determine if we should continue refining or move to selection."""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)
    
    if iteration >= max_iterations:
        return "select"
    return "refine"
