import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment!")


def seed_generator_node(state: MessagesState):
    # Extract requirements and constraints from state
    requirements = state.get("requirements")
    constraints = state.get("constraints")
    
    if not requirements:
        raise ValueError("Missing 'requirements' in state")
    if not constraints:
        raise ValueError("Missing 'constraints' in state")
    
    sampling_interval = requirements.get("sampling_interval")
    alert_thresholds = requirements.get("alert_thresholds")
    battery_life = requirements.get("battery_life")
    
    adc_bits = constraints.get("adc_bits")
    ble_range = constraints.get("ble_range")
    
    # Validate required fields
    if not all([sampling_interval, alert_thresholds, battery_life, adc_bits, ble_range]):
        missing = []
        if not sampling_interval: missing.append("sampling_interval")
        if not alert_thresholds: missing.append("alert_thresholds")
        if not battery_life: missing.append("battery_life")
        if not adc_bits: missing.append("adc_bits")
        if not ble_range: missing.append("ble_range")
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    
    # Format alert thresholds
    if isinstance(alert_thresholds, dict):
        alert_str = f"<{alert_thresholds['low']} / >{alert_thresholds['high']} mg/dL"
    else:
        alert_str = str(alert_thresholds)
    
    prompt = f"""
You are a CPS design assistant.
Select candidate components for a glucose monitoring system.

Requirements:
- Sampling interval: {sampling_interval}
- Alert thresholds: {alert_str}
- Battery: {battery_life}

Constraints:
- ADC_bits ∈ {adc_bits}
- BLE range {ble_range}

Generate 5 candidate sets. Format your response nicely for terminal display using this structure:

================================================================================
                    GLUCOSE MONITORING SYSTEM CANDIDATES
================================================================================

For each candidate, use:

────────────────────────────────────────────────────────────────────────────────
CANDIDATE SET #[number]
────────────────────────────────────────────────────────────────────────────────

  * Microcontroller: [value]
  * Sensor Type: [value]
  * ADC Bits: [value]
  * BLE Module: [value]
  * Battery: [value]

  Rationale:
    [explanation text with line breaks for readability]

After all 5 candidates, include a TOP PICK recommendation like this:

================================================================================
                                     TOP PICK 
================================================================================

  CANDIDATE SET #[number]

  * Microcontroller: [value]
  * Sensor Type: [value]
  * ADC Bits: [value]
  * BLE Module: [value]
  * Battery: [value]

  Why this is the best choice:
    [Explain why this candidate is recommended over the others, considering
    the balance of accuracy, power efficiency, cost, and reliability]

================================================================================

Use clear spacing, bullets (*), and visual separators. Make it easy to scan and read.
"""

    # Create the Groq LLM instance
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )
    
    agent = create_react_agent(
        model=llm,
        tools=[]
    )

    try:
        # Invoke the agent with the prompt as a message
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        # Extract the last message content from the result
        llm_response = result["messages"][-1].content
        print("\n" + llm_response)
    except Exception as e:
        print("LLM call failed:", e)
        return {"candidates": {"error": str(e)}}

    return {"candidates": llm_response}