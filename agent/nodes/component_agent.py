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

    # =========================
    # Pull Requirements
    # =========================
    sampling_interval = requirements.get("sampling_interval")
    alert_thresholds = requirements.get("alert_thresholds")
    battery_life = requirements.get("battery_life")
    alert_latency = requirements.get("alert_latency", "<10s")
    mard_target = requirements.get("mard_target", "<=10%")
    ble_reliability = requirements.get("ble_reliability", ">=99%")

    # =========================
    # Pull Constraints
    # =========================
    adc_bits = constraints.get("adc_bits")
    afe_compat = constraints.get("afe_sensor_compat", "AFE bias must match sensor output")
    sampling_stability = constraints.get("sampling_stability", "±0.2 min max drift")
    mcu_ble_interface = constraints.get("mcu_ble_interface", "{UART, SPI, I2C}")
    battery_form_factor = constraints.get("battery_form_factor", "Wearable patch form factor")
    ble_range = constraints.get("ble_range", "≤5m")

    # Validate required minimal fields
    missing = []
    if not sampling_interval: missing.append("sampling_interval")
    if not alert_thresholds: missing.append("alert_thresholds")
    if not battery_life: missing.append("battery_life")
    if not adc_bits: missing.append("adc_bits")
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    # Format alert thresholds
    if isinstance(alert_thresholds, dict):
        alert_str = f"<{alert_thresholds['low']} / >{alert_thresholds['high']} mg/dL"
    else:
        alert_str = str(alert_thresholds)

    # =========================
    # Build the Prompt
    # =========================
    prompt = f"""
    You are a CPS design assistant performing Design Space Exploration (DSE)
    for a continuous glucose monitoring (CGM) system. Generate 5 complete
    candidate hardware configurations that respect ALL objectives and constraints.

    ===========================
            SYSTEM OBJECTIVES
    ===========================
    1. Alert latency requirement: {alert_latency}
    2. Battery life requirement: {battery_life}
    3. Measurement accuracy target (MARD): {mard_target}
    4. BLE link reliability target: {ble_reliability}

    ===========================
            HARD CONSTRAINTS
    ===========================
    - Sampling interval: {sampling_interval}
    - Sampling stability: {sampling_stability}
    - Alert thresholds: {alert_str}
    - ADC bits must be one of: {adc_bits}
    - Sensor ↔ AFE compatibility: {afe_compat}
    - MCU ↔ BLE interface allowed: {mcu_ble_interface}
    - BLE range constraint: {ble_range}
    - Physical constraint: {battery_form_factor}

    ===========================
            DESIGN PARAMETERS
    ===========================
    1. Sensor & AFE:
       - Sensor type, bias voltage, response time (τ)
       - AFE gain, filtering, ADC resolution, noise floor

    2. Microcontroller:
       - MCU family (Cortex-M0/M4, Nordic NRF52, TI MSP430, etc.)
       - ADC sampling, clock modes, duty cycling

    3. BLE Subsystem:
       - BLE module, TX power, connection interval, packet bundling
       - Retry/ACK strategy

    4. Battery & Power:
       - Capacity, chemistry, regulator efficiency
       - Sleep strategy, power gating

    5. System Timing:
       - Sampling schedule relative to filtering window
       - BLE transmission scheduling

    ===========================
           REQUIRED OUTPUT
    ===========================
    Generate *exactly 5* candidate sets using this template:

    --------------------------------------------------------------------------------
    CANDIDATE SET #[number]
    --------------------------------------------------------------------------------
    * Microcontroller:
    * Sensor Type:
    * Sensor Bias Voltage:
    * Sensor τ (response time):
    * AFE Gain / Filter Strategy:
    * ADC Resolution (bits):
    * BLE Module / Tx Power / Interval:
    * Battery Capacity & Type:
    * Power Gating Strategy:
    * Sampling Interval:
    * BLE Bundling Strategy:

    Rationale:
    - Explain how this design meets latency, power, accuracy, and reliability goals
    - Discuss any trade-offs
    - Must satisfy ALL constraints
    --------------------------------------------------------------------------------

    After generating all 5 candidates, select a TOP CANDIDATE and provide a
    short justification.

    Return ONLY the raw text of the 5 candidates + the top candidate justification.
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