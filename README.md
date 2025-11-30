# Design Space Agent

### AI Integration component for Final Project - CS 8395: Special Topics - AI-Assisted Design (for Cyber Physical Systems)

By: Vivian Ma and Franklin Hu 
Due: 12/5/2025

A LangGraph-based agent for exploring CPS (Cyber-Physical Systems) design spaces. This agent generates, evaluates, and refines component candidates for our glucose monitor.


## Getting started: 

### 1. Clone the repository
```bash
git clone https://github.com/Vivian-X-Ma/design-space-agent.git
cd design-space-agent/agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file in the `agent` folder:
```
GROQ_API_KEY=your_groq_api_key_here
```

You can get a free API key from [Groq Console](https://console.groq.com/).

### 4. Run the agent
```bash
python test_agent.py
```

## How It Works

The agent follows a multi-step workflow:

1. **Seed Generation** - Creates 5 initial component candidates
2. **Evaluation** - Scores each candidate on accuracy, power, cost, and reliability
3. **Refinement** - Iteratively improves top 3 candidates (2 rounds)
4. **Selection** - Picks the best final design with detailed reasoning


## Customization

Edit `test_agent.py` to modify requirements and constraints:

```python
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
    "max_iterations": 2,  # Number of refinement cycles
    ...
}
```
