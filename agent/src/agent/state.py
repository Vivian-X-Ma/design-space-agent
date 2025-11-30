"""Define the state schema for the design space exploration agent."""

from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class ComponentCandidate(TypedDict):
    """A single component candidate design."""
    id: int
    microcontroller: str
    sensor_type: str
    adc_bits: int
    ble_module: str
    battery: str
    rationale: str


class EvaluationScore(TypedDict):
    """Evaluation scores for a candidate."""
    candidate_id: int
    accuracy_score: float
    power_score: float
    cost_score: float
    reliability_score: float
    overall_score: float
    feedback: str


class DesignState(TypedDict):
    """The state of the design space exploration agent."""
    # Input requirements
    requirements: dict
    constraints: dict
    
    # Messages for agent communication
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Generated candidates
    candidates: List[ComponentCandidate]
    initial_candidates: List[ComponentCandidate]  # Store original 5 candidates
    
    # Evaluation results
    evaluations: List[EvaluationScore]
    
    # Refinement iteration
    iteration: int
    max_iterations: int
    
    # Final selection
    top_pick: Optional[ComponentCandidate]
    selection_reasoning: str
    
    # Formatted output for display
    formatted_output: str
