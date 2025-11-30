"""Design Space Exploration Agent Graph.

This graph implements a multi-step design exploration workflow:
1. Seed Generator - Creates initial candidate designs
2. Evaluator - Scores each candidate on multiple criteria
3. Refinement Loop - Iteratively improves top candidates
4. Selector - Chooses the best final design
5. Formatter - Presents results in a readable format
"""

from langgraph.graph import StateGraph, START, END
from src.agent.state import DesignState
from src.agent.nodes import (
    seed_generator_node,
    evaluator_node,
    refinement_node,
    selector_node,
    formatter_node,
    should_continue_refining
)


def create_design_agent():
    """Create and compile the design space exploration agent graph."""
    
    # Create the graph with our custom state
    graph = StateGraph(DesignState)
    
    # Add all nodes
    graph.add_node("seed_generator", seed_generator_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("refinement", refinement_node)
    graph.add_node("selector", selector_node)
    graph.add_node("formatter", formatter_node)
    
    # Define the flow
    # Start -> Generate initial candidates
    graph.add_edge(START, "seed_generator")
    
    # Evaluate the candidates
    graph.add_edge("seed_generator", "evaluator")
    
    # After evaluation, decide: refine more or select final
    graph.add_conditional_edges(
        "evaluator",
        should_continue_refining,
        {
            "refine": "refinement",
            "select": "selector"
        }
    )
    
    # After refinement, evaluate again
    graph.add_edge("refinement", "evaluator")
    
    # After selection, format output
    graph.add_edge("selector", "formatter")
    
    # End after formatting
    graph.add_edge("formatter", END)
    
    # Compile the graph
    return graph.compile()


# Create the default agent instance
graph = create_design_agent()
