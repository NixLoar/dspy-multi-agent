from dotenv import load_dotenv
import os
import dspy

from src.agents.movie_analyst import call_movie_analysis_agent

load_dotenv()

_OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")

lm = dspy.LM(
    "openai/gpt-4.1",
    api_key=_OPEN_AI_KEY,
)

dspy.configure(lm=lm)


class OrchestratorSignature(dspy.Signature):
    """Coordinate specialist agents for movie recommendations."""

    user_input: str = dspy.InputField()
    final_recommendations: str = dspy.OutputField(desc="Coordinated recommendations")


orchestrator = dspy.ReAct(
    OrchestratorSignature,
    tools=[
        call_movie_analysis_agent,
    ],
)
