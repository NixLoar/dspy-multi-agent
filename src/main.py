import os
from dotenv import load_dotenv
import dspy
import mlflow
from agents.event_weather import call_event_weather_agent
from agents.outfit_recomendation import call_outfit_recommender_agent
from agents.user_profile import call_user_profile_agent

load_dotenv()

_OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")

mlflow.set_experiment("multiagent-dspy")
mlflow.dspy.autolog()

lm = dspy.LM(
    "openai/gpt-4.1",
    api_key=_OPEN_AI_KEY,
)

dspy.configure(lm=lm)


class OrchestratorSignature(dspy.Signature):
    """Coordinate specialist agents for a dress planner assistant."""

    user_input: str = dspy.InputField()
    final_recommendations: str = dspy.OutputField(desc="Coordinated recommendations")


if __name__ == "__main__":
    orchestrator = dspy.ReAct(
        OrchestratorSignature,
        tools=[
            call_event_weather_agent,
            call_user_profile_agent,
            call_outfit_recommender_agent,
        ],
    )

    user_input = input("Digite sua mensagem: ")
    final_answer = orchestrator(user_input=user_input).final_recommendations

    print(f"\nSugestão de look: {final_answer}")
