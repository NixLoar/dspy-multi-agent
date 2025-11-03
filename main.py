from dotenv import load_dotenv
import os
import dspy

load_dotenv()

_OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")

lm = dspy.LM(
    "openai/gpt-4.1",
    api_key=_OPEN_AI_KEY,
)

dspy.configure(lm=lm)


def evaluate_math(expression: str):
    return dspy.PythonInterpreter({}).execute(expression)


def search_wikipedia(query: str):
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(
        query, k=3
    )
    return [x["text"] for x in results]


react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

pred = react(
    question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?"
)

print(pred.answer)
