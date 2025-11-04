import dspy


class MovieAnalysisSignature(dspy.Signature):
    """Analyze movie preferences and generate recommendations."""

    movie_title: str = dspy.InputField()
    analysis_result: str = dspy.OutputField(desc="Thematic analysis with hypotheses")


movie_analysis_agent = dspy.ReAct(MovieAnalysisSignature, tools=[])


def call_movie_analysis_agent(movie_title: str) -> str:
    result = movie_analysis_agent(movie_title=movie_title)
    return result.analysis_result
