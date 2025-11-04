import dspy


class MovieAnalysisSignature(dspy.Signature):
    """Analyze movie preferences and generate recommendations."""

    movie_title: str = dspy.InputField()
    analysis_result: str = dspy.OutputField(desc="Thematic analysis with hypotheses")


class OrchestratorSignature(dspy.Signature):
    """Coordinate specialist agents for movie recommendations."""

    user_input: str = dspy.InputField()
    final_recommendations: str = dspy.OutputField(desc="Coordinated recommendations")
