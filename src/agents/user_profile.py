import json
from typing import Any
import dspy


class ExtractUserProfileSignature(dspy.Signature):
    """Interpret user input to identify gender and preferences."""

    user_input: str = dspy.InputField()
    gender: str = dspy.OutputField(
        desc="masculino, feminino, neutro (ou variação semelhante a mapear)"
    )
    preferences_json: str = dspy.OutputField(
        desc="JSON: {cores_preferidas?: [..], dresscode?: str, estilo?: str}"
    )


class UserProfileAgent(dspy.Module):
    """
    Use LLM to extract user profile information from input text.
    """

    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(ExtractUserProfileSignature)

    def forward(self, user_input: str) -> dict[str, Any]:
        prof = self.extractor(user_input=user_input)
        gender_raw = (prof.gender or "").strip().lower()
        if "fem" in gender_raw:
            gender_norm = "feminino"
        elif "masc" in gender_raw:
            gender_norm = "masculino"
        else:
            gender_norm = "neutro"

        try:
            prefs = json.loads(prof.preferences_json) if prof.preferences_json else {}
        except Exception:
            prefs = {}

        return {"gender": gender_norm, "preferences": prefs}


_user_profile_agent = UserProfileAgent()


def call_user_profile_agent(user_input: str) -> str:
    """
    Tool: Interpreta o perfil do usuário a partir do input (gênero e preferências).
    Retorno: JSON string com {gender, preferences}
    """
    result = _user_profile_agent(user_input=user_input)
    return json.dumps(result)
