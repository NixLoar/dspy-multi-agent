import json
from typing import Any
import dspy

from datasource.wardrobe import WARDROBE
from datasource.weather import WeatherDataSource


class OutfitRecommenderSignature(dspy.Signature):
    """Combine profile + weather and suggest up to 3 outfits."""

    profile_json: str = dspy.InputField()
    weather_json: str = dspy.InputField()
    combos_json: str = dspy.OutputField(desc="JSON list of up to 3 outfit suggestions")


class OutfitRecommenderAgent(dspy.Module):
    """
    Generate outfit recommendations based on user profile and weather.
    """

    def __init__(self):
        super().__init__()
        self.recommender = dspy.Predict(OutfitRecommenderSignature)
        self.weather_ds = WeatherDataSource()

    def forward(self, profile: dict[str, Any], weather: dict[str, Any]) -> list[str]:
        gender = profile.get("gender", "neutro")
        prefs = profile.get("preferences", {}) or {}

        forecast = weather.get("forecast", {})
        bucket = self.weather_ds.choose_weather_bucket(
            forecast.get("tmin", 19),
            forecast.get("tmax", 27),
            forecast.get("rain_chance", 30),
        )

        catalog = WARDROBE.get(gender, WARDROBE["neutro"]).get(bucket, [])

        colors = (
            set([color.lower() for color in prefs.get("cores_preferidas", [])])
            if prefs
            else set()
        )
        if colors:
            priorizadas = [
                color
                for color in catalog
                if any(color in color.lower() for color in colors)
            ]
            restantes = [color for color in catalog if color not in priorizadas]
            ordered = (priorizadas + restantes)[:3]
        else:
            ordered = catalog[:3]

        combos_json = json.dumps(ordered, ensure_ascii=False)
        _ = self.recommender(
            profile_json=json.dumps(profile, ensure_ascii=False),
            weather_json=json.dumps(weather, ensure_ascii=False),
            combos_json=combos_json,
        )

        return ordered


_outfit_agent = OutfitRecommenderAgent()


def call_outfit_recommender_agent(context_json: str) -> str:
    """
    Tool: Recebe um JSON com {"profile": {...}, "weather": {...}} e retorna até 3 combinações possíveis.
    Retorno: JSON string com lista de sugestões de looks.
    """
    ctx = json.loads(context_json)
    profile = ctx.get("profile", {})
    weather = ctx.get("weather", {})
    combos = _outfit_agent(profile=profile, weather=weather)
    return json.dumps(combos)
