from datetime import datetime
import json
from typing import Any
import dspy

from datasource.weather import WeatherDataSource


class WeatherResultSignature(dspy.Signature):
    """Return weather for the interpreted date and location."""

    event_date: str = dspy.InputField()
    location: str = dspy.InputField()
    forecast_json: str = dspy.OutputField(
        desc="JSON with summary,tmin,tmax,rain_chance"
    )


class ExtractDateLocationSignature(dspy.Signature):
    """Interpret user input to identify event date and location."""

    user_input: str = dspy.InputField()
    today: str = dspy.InputField(desc="Today's date in DD/MM/YYYY")
    event_date: str = dspy.OutputField(desc="Event date normalized to DD/MM/YYYY")
    location: str = dspy.OutputField(desc="City/Location for the event")


class EventWeatherAgent(dspy.Module):
    """
    Retrieve weather information for a given event date and location.
    """

    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(ExtractDateLocationSignature)
        self.weathermapper = dspy.Predict(WeatherResultSignature)
        self.weather_ds = WeatherDataSource()

    def forward(self, user_input: str) -> dict[str, Any]:
        today = datetime.now().strftime("%d/%m/%Y")
        data_extracted = self.extractor(user_input=user_input, today=today)

        weather = self.weather_ds.get_forecast(date_ddmmyyyy=data_extracted.event_date)

        result = {
            "event_date": data_extracted.event_date,
            "location": data_extracted.location,
            "forecast": {
                "summary": weather["summary"],
                "tmin": weather["tmin"],
                "tmax": weather["tmax"],
                "rain_chance": weather["rain_chance"],
            },
        }

        return result


_event_weather_agent = EventWeatherAgent()


def call_event_weather_agent(user_input: str) -> str:
    """
    Tool: Interpreta data+local do evento a partir do input do usuário e retorna a previsão do tempo.
    Retorno: JSON string com {event_date, location, forecast{summary,tmin,tmax,rain_chance}}
    """
    result = _event_weather_agent(user_input=user_input)
    return json.dumps(result)
