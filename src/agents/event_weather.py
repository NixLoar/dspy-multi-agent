from datetime import datetime
import json
from typing import Any
import dspy

from datasource.weather import WeatherDataSource


class ExtractDateLocationSignature(dspy.Signature):
    """Interpret user input to identify event date and location."""

    user_input: str = dspy.InputField()
    event_date: str = dspy.OutputField(desc="Event date normalized to DD/MM/YYYY")
    location: str = dspy.OutputField(desc="City/Location for the event")
    forecast_json: str = dspy.OutputField(
        desc="JSON with summary,tmin,tmax,rain_chance"
    )


def fetch_weather_tool(event_date: str, location: str) -> dict[str, Any]:
    """Tool: Fetch weather forecast for given date and location."""
    weather_ds = WeatherDataSource()
    return weather_ds.get_forecast(date_ddmmyyyy=event_date)


def extract_date_location_tool(user_input: str) -> dict[str, str]:
    """Tool: Extract event date and location from user input."""
    today = datetime.now().strftime("%d/%m/%Y")
    extractor = dspy.Predict(ExtractDateLocationSignature)
    data = extractor(user_input=user_input, today=today)
    return {"event_date": data.event_date, "location": data.location}


_event_weather_agent = dspy.ReAct(
    ExtractDateLocationSignature,
    tools=[extract_date_location_tool, fetch_weather_tool],
)


def _get_event_weather_json(pred: dspy.Prediction) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_date": getattr(pred, "event_date", None),
        "location": getattr(pred, "location", None),
    }

    raw_forecast = getattr(pred, "forecast_json", None)
    if isinstance(raw_forecast, str):
        try:
            payload["forecast"] = json.loads(raw_forecast)
        except json.JSONDecodeError:
            payload["forecast"] = raw_forecast
    else:
        payload["forecast"] = raw_forecast

    return payload


def call_event_weather_agent(user_input: str) -> str:
    """
    Tool: Interpreta data+local do evento a partir do input do usuário e retorna a previsão do tempo.
    Retorno: JSON string com {event_date, location, forecast{summary,tmin,tmax,rain_chance}}
    """
    pred = _event_weather_agent(user_input=user_input)

    result = _get_event_weather_json(pred)

    return json.dumps(result)
