from typing import Any


class WeatherDataSource:
    def __init__(
        self,
        db: dict[tuple, dict[str, Any]] | None = None,
        fallback: dict[str, Any] | None = None,
    ):
        self._db: dict[tuple, dict[str, Any]] = db or {
            ("05-11-2025", "são paulo"): {
                "summary": "Parcialmente nublado",
                "tmin": 18,
                "tmax": 26,
                "rain_chance": 40,
            },
            ("05-11-2025", "rio de janeiro"): {
                "summary": "Quente e úmido",
                "tmin": 22,
                "tmax": 32,
                "rain_chance": 20,
            },
            ("06-11-2025", "são paulo"): {
                "summary": "Chuva leve",
                "tmin": 17,
                "tmax": 23,
                "rain_chance": 70,
            },
        }

        self._fallback = fallback or {
            "summary": "Tempo estável",
            "tmin": 19,
            "tmax": 27,
            "rain_chance": 30,
        }

    def get_forecast(self, date_ddmmyyyy: str) -> dict[str, Any]:
        return self._db.get(date_ddmmyyyy, self._fallback)

    def choose_weather_bucket(self, tmin: int, tmax: int, rain_chance: int) -> str:
        if rain_chance >= 60:
            return "chuva"
        avg = (tmin + tmax) / 2
        if avg <= 18:
            return "frio"
        if avg >= 27:
            return "quente"
        return "ameno"
