#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Integration for Research Data
Working with REST APIs for data acquisition

@author: Hrishikesh Terdalkar
"""

###############################################################################

import time
import json
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np

###############################################################################


class ResearchDataAPI:
    """Base class for research data API integration"""

    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()

        # Common headers for API requests
        self.headers = {
            "User-Agent": "Research Data Collector/1.0",
            "Accept": "application/json",
        }

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def make_request(self, endpoint, params=None, method="GET"):
        """Make API request with error handling"""
        url = f"{self.base_url}/{endpoint}" if self.base_url else endpoint

        try:
            if method == "GET":
                response = self.session.get(
                    url, params=params, headers=self.headers
                )
            elif method == "POST":
                response = self.session.post(
                    url, json=params, headers=self.headers
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()  # Raise exception for bad status codes
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None

    def rate_limit_delay(self, delay_seconds=1):
        """Simple rate limiting"""
        time.sleep(delay_seconds)


class WeatherDataCollector(ResearchDataAPI):
    """Collect weather data for environmental studies"""

    def __init__(self, api_key=None):
        # Using OpenWeatherMap API structure
        super().__init__("http://api.openweathermap.org/data/2.5", api_key)

    def get_current_weather(self, city, country_code=None):
        """Get current weather data (mock implementation)"""
        # Mock data for demonstration - in real implementation, use actual API
        city_label = f"{city},{country_code}" if country_code else city

        mock_data = {
            "weather": [{"main": "Clear", "description": "clear sky"}],
            "main": {
                "temp": 15.5 + np.random.normal(0, 3),
                "pressure": 1013 + np.random.normal(0, 5),
                "humidity": 65 + np.random.normal(0, 10),
                "temp_min": 13.0,
                "temp_max": 18.0,
            },
            "wind": {"speed": 3.1 + np.random.normal(0, 1), "deg": 240},
            "name": city,
            "dt": int(datetime.now().timestamp()),
        }

        return self._process_weather_data(mock_data)

    def get_historical_weather(self, city, days=7):
        """Generate historical weather data (mock)"""
        historical_data = []
        end_date = datetime.now()

        for i in range(days):
            current_date = end_date - timedelta(days=i)

            # Generate realistic seasonal data
            day_of_year = current_date.timetuple().tm_yday
            base_temp = 10 + 10 * np.sin(
                2 * np.pi * (day_of_year - 80) / 365
            )  # Seasonal variation

            daily_data = {
                "date": current_date.strftime("%Y-%m-%d"),
                "temperature": base_temp + np.random.normal(0, 2),
                "pressure": 1013 + np.random.normal(0, 3),
                "humidity": 60 + np.random.normal(0, 15),
                "wind_speed": 3 + abs(np.random.normal(0, 1.5)),
                "conditions": np.random.choice(
                    ["Clear", "Cloudy", "Rain", "Snow"]
                ),
            }
            historical_data.append(daily_data)

        return historical_data

    def _process_weather_data(self, raw_data):
        """Process raw API response into structured format"""
        return {
            "city": raw_data.get("name", "Unknown"),
            "temperature": raw_data["main"]["temp"],
            "pressure": raw_data["main"]["pressure"],
            "humidity": raw_data["main"]["humidity"],
            "wind_speed": raw_data["wind"]["speed"],
            "conditions": raw_data["weather"][0]["main"],
            "timestamp": datetime.fromtimestamp(raw_data["dt"]).isoformat(),
        }

    def analyze_weather_trends(self, historical_data):
        """Analyze weather trends for research"""
        df = pd.DataFrame(historical_data)
        df["date"] = pd.to_datetime(df["date"])

        analysis = {
            "analysis_period": {
                "start": df["date"].min().strftime("%Y-%m-%d"),
                "end": df["date"].max().strftime("%Y-%m-%d"),
                "days": len(df),
            },
            "temperature_analysis": {
                "mean": df["temperature"].mean(),
                "trend": (
                    "increasing"
                    if df["temperature"].iloc[-1] > df["temperature"].iloc[0]
                    else "decreasing"
                ),
                "daily_variation": df["temperature"].std(),
            },
            "pressure_analysis": {
                "mean": df["pressure"].mean(),
                "correlation_with_temp": df["temperature"].corr(
                    df["pressure"]
                ),
            },
            "condition_frequency": df["conditions"].value_counts().to_dict(),
        }

        return analysis, df


class MaterialPropertiesAPI(ResearchDataAPI):
    """Mock API for material properties data"""

    def __init__(self):
        super().__init__("https://api.materialsproject.org", "mock_key")

        # Mock material database
        self.materials_db = {
            "aluminum": {
                "density": 2.70,  # g/cm^3
                "youngs_modulus": 69,  # GPa
                "thermal_conductivity": 237,  # W/m*K
                "specific_heat": 0.897,  # J/g*K
            },
            "steel": {
                "density": 7.85,
                "youngs_modulus": 200,
                "thermal_conductivity": 50,
                "specific_heat": 0.466,
            },
            "copper": {
                "density": 8.96,
                "youngs_modulus": 110,
                "thermal_conductivity": 401,
                "specific_heat": 0.385,
            },
        }

    def get_material_properties(self, material_name):
        """Get properties for a specific material"""
        material_name = material_name.lower()

        if material_name in self.materials_db:
            return self.materials_db[material_name]
        else:
            return {"error": f"Material '{material_name}' not found"}

    def compare_materials(self, material_list, property_name):
        """Compare specific property across materials"""
        comparison = {}

        for material in material_list:
            props = self.get_material_properties(material)
            if property_name in props:
                comparison[material] = props[property_name]

        return comparison


def demonstrate_api_integration():
    """Demonstrate API integration with research data"""
    print("=== API INTEGRATION DEMONSTRATION ===")

    # Weather data collection
    weather_collector = WeatherDataCollector()

    print("1. Current Weather Data:")
    current_weather = weather_collector.get_current_weather("London", "UK")
    for key, value in current_weather.items():
        print(f"   {key}: {value}")

    print("\n2. Historical Weather Analysis:")
    historical_data = weather_collector.get_historical_weather("London", 30)
    analysis, weather_df = weather_collector.analyze_weather_trends(
        historical_data
    )

    print(
        f"   Period: {analysis['analysis_period']['start']} to {analysis['analysis_period']['end']}"
    )
    print(
        f"   Mean temperature: {analysis['temperature_analysis']['mean']:.1f} deg C"
    )
    print(f"   Trend: {analysis['temperature_analysis']['trend']}")

    # Material properties API
    materials_api = MaterialPropertiesAPI()

    print("\n3. Material Properties:")
    materials = ["aluminum", "steel", "copper"]
    for material in materials:
        props = materials_api.get_material_properties(material)
        print(f"   {material.capitalize()}:")
        print(f"     Density: {props['density']} g/cm^3")
        print(f"     Young's Modulus: {props['youngs_modulus']} GPa")

    print("\n4. Material Comparison (Thermal Conductivity):")
    comparison = materials_api.compare_materials(
        materials, "thermal_conductivity"
    )
    for material, conductivity in comparison.items():
        print(f"   {material}: {conductivity} W/m*K")

    # Save data for further analysis
    weather_df.to_csv("weather_analysis_data.csv", index=False)
    print("\nWeather data saved to weather_analysis_data.csv")


if __name__ == "__main__":
    demonstrate_api_integration()
