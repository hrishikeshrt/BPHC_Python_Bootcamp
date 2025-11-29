#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research Dashboard with Flask (advanced / optional)

This script is a larger example that combines several
ideas from the rest of Session 4 into a small web app.

If you are just starting with Python, treat this as a
preview. You can run it and click around without fully
understanding the internals yet.

@author: Hrishikesh Terdalkar
"""

###############################################################################

import io
import os
import json
import base64
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib

# Use a non-interactive backend so plots work safely in web requests
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file

###############################################################################

# For this demonstration, we'll create a simple Flask app
# In production, you'd separate templates into their own files

app = Flask(__name__)


class ResearchDashboard:
    """Research dashboard backend functionality"""

    def __init__(self):
        self.experiments = self.load_sample_data()

    def load_sample_data(self):
        """Load sample research data for demonstration"""
        # Generate sample engineering data
        experiments = {}

        # Thermal experiment
        time_points = np.arange(0, 120, 5)
        experiments["thermal_study"] = {
            "time": time_points,
            "temperature": 25
            + 10 * np.sin(2 * np.pi * time_points / 60)
            + np.random.normal(0, 1, len(time_points)),
            "heat_flow": 100
            + 30 * np.cos(2 * np.pi * time_points / 30)
            + np.random.normal(0, 5, len(time_points)),
        }

        # Structural experiment
        load_points = np.linspace(0, 1000, 50)
        experiments["structural_test"] = {
            "load": load_points,
            "displacement": 0.1 * load_points
            + 0.001 * load_points**2
            + np.random.normal(0, 0.5, len(load_points)),
            "stress": load_points / 100,  # Simplified stress calculation
        }

        # Fluid dynamics experiment
        velocity_points = np.linspace(0.1, 5.0, 40)
        experiments["flow_analysis"] = {
            "velocity": velocity_points,
            "pressure_drop": 10 * velocity_points**2
            + np.random.normal(0, 2, len(velocity_points)),
            "reynolds_number": 1000 * velocity_points,
        }

        return experiments

    def analyze_experiment(self, experiment_name, analysis_type):
        """Perform analysis on experiment data"""
        data = self.experiments[experiment_name]
        df = pd.DataFrame(data)

        analysis = {
            "experiment": experiment_name,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
        }

        if analysis_type == "basic_stats":
            # Basic statistical analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            stats = {}
            for col in numeric_cols:
                stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
            analysis["statistics"] = stats

        elif analysis_type == "trend_analysis":
            # Trend analysis
            trends = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            # Try to find time or independent variable
            indep_vars = [
                col
                for col in df.columns
                if col in ["time", "load", "velocity"]
            ]
            indep_var = indep_vars[0] if indep_vars else None

            for col in numeric_cols:
                if col != indep_var:
                    if indep_var:
                        # Linear regression
                        slope, intercept = np.polyfit(
                            df[indep_var], df[col], 1
                        )
                        trends[col] = {
                            "slope": float(slope),
                            "intercept": float(intercept),
                            "correlation": float(df[indep_var].corr(df[col])),
                        }

            analysis["trends"] = trends

        return analysis

    def create_plot(self, experiment_name, x_col, y_col, plot_type="line"):
        """Create visualization plot"""
        data = self.experiments[experiment_name]

        plt.figure(figsize=(10, 6))

        if plot_type == "line":
            plt.plot(
                data[x_col], data[y_col], "bo-", linewidth=2, markersize=4
            )
        elif plot_type == "scatter":
            plt.scatter(data[x_col], data[y_col], alpha=0.7, s=30)

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{experiment_name}: {y_col} vs {x_col}")
        plt.grid(True, alpha=0.3)

        # Save plot to bytes for web display
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format="png", dpi=300, bbox_inches="tight")
        img_bytes.seek(0)
        plt.close()

        # Convert to base64 for HTML embedding
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"


# Global dashboard instance
dashboard = ResearchDashboard()


# Flask Routes
@app.route("/")
def index():
    """Main dashboard page"""
    experiments_html_parts = []

    for exp_name, exp_data in dashboard.experiments.items():
        df = pd.DataFrame(exp_data)
        columns = list(df.columns)
        description = f"{exp_name.replace('_', ' ').title()} Data"
        options_html = "".join(
            f"<option value='{col}'>{col}</option>" for col in columns
        )
        experiments_html_parts.append(
            f"""
        <div class="experiment">
            <h3>{exp_name.replace('_', ' ').title()}</h3>
            <p>{description} - {len(df)} data points</p>
            <p>Columns: {', '.join(columns)}</p>
            <button onclick="analyzeExperiment('{exp_name}', 'basic_stats')">Basic Statistics</button>
            <button onclick="analyzeExperiment('{exp_name}', 'trend_analysis')">Trend Analysis</button>
            <div style="margin-top: 10px;">
                <label>
                    X:
                    <select id="xcol-{exp_name}">
                        {options_html}
                    </select>
                </label>
                <label>
                    Y:
                    <select id="ycol-{exp_name}">
                        {options_html}
                    </select>
                </label>
                <button onclick="plotFromSelect('{exp_name}')">
                    Plot
                </button>
            </div>
            <div id="results-{exp_name}"></div>
            <div id="plot-{exp_name}"></div>
        </div>
        """
        )

    experiments_html = "\n".join(experiments_html_parts)

    return f"""
    <html>
    <head>
        <title>Research Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .experiment {{ border: 1px solid #ccc; padding: 20px; margin: 10px; border-radius: 5px; }}
            .plot {{ margin: 20px 0; }}
            button {{ padding: 10px 15px; margin: 5px; cursor: pointer; }}
        </style>
    </head>
    <body>
        <h1>Research Data Dashboard</h1>
        <p>Interactive dashboard for research data analysis and visualization</p>

        <h2>Available Experiments</h2>
        {experiments_html}

        <script>
        function analyzeExperiment(experimentName, analysisType) {{
            fetch('/analyze', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    experiment: experimentName,
                    analysis_type: analysisType
                }})
            }})
            .then(response => response.json())
            .then(data => {{
                document.getElementById('results-' + experimentName).innerHTML =
                    '<h4>Analysis Results:</h4><pre>' + JSON.stringify(data, null, 2) + '</pre>';
            }});
        }}

        function plotFromSelect(experimentName) {{
            const xSelect = document.getElementById('xcol-' + experimentName);
            const ySelect = document.getElementById('ycol-' + experimentName);
            if (!xSelect || !ySelect) return;
            const xCol = xSelect.value;
            const yCol = ySelect.value;
            createPlot(experimentName, xCol, yCol);
        }}

        function createPlot(experimentName, xCol, yCol) {{
            fetch('/plot', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    experiment: experimentName,
                    x_column: xCol,
                    y_column: yCol
                }})
            }})
            .then(response => response.json())
            .then(data => {{
                document.getElementById('plot-' + experimentName).innerHTML =
                    '<h4>Plot:</h4><img src="' + data.plot_url + '" style="max-width: 100%;">';
            }});
        }}
        </script>
    </body>
    </html>
    """


@app.route("/analyze", methods=["POST"])
def analyze_data():
    """API endpoint for data analysis"""
    request_data = request.json
    experiment = request_data.get("experiment")
    analysis_type = request_data.get("analysis_type")

    if experiment not in dashboard.experiments:
        return jsonify({"error": "Experiment not found"}), 404

    analysis_results = dashboard.analyze_experiment(experiment, analysis_type)
    return jsonify(analysis_results)


@app.route("/plot", methods=["POST"])
def create_plot():
    """API endpoint for creating plots"""
    request_data = request.json
    experiment = request_data.get("experiment")
    x_column = request_data.get("x_column")
    y_column = request_data.get("y_column")

    if experiment not in dashboard.experiments:
        return jsonify({"error": "Experiment not found"}), 404

    if (
        x_column not in dashboard.experiments[experiment]
        or y_column not in dashboard.experiments[experiment]
    ):
        return jsonify({"error": "Invalid columns"}), 400

    plot_url = dashboard.create_plot(experiment, x_column, y_column)
    return jsonify({"plot_url": plot_url})


@app.route("/export/<experiment_name>")
def export_data(experiment_name):
    """Export experiment data as CSV"""
    if experiment_name not in dashboard.experiments:
        return "Experiment not found", 404

    data = dashboard.experiments[experiment_name]
    df = pd.DataFrame(data)

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=f"{experiment_name}_data.csv",
    )


@app.route("/api/experiments")
def get_experiments_list():
    """API endpoint to get list of experiments"""
    experiments_list = []

    for exp_name, exp_data in dashboard.experiments.items():
        df = pd.DataFrame(exp_data)
        experiments_list.append(
            {
                "name": exp_name,
                "columns": list(df.columns),
                "record_count": len(df),
                "description": f"{exp_name.replace('_', ' ').title()} Dataset",
            }
        )

    return jsonify(experiments_list)


def run_dashboard_demonstration():
    """Demonstrate dashboard functionality without running Flask server"""
    print("=== RESEARCH DASHBOARD DEMONSTRATION ===")

    # Test analysis functionality
    print("1. Data Analysis Examples:")

    analysis_types = ["basic_stats", "trend_analysis"]
    experiments = ["thermal_study", "structural_test"]

    for exp in experiments:
        for analysis_type in analysis_types:
            results = dashboard.analyze_experiment(exp, analysis_type)
            print(f"\n{exp} - {analysis_type}:")

            if "statistics" in results:
                print("  Statistics calculated for all numeric columns")
            if "trends" in results:
                print(
                    f"  Trends analyzed: {len(results['trends'])} relationships"
                )

    # Test plot generation
    print("\n2. Plot Generation:")
    plot_combinations = [
        ("thermal_study", "time", "temperature"),
        ("structural_test", "load", "displacement"),
        ("flow_analysis", "velocity", "pressure_drop"),
    ]

    for exp, x, y in plot_combinations:
        try:
            plot_data = dashboard.create_plot(exp, x, y)
            print(f"  {exp}: {y} vs {x} - Plot generated successfully")
        except Exception as e:
            print(f"  {exp}: Error generating plot - {e}")

    print("\n3. Dashboard Features:")
    print("  - Interactive web interface")
    print("  - Real-time data analysis")
    print("  - Dynamic plot generation")
    print("  - Data export functionality")
    print("  - REST API for programmatic access")

    print("\nTo run the actual web dashboard:")
    print("  flask --app session4/04_research_dashboard.py run --port 5000")
    print("Then visit http://localhost:5000 in your browser")


if __name__ == "__main__":
    # When run directly, demonstrate functionality
    run_dashboard_demonstration()

    # Uncomment the line below to actually run the Flask server
    # app.run(debug=True, port=5000)
