#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Demonstration
Combining all Session 4 technologies in a complete example

@author: Hrishikesh Terdalkar
"""

###############################################################################

import json
import sqlite3
import multiprocessing as mp
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

###############################################################################

RNG_SEED = 42
np.random.seed(RNG_SEED)


class IntegratedResearchSystem:
    """
    Complete research system integrating database, APIs, parallel processing, and web interface concepts
    """

    def __init__(self, db_path="integrated_research.db"):
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        """Setup SQLite database for the integrated system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create experiments table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY,
                experiment_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                researcher TEXT,
                start_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create api_data table for external data
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS api_data (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                data_type TEXT NOT NULL,
                raw_data TEXT,
                processed_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create analysis_results table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY,
                experiment_id TEXT,
                analysis_type TEXT,
                parameters TEXT,
                results TEXT,
                processing_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def simulate_api_data_collection(self, days=7):
        """Simulate collecting data from multiple APIs"""
        print("Simulating API data collection...")

        # Simulate weather data collection
        weather_data = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            daily_weather = {
                "date": date.strftime("%Y-%m-%d"),
                "temperature": 15
                + 10 * np.sin(2 * np.pi * i / 7)
                + np.random.normal(0, 2),
                "humidity": 60
                + 20 * np.cos(2 * np.pi * i / 7)
                + np.random.normal(0, 5),
                "source": "weather_api",
            }
            weather_data.append(daily_weather)

        # Simulate material properties data
        materials = ["aluminum", "steel", "copper", "titanium"]
        material_data = []
        for material in materials:
            material_props = {
                "material": material,
                "density": np.random.uniform(1, 10),
                "youngs_modulus": np.random.uniform(50, 300),
                "thermal_conductivity": np.random.uniform(10, 400),
                "source": "materials_api",
            }
            material_data.append(material_props)

        return {"weather": weather_data, "materials": material_data}

    def store_api_data(self, api_data):
        """Store API data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for data_type, records in api_data.items():
            for record in records:
                cursor.execute(
                    """
                    INSERT INTO api_data (source, data_type, raw_data)
                    VALUES (?, ?, ?)
                """,
                    (record["source"], data_type, json.dumps(record)),
                )

        conn.commit()
        conn.close()
        print(
            f"Stored {sum(len(records) for records in api_data.values())} API records"
        )

    def parallel_data_processing(self, processing_tasks):
        """Process data in parallel"""

        def process_task(task_id, task_type):
            """Individual processing task"""
            start_time = datetime.now()

            if task_type == "statistical_analysis":
                # Simulate statistical analysis
                data = np.random.normal(0, 1, 1000)
                result = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "task_id": task_id,
                    "task_type": task_type,
                }
                processing_time = (datetime.now() - start_time).total_seconds()

            elif task_type == "signal_processing":
                # Simulate signal processing
                time_points = np.linspace(0, 10, 1000)
                signal = np.sin(
                    2 * np.pi * 5 * time_points
                ) + 0.5 * np.random.normal(0, 1, 1000)
                fft_result = np.fft.fft(signal)
                dominant_freq = np.argmax(np.abs(fft_result[1:500])) + 1

                result = {
                    "dominant_frequency": dominant_freq,
                    "signal_power": float(np.mean(signal**2)),
                    "task_id": task_id,
                    "task_type": task_type,
                }
                processing_time = (datetime.now() - start_time).total_seconds()

            else:
                result = {
                    "error": "Unknown task type",
                    "task_id": task_id,
                    "task_type": task_type,
                }
                processing_time = 0

            return result, processing_time

        print(f"Processing {len(processing_tasks)} tasks in parallel...")

        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [
                executor.submit(process_task, task["id"], task["type"])
                for task in processing_tasks
            ]
            results = [future.result() for future in futures]

        # Store results in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for result, processing_time in results:
            cursor.execute(
                """
                INSERT INTO analysis_results (analysis_type, parameters, results, processing_time)
                VALUES (?, ?, ?, ?)
            """,
                (
                    result.get("task_type", "parallel_processing"),
                    json.dumps({"task_id": result["task_id"]}),
                    json.dumps(result),
                    processing_time,
                ),
            )

        conn.commit()
        conn.close()

        return results

    def generate_research_report(self):
        """Generate a comprehensive research report"""
        conn = sqlite3.connect(self.db_path)

        # Get summary statistics
        experiments_count = conn.execute(
            "SELECT COUNT(*) FROM experiments"
        ).fetchone()[0]
        api_records_count = conn.execute(
            "SELECT COUNT(*) FROM api_data"
        ).fetchone()[0]
        analysis_count = conn.execute(
            "SELECT COUNT(*) FROM analysis_results"
        ).fetchone()[0]

        # Get recent analysis results
        recent_analysis = conn.execute(
            """
            SELECT analysis_type, AVG(processing_time) as avg_time, COUNT(*) as count
            FROM analysis_results
            GROUP BY analysis_type
        """
        ).fetchall()

        conn.close()

        report = {
            "generated_at": datetime.now().isoformat(),
            "system_summary": {
                "total_experiments": experiments_count,
                "api_data_records": api_records_count,
                "analysis_runs": analysis_count,
            },
            "performance_metrics": {
                analysis_type: {"average_time": avg_time, "run_count": count}
                for analysis_type, avg_time, count in recent_analysis
            },
            "recommendations": [
                "Consider implementing caching for frequently accessed API data",
                "Optimize database queries for better performance",
                "Implement data validation for API responses",
                "Add more parallel processing for computationally intensive tasks",
            ],
        }

        return report

    def demonstrate_integrated_workflow(self):
        """Demonstrate the complete integrated workflow"""
        print("=== INTEGRATED RESEARCH SYSTEM DEMONSTRATION ===")
        print("1. API Data Collection and Storage")
        print("-" * 50)

        # Collect and store API data
        api_data = self.simulate_api_data_collection(days=5)
        self.store_api_data(api_data)

        print(f"Collected weather data: {len(api_data['weather'])} days")
        print(
            f"Collected material data: {len(api_data['materials'])} materials"
        )

        print("\n2. Parallel Data Processing")
        print("-" * 50)

        # Create processing tasks
        processing_tasks = []
        for i in range(8):  # 8 tasks to demonstrate parallelism
            task_type = (
                "statistical_analysis" if i % 2 == 0 else "signal_processing"
            )
            processing_tasks.append({"id": i, "type": task_type})

        results = self.parallel_data_processing(processing_tasks)

        print(f"Completed {len(results)} parallel tasks")
        for result, processing_time in results[:3]:  # Show first 3 results
            print(f"  Task {result['task_id']}: {processing_time:.3f}s")

        print("\n3. Data Analysis and Reporting")
        print("-" * 50)

        report = self.generate_research_report()

        print("System Summary:")
        for metric, value in report["system_summary"].items():
            print(f"  {metric}: {value}")

        print("\nPerformance Metrics:")
        for analysis_type, metrics in report["performance_metrics"].items():
            print(
                f"  {analysis_type}: {metrics['run_count']} runs, avg {metrics['average_time']:.3f}s"
            )

        print("\n4. Web Dashboard Integration Ready")
        print("-" * 50)
        print("The system is now ready for web dashboard integration.")
        print("Key endpoints that would be available:")
        print("  - /api/experiments - List all experiments")
        print("  - /api/data/weather - Weather data API")
        print("  - /api/analysis/results - Analysis results")
        print("  - /api/report - System report")

        # Save detailed report
        with open("integrated_system_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("\nDetailed report saved to integrated_system_report.json")


def main():
    """Main demonstration function"""
    system = IntegratedResearchSystem()
    system.demonstrate_integrated_workflow()


if __name__ == "__main__":
    main()
