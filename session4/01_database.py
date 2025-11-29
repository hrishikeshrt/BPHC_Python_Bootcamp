#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research Database Integration (advanced / optional)

This script uses SQLAlchemy (an Object-Relational Mapper)
to talk to a SQLite database. It is here to show what a
more `real world' database layer can look like.

If you are new to Python, you do not need to understand
every line - you can simply run the script once to see
that experiments and data points are stored in a database.

@author: Hrishikesh Terdalkar
"""

###############################################################################

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

###############################################################################

RNG_SEED = 42
np.random.seed(RNG_SEED)

Base = declarative_base()


class ResearchExperiment(Base):
    """ORM class for research experiments"""

    __tablename__ = "research_experiments"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(String(100), unique=True, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    researcher = Column(String(100))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<Experiment(id={self.experiment_id}, title='{self.title}')>"


class ExperimentalData(Base):
    """ORM class for experimental data points"""

    __tablename__ = "experimental_data"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    parameter_name = Column(String(100), nullable=False)
    parameter_value = Column(Float, nullable=False)
    uncertainty = Column(Float)
    units = Column(String(50))
    notes = Column(Text)

    def __repr__(self):
        return f"<DataPoint(exp={self.experiment_id}, param={self.parameter_name}, value={self.parameter_value})>"


class ResearchDatabase:
    """Database management class for research data"""

    def __init__(self, db_url="sqlite:///research_data.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def create_experiment(
        self,
        experiment_id,
        title,
        researcher,
        description="",
        start_date=None,
        end_date=None,
    ):
        """Create a new experiment record

        If an experiment with the same ``experiment_id`` already exists,
        return the existing record instead of raising an error. This makes
        the demonstration script safe to run multiple times.
        """
        # Check for existing experiment first (idempotent behaviour for the demo)
        existing = (
            self.session.query(ResearchExperiment)
            .filter_by(experiment_id=experiment_id)
            .first()
        )
        if existing is not None:
            return existing

        experiment = ResearchExperiment(
            experiment_id=experiment_id,
            title=title,
            researcher=researcher,
            description=description,
            start_date=start_date or datetime.now(),
            end_date=end_date,
        )
        self.session.add(experiment)
        self.session.commit()
        return experiment

    def add_data_point(
        self,
        experiment_id,
        parameter_name,
        parameter_value,
        timestamp=None,
        uncertainty=None,
        units="",
        notes="",
    ):
        """Add a single data point"""
        data_point = ExperimentalData(
            experiment_id=experiment_id,
            parameter_name=parameter_name,
            parameter_value=parameter_value,
            timestamp=timestamp or datetime.now(),
            uncertainty=uncertainty,
            units=units,
            notes=notes,
        )
        self.session.add(data_point)
        self.session.commit()
        return data_point

    def add_data_batch(self, experiment_id, data_frame, time_column="time"):
        """Add multiple data points from a DataFrame"""
        data_points = []

        for idx, row in data_frame.iterrows():
            if time_column in data_frame.columns:
                timestamp = pd.to_datetime(row[time_column])
                if isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.to_pydatetime()
            else:
                timestamp = datetime.now()

            for col in data_frame.columns:
                if col != time_column and pd.api.types.is_numeric_dtype(
                    data_frame[col]
                ):
                    data_point = ExperimentalData(
                        experiment_id=experiment_id,
                        parameter_name=col,
                        parameter_value=float(row[col]),
                        timestamp=timestamp,
                    )
                    data_points.append(data_point)

        self.session.bulk_save_objects(data_points)
        self.session.commit()
        return len(data_points)

    def get_experiment_data(self, experiment_id, parameter_name=None):
        """Retrieve data for a specific experiment"""
        query = self.session.query(ExperimentalData).filter_by(
            experiment_id=experiment_id
        )

        if parameter_name:
            query = query.filter_by(parameter_name=parameter_name)

        results = query.all()

        # Convert to DataFrame for analysis
        if results:
            data = []
            for result in results:
                data.append(
                    {
                        "timestamp": result.timestamp,
                        "parameter": result.parameter_name,
                        "value": result.parameter_value,
                        "uncertainty": result.uncertainty,
                        "units": result.units,
                    }
                )
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()

    def get_experiment_statistics(self, experiment_id):
        """Calculate statistics for an experiment"""
        data = self.get_experiment_data(experiment_id)

        if data.empty:
            return None

        statistics = {}
        parameters = data["parameter"].unique()

        for param in parameters:
            param_data = data[data["parameter"] == param]["value"]
            statistics[param] = {
                "count": len(param_data),
                "mean": float(param_data.mean()),
                "std": float(param_data.std()),
                "min": float(param_data.min()),
                "max": float(param_data.max()),
            }

        return statistics

    def export_experiment_to_csv(self, experiment_id, output_file):
        """Export experiment data to CSV"""
        data = self.get_experiment_data(experiment_id)

        if not data.empty:
            data.to_csv(output_file, index=False)
            return True
        return False

    def list_experiments(self):
        """List all experiments in the database"""
        return self.session.query(ResearchExperiment).all()


def demonstrate_database_operations():
    """Demonstrate database operations with sample data"""
    print("=== RESEARCH DATABASE DEMONSTRATION ===")

    # Initialize database
    db = ResearchDatabase()

    # Create sample experiment
    experiment = db.create_experiment(
        experiment_id="thermal_study_001",
        title="Thermal Conductivity Measurement",
        researcher="PhD Student",
        description="Measuring thermal conductivity of composite materials",
    )
    print(f"Created experiment: {experiment}")

    csv_path = Path("engineering_test_data.csv")
    if csv_path.exists():
        thermal_df = pd.read_csv(csv_path)
        if "time" not in thermal_df.columns:
            anchor = pd.Timestamp("2024-01-01 09:00:00")
            if "Time_min" in thermal_df.columns:
                thermal_df["time"] = anchor + pd.to_timedelta(
                    thermal_df["Time_min"], unit="m"
                )
            else:
                thermal_df["time"] = pd.date_range(
                    start=anchor, periods=len(thermal_df), freq="min"
                )
        print(f"Loaded dataset from {csv_path}")
    else:
        time_points = pd.date_range(start="2024-01-01", periods=100, freq="H")
        temperatures = (
            25
            + 10 * np.sin(2 * np.pi * np.arange(100) / 24)
            + np.random.normal(0, 1, 100)
        )
        heat_flux = (
            100
            + 20 * np.cos(2 * np.pi * np.arange(100) / 12)
            + np.random.normal(0, 5, 100)
        )

        thermal_df = pd.DataFrame(
            {
                "time": time_points,
                "temperature": temperatures,
                "heat_flux": heat_flux,
            }
        )
        print("Generated synthetic thermal dataset for demonstration")

    # Add data batch
    points_added = db.add_data_batch("thermal_study_001", thermal_df, "time")
    print(f"Added {points_added} data points")

    # Retrieve and analyze data
    data = db.get_experiment_data("thermal_study_001")
    print(f"Retrieved data shape: {data.shape}")

    statistics = db.get_experiment_statistics("thermal_study_001")
    print("\nExperiment Statistics:")
    for param, stats in statistics.items():
        print(f"{param}: mean={stats['mean']:.2f} +/- {stats['std']:.2f}")

    # List all experiments
    experiments = db.list_experiments()
    print(f"\nTotal experiments in database: {len(experiments)}")

    # Export to CSV
    db.export_experiment_to_csv("thermal_study_001", "thermal_data_export.csv")
    print("Data exported to thermal_data_export.csv")


if __name__ == "__main__":
    demonstrate_database_operations()
