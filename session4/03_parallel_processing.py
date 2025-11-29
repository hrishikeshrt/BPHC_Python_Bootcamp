#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Processing for Research (intermediate)

Goal: show how running the same calculation many times
can be sped up by using multiple CPU cores.

You do not need to understand every detail of the Monte
Carlo examples - focus on the difference between the
sequential and parallel timings printed by the script.

@author: Hrishikesh Terdalkar
"""

###############################################################################

import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

###############################################################################

RNG_SEED = 42
np.random.seed(RNG_SEED)


class ResearchParallelProcessor:
    """Parallel processing utilities for research applications"""

    @staticmethod
    def get_system_info():
        """Get information about available processing resources"""
        cpu_total = mp.cpu_count()
        print("=== SYSTEM INFORMATION ===")
        print(f"CPU Cores: {cpu_total}")

        return cpu_total


class MonteCarloSimulator:
    """Parallel Monte Carlo simulation for engineering applications"""

    def __init__(self):
        self.results = []

    @staticmethod
    def monte_carlo_trial(
        trial_id, num_samples=1000, simulation_type="structural"
    ):
        """Single Monte Carlo trial - simulating different engineering scenarios"""

        if simulation_type == "structural":
            # Structural reliability simulation
            load = np.random.normal(100, 15, num_samples)  # kN - random load
            strength = np.random.normal(
                150, 20, num_samples
            )  # kN - material strength
            safety_margin = strength - load
            failures = np.sum(safety_margin < 0)
            reliability = 1 - (failures / num_samples)

            return {
                "trial_id": trial_id,
                "reliability": reliability,
                "mean_safety_margin": np.mean(safety_margin),
                "failure_probability": failures / num_samples,
            }

        elif simulation_type == "thermal":
            # Thermal analysis simulation
            initial_temp = np.random.normal(20, 2, num_samples)
            heat_input = np.random.normal(1000, 100, num_samples)
            material_resistance = np.random.normal(0.5, 0.1, num_samples)
            final_temp = initial_temp + heat_input * material_resistance

            # Check for overheating
            overheat_count = np.sum(final_temp > 100)

            return {
                "trial_id": trial_id,
                "mean_final_temp": np.mean(final_temp),
                "overheat_probability": overheat_count / num_samples,
                "temp_std": np.std(final_temp),
            }

        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")

    def run_sequential_simulation(
        self, num_trials=100, num_samples=1000, simulation_type="structural"
    ):
        """Run simulation sequentially for comparison"""
        print(f"Running {num_trials} {simulation_type} trials sequentially...")

        start_time = time.time()
        results = []

        for i in range(num_trials):
            result = self.monte_carlo_trial(i, num_samples, simulation_type)
            results.append(result)

        sequential_time = time.time() - start_time
        return results, sequential_time

    def run_parallel_simulation(
        self,
        num_trials=100,
        num_samples=1000,
        simulation_type="structural",
        num_workers=None,
    ):
        """Run simulation in parallel"""
        if num_workers is None:
            num_workers = mp.cpu_count()

        print(
            f"Running {num_trials} {simulation_type} trials in parallel ({num_workers} workers)..."
        )

        start_time = time.time()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    self.monte_carlo_trial, i, num_samples, simulation_type
                )
                for i in range(num_trials)
            ]
            results = [future.result() for future in futures]

        parallel_time = time.time() - start_time
        return results, parallel_time

    def analyze_simulation_results(self, results, simulation_type):
        """Analyze and summarize simulation results"""
        if simulation_type == "structural":
            reliabilities = [r["reliability"] for r in results]
            safety_margins = [r["mean_safety_margin"] for r in results]

            summary = {
                "simulation_type": simulation_type,
                "trials": len(results),
                "mean_reliability": np.mean(reliabilities),
                "reliability_std": np.std(reliabilities),
                "mean_safety_margin": np.mean(safety_margins),
                "reliability_95_ci": [
                    np.percentile(reliabilities, 2.5),
                    np.percentile(reliabilities, 97.5),
                ],
            }

        elif simulation_type == "thermal":
            final_temps = [r["mean_final_temp"] for r in results]
            overheat_probs = [r["overheat_probability"] for r in results]

            summary = {
                "simulation_type": simulation_type,
                "trials": len(results),
                "mean_final_temp": np.mean(final_temps),
                "overheat_probability": np.mean(overheat_probs),
                "temp_variation": np.std(final_temps),
            }

        return summary


class DataBatchProcessor:
    """Parallel processing for batch data operations"""

    @staticmethod
    def process_data_chunk(chunk_data, operation="statistics"):
        """Process a chunk of data with specified operation"""

        if operation == "statistics":
            return {
                "chunk_size": len(chunk_data),
                "mean": np.mean(chunk_data),
                "std": np.std(chunk_data),
                "min": np.min(chunk_data),
                "max": np.max(chunk_data),
            }

        elif operation == "fft":
            # Simulate FFT analysis
            fft_result = np.fft.fft(chunk_data)
            magnitude = np.abs(fft_result)
            return {
                "chunk_size": len(chunk_data),
                "dominant_frequency": np.argmax(
                    magnitude[1 : len(magnitude) // 2]
                )
                + 1,
                "max_magnitude": np.max(magnitude),
            }

        else:
            raise ValueError(f"Unknown operation: {operation}")

    def process_large_dataset(
        self, data, chunk_size=1000, operation="statistics", parallel=True
    ):
        """Process large dataset in chunks, optionally in parallel"""
        chunks = [
            data[i : i + chunk_size] for i in range(0, len(data), chunk_size)
        ]
        print(f"Processing {len(data)} points in {len(chunks)} chunks...")

        start_time = time.time()

        if parallel:
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self.process_data_chunk, chunk, operation)
                    for chunk in chunks
                ]
                results = [future.result() for future in futures]
        else:
            results = [
                self.process_data_chunk(chunk, operation) for chunk in chunks
            ]

        processing_time = time.time() - start_time
        return results, processing_time


def demonstrate_parallel_processing():
    """Demonstrate parallel processing capabilities"""
    print("=== PARALLEL PROCESSING DEMONSTRATION ===")

    # System information
    cpu_count = ResearchParallelProcessor.get_system_info()

    # Monte Carlo simulation comparison
    simulator = MonteCarloSimulator()

    print("\n1. Monte Carlo Simulation - Structural Reliability")
    print("-" * 50)

    # Sequential simulation
    seq_results, seq_time = simulator.run_sequential_simulation(
        num_trials=50, num_samples=5000, simulation_type="structural"
    )

    # Parallel simulation
    par_results, par_time = simulator.run_parallel_simulation(
        num_trials=50, num_samples=5000, simulation_type="structural"
    )

    # Performance comparison
    speedup = seq_time / par_time
    efficiency = (speedup / cpu_count) * 100

    print(f"Sequential time: {seq_time:.2f} seconds")
    print(f"Parallel time: {par_time:.2f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.1f}%")

    # Analyze results
    summary = simulator.analyze_simulation_results(par_results, "structural")
    print(f"\nSimulation Results:")
    print(f"Mean reliability: {summary['mean_reliability']:.3f}")
    print(f"Reliability std: {summary['reliability_std']:.3f}")
    print(
        f"95% CI: [{summary['reliability_95_ci'][0]:.3f}, {summary['reliability_95_ci'][1]:.3f}]"
    )

    # Thermal simulation
    print("\n2. Monte Carlo Simulation - Thermal Analysis")
    print("-" * 50)

    thermal_results, thermal_time = simulator.run_parallel_simulation(
        num_trials=30, simulation_type="thermal"
    )

    thermal_summary = simulator.analyze_simulation_results(
        thermal_results, "thermal"
    )
    print(f"Parallel time: {thermal_time:.2f} seconds")
    print(
        f"Mean final temperature: {thermal_summary['mean_final_temp']:.1f} deg C"
    )
    print(
        f"Overheat probability: {thermal_summary['overheat_probability']:.3f}"
    )

    # Batch data processing
    print("\n3. Batch Data Processing")
    print("-" * 50)

    processor = DataBatchProcessor()

    # Generate large dataset
    large_data = np.random.normal(0, 1, 50000)  # 50,000 data points

    # Sequential processing
    seq_batch, seq_batch_time = processor.process_large_dataset(
        large_data, chunk_size=1000, operation="statistics", parallel=False
    )

    # Parallel processing
    par_batch, par_batch_time = processor.process_large_dataset(
        large_data, chunk_size=1000, operation="statistics", parallel=True
    )

    batch_speedup = seq_batch_time / par_batch_time

    print(f"Dataset size: {len(large_data)} points")
    print(f"Sequential batch time: {seq_batch_time:.2f} seconds")
    print(f"Parallel batch time: {par_batch_time:.2f} seconds")
    print(f"Batch processing speedup: {batch_speedup:.2f}x")

    # Verify results are consistent
    seq_means = [chunk["mean"] for chunk in seq_batch]
    par_means = [chunk["mean"] for chunk in par_batch]

    if np.allclose(seq_means, par_means, rtol=1e-10):
        print("[OK] Sequential and parallel results are identical")
    else:
        print(
            "[WARN] Results differ between sequential and parallel processing"
        )


if __name__ == "__main__":
    demonstrate_parallel_processing()
