#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Research Data Analyzer
Comprehensive tool with multiple analysis options

@author: Hrishikesh Terdalkar
"""

###############################################################################

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################


class ResearchDataAnalyzer:
    """Advanced research data analysis tool"""

    def __init__(self):
        self.parser = self.setup_parser()

    def setup_parser(self):
        """Setup advanced argument parser"""
        parser = argparse.ArgumentParser(
            description="Advanced Research Data Analyzer",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python session3/04_advanced_analyzer.py -i engineering_test_data.csv --stats --plot
  python session3/04_advanced_analyzer.py -i engineering_test_data.csv --trends --correlations
  python session3/04_advanced_analyzer.py -i engineering_test_data.csv --filter "Temperature_C > 25" --verbose
            """,
        )

        # Input/output
        parser.add_argument(
            "--input", "-i", required=True, help="Input CSV file"
        )
        parser.add_argument(
            "--output",
            "-o",
            default="advanced_analysis",
            help="Output directory",
        )

        # Analysis options
        parser.add_argument(
            "--stats", action="store_true", help="Calculate statistics"
        )
        parser.add_argument(
            "--trends", action="store_true", help="Analyze trends"
        )
        parser.add_argument(
            "--correlations",
            action="store_true",
            help="Calculate correlations",
        )

        # Output options
        parser.add_argument(
            "--plot", action="store_true", help="Generate plots"
        )
        parser.add_argument(
            "--export-csv", action="store_true", help="Export processed data"
        )

        # Filtering
        parser.add_argument(
            "--filter",
            type=str,
            help='Filter condition (e.g., "Temperature > 25")',
        )

        # General
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Verbose output"
        )

        return parser

    def load_data(self, input_file):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(input_file)
            if self.args.verbose:
                print(f"[OK] Loaded {len(df)} rows from {input_file}")
            return df
        except Exception as e:
            print(f"[ERROR] Loading {input_file}: {e}")
            return None

    def calculate_statistics(self, df):
        """Calculate comprehensive statistics"""
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            stats[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q1": float(df[col].quantile(0.25)),
                "q3": float(df[col].quantile(0.75)),
            }

        return stats

    def analyze_trends(self, df):
        """Analyze trends in time series data"""
        trends = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Try to find time column
        time_cols = [col for col in df.columns if "time" in col.lower()]
        time_col = time_cols[0] if time_cols else None

        for col in numeric_cols:
            if col != time_col:
                if time_col:
                    # Linear regression using time
                    slope, intercept = np.polyfit(df[time_col], df[col], 1)
                else:
                    # Use index as proxy for time
                    slope, intercept = np.polyfit(range(len(df)), df[col], 1)

                trends[col] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "trend": "increasing" if slope > 0 else "decreasing",
                }

        return trends

    def generate_plots(self, df, output_dir):
        """Generate visualization plots"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Time series plots
        time_cols = [col for col in df.columns if "time" in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            for col in numeric_cols:
                if col != time_col:
                    plt.figure(figsize=(10, 6))
                    plt.plot(df[time_col], df[col], "bo-", alpha=0.7)
                    plt.xlabel(time_col)
                    plt.ylabel(col)
                    plt.title(f"{col} vs {time_col}")
                    plt.grid(True, alpha=0.3)
                    plt.savefig(
                        f"{output_dir}/{col}_plot.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

        if self.args.verbose:
            print("[OK] Plots generated")

    def run_analysis(self, args):
        """Run complete analysis"""
        self.args = args

        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        # Load data
        df = self.load_data(args.input)
        if df is None:
            return False

        results = {
            "analysis_date": datetime.now().isoformat(),
            "input_file": args.input,
            "data_shape": df.shape,
        }

        # Apply filter if specified
        if args.filter:
            try:
                # Simple filter implementation
                if ">" in args.filter:
                    col, value = args.filter.split(">")
                    col, value = col.strip(), float(value.strip())
                    initial_count = len(df)
                    df = df[df[col] > value]
                    results["filter_applied"] = args.filter
                    results["records_filtered"] = (
                        f"{initial_count} -> {len(df)}"
                    )
            except Exception as e:
                print(f"Warning: Filter not applied - {e}")

        # Perform analyses
        if args.stats:
            results["statistics"] = self.calculate_statistics(df)

        if args.trends:
            results["trends"] = self.analyze_trends(df)

        if (
            args.correlations
            and len(df.select_dtypes(include=[np.number]).columns) > 1
        ):
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            results["correlations"] = corr_matrix.to_dict()

        # Generate outputs
        if args.plot:
            self.generate_plots(df, args.output)

        if args.export_csv:
            df.to_csv(f"{args.output}/processed_data.csv", index=False)

        # Save results
        with open(f"{args.output}/analysis_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        if args.verbose:
            self.print_summary(results)

        return True

    def print_summary(self, results):
        """Print analysis summary"""
        print("\n" + "=" * 50)
        print("ADVANCED ANALYSIS SUMMARY")
        print("=" * 50)

        print(f"Input: {results['input_file']}")
        print(f"Records: {results['data_shape'][0]}")

        if "statistics" in results:
            print("\nStatistics:")
            for col, stats in results["statistics"].items():
                print(
                    f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}"
                )

        if "trends" in results:
            print("\nTrends:")
            for col, trend in results["trends"].items():
                print(
                    f"  {col}: slope={trend['slope']:.4f} ({trend['trend']})"
                )


def main():
    """Main function"""
    analyzer = ResearchDataAnalyzer()
    args = analyzer.parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)

    success = analyzer.run_analysis(args)

    if success:
        print(f"\n[OK] Advanced analysis complete! Results in {args.output}/")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
