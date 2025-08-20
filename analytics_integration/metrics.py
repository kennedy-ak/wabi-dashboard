#!/usr/bin/env python3
"""
LangSmith Dashboard Plot Generator
Fetches data from LangSmith and creates various dashboard plots
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from langsmith import Client
import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class LangSmithDashboard:
    def __init__(self, api_key: str = None, project_name: str = "wabi-fastapi"):
        """
        Initialize the LangSmith Dashboard

        Args:
            api_key: LangSmith API key (or set LANGSMITH_API_KEY env var)
            project_name: Default project name to fetch data from
        """
        if api_key:
            os.environ["LANGSMITH_API_KEY"] = api_key

        self.client = Client()
        self.project_name = project_name
        self.runs_data = None

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def fetch_runs(self, project_name: str = None, limit: int = 1000,
                   days_back: int = 30, status_filter: str = None) -> pd.DataFrame:
        """
        Fetch runs data from LangSmith

        Args:
            project_name: Project to fetch from (uses default if None)
            limit: Maximum number of runs to fetch
            days_back: How many days back to fetch data
            status_filter: Filter by status ('success', 'error', etc.)

        Returns:
            DataFrame with processed runs data
        """
        project = project_name or self.project_name
        if not project:
            raise ValueError("Project name must be provided")

        print(f"Fetching runs from project: {project}")

        # Set up filters
        filter_conditions = []
        if status_filter:
            filter_conditions.append(f'eq(status, "{status_filter}")')

        # Date filter for recent data
        start_date = datetime.now() - timedelta(days=days_back)
        filter_conditions.append(f'gte(start_time, "{start_date.isoformat()}")')

        filter_str = ' and '.join(filter_conditions) if filter_conditions else None

        try:
            # Fetch runs
            runs = self.client.list_runs(
                project_name=project,
                limit=limit,
                filter=filter_str
            )

            # Convert to list and process
            runs_list = list(runs)
            print(f"Fetched {len(runs_list)} runs")

            # Extract data
            data = []
            for run in runs_list:
                # Calculate latency
                latency = None
                if run.start_time and run.end_time:
                    latency = (run.end_time - run.start_time).total_seconds()

                # Extract token counts if available
                prompt_tokens = None
                completion_tokens = None
                total_tokens = None

                if run.outputs and isinstance(run.outputs, dict):
                    usage = run.outputs.get('usage', {})
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get('prompt_tokens')
                        completion_tokens = usage.get('completion_tokens')
                        total_tokens = usage.get('total_tokens')

                data.append({
                    'id': str(run.id),
                    'name': run.name or 'Unknown',
                    'start_time': run.start_time,
                    'end_time': run.end_time,
                    'latency': latency,
                    'status': run.status,
                    'run_type': run.run_type,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens,
                    'error': run.error if hasattr(run, 'error') else None,
                    'parent_run_id': str(run.parent_run_id) if run.parent_run_id else None
                })

            self.runs_data = pd.DataFrame(data)

            # Clean up data
            if not self.runs_data.empty:
                self.runs_data['start_time'] = pd.to_datetime(self.runs_data['start_time'])
                self.runs_data['end_time'] = pd.to_datetime(self.runs_data['end_time'])

                # Create time-based features
                self.runs_data['hour'] = self.runs_data['start_time'].dt.hour
                self.runs_data['day_of_week'] = self.runs_data['start_time'].dt.day_name()
                self.runs_data['date'] = self.runs_data['start_time'].dt.date

            return self.runs_data

        except Exception as e:
            print(f"Error fetching runs: {e}")
            return pd.DataFrame()

    def plot_latency_timeline(self, save_path: str = None):
        """Plot latency over time"""
        if self.runs_data is None or self.runs_data.empty:
            print("No data available. Fetch runs first.")
            return

        plt.figure(figsize=(15, 8))

        # Filter out None latencies
        valid_data = self.runs_data.dropna(subset=['latency'])

        if valid_data.empty:
            print("No latency data available for plotting")
            return

        # Main plot
        plt.subplot(2, 2, 1)
        plt.plot(valid_data['start_time'], valid_data['latency'],
                alpha=0.7, linewidth=1, marker='o', markersize=2)
        plt.title('Request Latency Over Time')
        plt.xlabel('Time')
        plt.ylabel('Latency (seconds)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Moving average
        if len(valid_data) > 10:
            window_size = min(20, len(valid_data) // 5)
            moving_avg = valid_data.set_index('start_time')['latency'].rolling(
                window=f'{window_size}T'
            ).mean()
            plt.plot(moving_avg.index, moving_avg.values,
                    color='red', linewidth=2, label=f'{window_size}min Moving Avg')
            plt.legend()

        # Latency distribution
        plt.subplot(2, 2, 2)
        plt.hist(valid_data['latency'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Latency Distribution')
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Frequency')
        plt.axvline(valid_data['latency'].mean(), color='red',
                   linestyle='--', label=f'Mean: {valid_data["latency"].mean():.2f}s')
        plt.legend()

        # Latency by hour
        plt.subplot(2, 2, 3)
        hourly_latency = valid_data.groupby('hour')['latency'].agg(['mean', 'std']).reset_index()
        plt.errorbar(hourly_latency['hour'], hourly_latency['mean'],
                    yerr=hourly_latency['std'], capsize=5, marker='o')
        plt.title('Average Latency by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Latency (seconds)')
        plt.grid(True, alpha=0.3)

        # Status distribution
        plt.subplot(2, 2, 4)
        status_counts = self.runs_data['status'].value_counts()
        plt.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
        plt.title('Run Status Distribution')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_token_usage(self, save_path: str = None):
        """Plot token usage metrics"""
        if self.runs_data is None or self.runs_data.empty:
            print("No data available. Fetch runs first.")
            return

        # Filter runs with token data
        token_data = self.runs_data.dropna(subset=['total_tokens'])

        if token_data.empty:
            print("No token usage data available")
            return

        plt.figure(figsize=(15, 10))

        # Token usage over time
        plt.subplot(2, 3, 1)
        plt.plot(token_data['start_time'], token_data['total_tokens'],
                alpha=0.7, marker='o', markersize=2)
        plt.title('Total Token Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Total Tokens')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Token type breakdown
        plt.subplot(2, 3, 2)
        prompt_data = token_data.dropna(subset=['prompt_tokens'])
        completion_data = token_data.dropna(subset=['completion_tokens'])

        if not prompt_data.empty and not completion_data.empty:
            plt.scatter(prompt_data['prompt_tokens'], completion_data['completion_tokens'],
                       alpha=0.6, s=30)
            plt.title('Prompt vs Completion Tokens')
            plt.xlabel('Prompt Tokens')
            plt.ylabel('Completion Tokens')
            plt.grid(True, alpha=0.3)

        # Daily token consumption
        plt.subplot(2, 3, 3)
        daily_tokens = token_data.groupby('date')['total_tokens'].sum().reset_index()
        plt.bar(range(len(daily_tokens)), daily_tokens['total_tokens'])
        plt.title('Daily Token Consumption')
        plt.xlabel('Days')
        plt.ylabel('Total Tokens')
        plt.xticks(range(0, len(daily_tokens), max(1, len(daily_tokens)//10)),
                  [d.strftime('%m-%d') for d in daily_tokens['date'][::max(1, len(daily_tokens)//10)]],
                  rotation=45)

        # Token usage distribution
        plt.subplot(2, 3, 4)
        plt.hist(token_data['total_tokens'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Token Usage Distribution')
        plt.xlabel('Total Tokens')
        plt.ylabel('Frequency')
        plt.axvline(token_data['total_tokens'].mean(), color='red',
                   linestyle='--', label=f'Mean: {token_data["total_tokens"].mean():.0f}')
        plt.legend()

        # Hourly token patterns
        plt.subplot(2, 3, 5)
        hourly_tokens = token_data.groupby('hour')['total_tokens'].mean()
        plt.bar(hourly_tokens.index, hourly_tokens.values)
        plt.title('Average Token Usage by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Tokens')

        # Cost estimation (approximate)
        plt.subplot(2, 3, 6)
        if not prompt_data.empty and not completion_data.empty:
            # Rough cost estimation (adjust rates as needed)
            INPUT_COST_PER_1K = 0.003  # Example rate
            OUTPUT_COST_PER_1K = 0.015  # Example rate

            token_data['estimated_cost'] = (
                (token_data['prompt_tokens'] / 1000 * INPUT_COST_PER_1K) +
                (token_data['completion_tokens'] / 1000 * OUTPUT_COST_PER_1K)
            )

            daily_cost = token_data.groupby('date')['estimated_cost'].sum().reset_index()
            plt.plot(range(len(daily_cost)), daily_cost['estimated_cost'], marker='o')
            plt.title('Estimated Daily Cost')
            plt.xlabel('Days')
            plt.ylabel('Cost ($)')
            plt.xticks(range(0, len(daily_cost), max(1, len(daily_cost)//5)),
                      [d.strftime('%m-%d') for d in daily_cost['date'][::max(1, len(daily_cost)//5)]],
                      rotation=45)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Token usage plot saved to {save_path}")

        plt.show()

    def plot_error_analysis(self, save_path: str = None):
        """Plot error analysis and success rates"""
        if self.runs_data is None or self.runs_data.empty:
            print("No data available. Fetch runs first.")
            return

        plt.figure(figsize=(15, 8))

        # Success rate over time
        plt.subplot(2, 3, 1)
        daily_success = self.runs_data.groupby('date').agg({
            'status': lambda x: (x == 'success').mean() * 100
        }).reset_index()

        plt.plot(range(len(daily_success)), daily_success['status'],
                marker='o', linewidth=2)
        plt.title('Daily Success Rate')
        plt.xlabel('Days')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 105)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, len(daily_success), max(1, len(daily_success)//5)),
                  [d.strftime('%m-%d') for d in daily_success['date'][::max(1, len(daily_success)//5)]],
                  rotation=45)

        # Error distribution
        plt.subplot(2, 3, 2)
        error_runs = self.runs_data[self.runs_data['status'] != 'success']
        if not error_runs.empty:
            error_counts = error_runs['status'].value_counts()
            plt.pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%')
        plt.title('Error Type Distribution')

        # Hourly success rate
        plt.subplot(2, 3, 3)
        hourly_success = self.runs_data.groupby('hour').agg({
            'status': lambda x: (x == 'success').mean() * 100
        })
        plt.bar(hourly_success.index, hourly_success['status'])
        plt.title('Success Rate by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 105)

        # Run volume over time
        plt.subplot(2, 3, 4)
        daily_volume = self.runs_data.groupby('date').size().reset_index(name='count')
        plt.bar(range(len(daily_volume)), daily_volume['count'])
        plt.title('Daily Run Volume')
        plt.xlabel('Days')
        plt.ylabel('Number of Runs')
        plt.xticks(range(0, len(daily_volume), max(1, len(daily_volume)//5)),
                  [d.strftime('%m-%d') for d in daily_volume['date'][::max(1, len(daily_volume)//5)]],
                  rotation=45)

        # Latency vs Success Rate correlation
        plt.subplot(2, 3, 5)
        if 'latency' in self.runs_data.columns:
            latency_success = self.runs_data.dropna(subset=['latency'])
            if not latency_success.empty:
                success_binary = (latency_success['status'] == 'success').astype(int)
                plt.scatter(latency_success['latency'], success_binary, alpha=0.6)
                plt.title('Latency vs Success')
                plt.xlabel('Latency (seconds)')
                plt.ylabel('Success (1) / Failure (0)')
                plt.grid(True, alpha=0.3)

        # Run type distribution
        plt.subplot(2, 3, 6)
        if 'run_type' in self.runs_data.columns:
            type_counts = self.runs_data['run_type'].value_counts()
            plt.bar(range(len(type_counts)), type_counts.values)
            plt.title('Run Type Distribution')
            plt.xlabel('Run Type')
            plt.ylabel('Count')
            plt.xticks(range(len(type_counts)), type_counts.index, rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error analysis plot saved to {save_path}")

        plt.show()

    def plot_performance_heatmap(self, save_path: str = None):
        """Create performance heatmaps"""
        if self.runs_data is None or self.runs_data.empty:
            print("No data available. Fetch runs first.")
            return

        plt.figure(figsize=(15, 10))

        # Hourly performance heatmap
        plt.subplot(2, 2, 1)
        hourly_daily = self.runs_data.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)

        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hourly_daily = hourly_daily.reindex(day_order, fill_value=0)

        sns.heatmap(hourly_daily, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Run Volume Heatmap (Day vs Hour)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')

        # Latency heatmap
        plt.subplot(2, 2, 2)
        if 'latency' in self.runs_data.columns:
            latency_heatmap = self.runs_data.dropna(subset=['latency']).groupby(
                ['day_of_week', 'hour']
            )['latency'].mean().unstack(fill_value=0)
            latency_heatmap = latency_heatmap.reindex(day_order, fill_value=0)

            sns.heatmap(latency_heatmap, annot=True, fmt='.2f', cmap='RdYlBu_r')
            plt.title('Average Latency Heatmap (seconds)')
            plt.xlabel('Hour of Day')
            plt.ylabel('Day of Week')

        # Success rate heatmap
        plt.subplot(2, 2, 3)
        success_heatmap = self.runs_data.groupby(['day_of_week', 'hour']).agg({
            'status': lambda x: (x == 'success').mean() * 100
        }).unstack(fill_value=0)
        success_heatmap = success_heatmap.reindex(day_order, fill_value=0)

        sns.heatmap(success_heatmap, annot=True, fmt='.1f', cmap='RdYlGn')
        plt.title('Success Rate Heatmap (%)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')

        # Token usage heatmap
        plt.subplot(2, 2, 4)
        if 'total_tokens' in self.runs_data.columns:
            token_data = self.runs_data.dropna(subset=['total_tokens'])
            if not token_data.empty:
                token_heatmap = token_data.groupby(['day_of_week', 'hour'])['total_tokens'].mean().unstack(fill_value=0)
                token_heatmap = token_heatmap.reindex(day_order, fill_value=0)

                sns.heatmap(token_heatmap, annot=True, fmt='.0f', cmap='Blues')
                plt.title('Average Token Usage Heatmap')
                plt.xlabel('Hour of Day')
                plt.ylabel('Day of Week')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance heatmap saved to {save_path}")

        plt.show()

    def generate_summary_stats(self):
        """Generate and print summary statistics"""
        if self.runs_data is None or self.runs_data.empty:
            print("No data available. Fetch runs first.")
            return

        print("=" * 60)
        print("LANGSMITH DASHBOARD SUMMARY")
        print("=" * 60)

        total_runs = len(self.runs_data)
        success_rate = (self.runs_data['status'] == 'success').mean() * 100

        print(f"Total Runs: {total_runs}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Date Range: {self.runs_data['start_time'].min().strftime('%Y-%m-%d')} to {self.runs_data['start_time'].max().strftime('%Y-%m-%d')}")

        # Latency stats
        latency_data = self.runs_data.dropna(subset=['latency'])
        if not latency_data.empty:
            print(f"\nLatency Statistics:")
            print(f"  Average: {latency_data['latency'].mean():.2f}s")
            print(f"  Median: {latency_data['latency'].median():.2f}s")
            print(f"  95th Percentile: {latency_data['latency'].quantile(0.95):.2f}s")
            print(f"  Max: {latency_data['latency'].max():.2f}s")

        # Token stats
        token_data = self.runs_data.dropna(subset=['total_tokens'])
        if not token_data.empty:
            print(f"\nToken Usage Statistics:")
            print(f"  Total Tokens Used: {token_data['total_tokens'].sum():,.0f}")
            print(f"  Average per Run: {token_data['total_tokens'].mean():.0f}")
            print(f"  Max per Run: {token_data['total_tokens'].max():.0f}")

        # Error analysis
        error_runs = self.runs_data[self.runs_data['status'] != 'success']
        if not error_runs.empty:
            print(f"\nError Analysis:")
            print(f"  Total Errors: {len(error_runs)}")
            print(f"  Error Rate: {(len(error_runs) / total_runs * 100):.1f}%")
            print(f"  Most Common Error: {error_runs['status'].mode().iloc[0] if not error_runs['status'].mode().empty else 'N/A'}")

        print("=" * 60)

    def create_dashboard(self, project_name: str = None, days_back: int = 30,
                        limit: int = 1000, save_plots: bool = False):
        """
        Create complete dashboard with all plots

        Args:
            project_name: Project to analyze
            days_back: Days of data to fetch
            limit: Maximum runs to fetch
            save_plots: Whether to save plots to files
        """
        print("Creating LangSmith Dashboard...")
        print("=" * 50)

        # Fetch data
        self.fetch_runs(project_name, limit, days_back)

        if self.runs_data is None or self.runs_data.empty:
            print("No data fetched. Please check your project name and API key.")
            return

        # Generate summary
        self.generate_summary_stats()

        # Create plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\nGenerating plots...")

        self.plot_latency_timeline(
            f"latency_dashboard_{timestamp}.png" if save_plots else None
        )

        self.plot_token_usage(
            f"token_dashboard_{timestamp}.png" if save_plots else None
        )

        self.plot_error_analysis(
            f"error_dashboard_{timestamp}.png" if save_plots else None
        )

        self.plot_performance_heatmap(
            f"performance_heatmap_{timestamp}.png" if save_plots else None
        )

        print("\nDashboard generation complete!")


def main():
    """
    Main function to run the dashboard
    Modify the parameters below for your specific use case
    """

    # Configuration - UPDATE THESE VALUES
    API_KEY = "your-langsmith-api-key-here"  # Or set LANGSMITH_API_KEY env var
    PROJECT_NAME = "your-project-name"       # Your LangSmith project name
    DAYS_BACK = 30                          # Days of data to fetch
    LIMIT = 1000                            # Maximum runs to fetch
    SAVE_PLOTS = True                       # Save plots to PNG files

    # Initialize dashboard
    dashboard = LangSmithDashboard(api_key=API_KEY, project_name=PROJECT_NAME)

    # Create complete dashboard
    dashboard.create_dashboard(
        project_name=PROJECT_NAME,
        days_back=DAYS_BACK,
        limit=LIMIT,
        save_plots=SAVE_PLOTS
    )

    # Optional: Access the raw data for custom analysis
    if dashboard.runs_data is not None:
        print(f"\nRaw data available in dashboard.runs_data DataFrame")
        print(f"Shape: {dashboard.runs_data.shape}")
        print(f"Columns: {list(dashboard.runs_data.columns)}")

        # Example custom analysis
        print("\nSample of fetched data:")
        print(dashboard.runs_data[['name', 'start_time', 'latency', 'status', 'total_tokens']].head())


if __name__ == "__main__":
    # Example usage scenarios:

    # Scenario 1: Quick dashboard for default project
    # dashboard = LangSmithDashboard(project_name="my-project")
    # dashboard.create_dashboard()

    # Scenario 2: Custom analysis
    # dashboard = LangSmithDashboard()
    # dashboard.fetch_runs("my-project", limit=500, days_back=7)
    # dashboard.plot_latency_timeline()
    # dashboard.generate_summary_stats()

    # Scenario 3: Multiple projects comparison
    # dashboard = LangSmithDashboard()
    # for project in ["project-1", "project-2", "project-3"]:
    #     print(f"\n--- Analysis for {project} ---")
    #     dashboard.fetch_runs(project, days_back=14)
    #     dashboard.generate_summary_stats()

    # Run main function
    main()


# Installation requirements (run these in your terminal):
"""
pip install langsmith pandas matplotlib seaborn numpy

# Required environment variable:
export LANGSMITH_API_KEY="your-api-key-here"

# Or set it in the script above
"""

# Usage Examples:
"""
# Basic usage:
dashboard = LangSmithDashboard(project_name="my-project")
dashboard.create_dashboard()

# Fetch specific data:
df = dashboard.fetch_runs("my-project", limit=200, days_back=7, status_filter="success")

# Individual plots:
dashboard.plot_latency_timeline(save_path="latency.png")
dashboard.plot_token_usage(save_path="tokens.png")
dashboard.plot_error_analysis(save_path="errors.png")
dashboard.plot_performance_heatmap(save_path="heatmap.png")

# Get summary statistics:
dashboard.generate_summary_stats()
"""
