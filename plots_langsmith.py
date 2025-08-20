import os
from datetime import datetime, timedelta
from langsmith import Client
from dotenv import load_dotenv
from statistics import quantiles
from flask import Flask, jsonify

# Load environment variables
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading environment variables...")
load_dotenv()
api_key = os.getenv("LANGSMITH_API_KEY")  # Set in .env: LANGSMITH_API_KEY=ls__your_key
project_name = "wabi-fastapi"  # Your project name
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Environment variables loaded. API Key and project name set.")

# Initialize LangSmith client
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing LangSmith client...")
client = Client(api_key=api_key)
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LangSmith client initialized successfully.")

# Fetch and process data from TracingProject with pagination
def fetch_langsmith_metrics():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting fetch_langsmith_metrics function...")
    try:
        # Fetch runs from the last 5 days
        one_week_ago = datetime.now() - timedelta(days=0.1)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Setting time range to last 5 days. Start time: {one_week_ago}")
        runs = []
        offset = 0
        batch_size = 10  # Smaller batch size for efficient pagination

        while True:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching batch at offset {offset} with limit {batch_size}...")
            batch = client.list_runs(
                project_name=project_name,
                start_time=one_week_ago,
                limit=batch_size,
                offset=offset
            )
            batch = list(batch)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetched {len(batch)} runs in current batch.")
            runs.extend(batch)
            if len(batch) < batch_size:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No more batches to fetch. Total runs: {len(runs)}")
                break
            offset += batch_size

        if not runs:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No runs found in the last 5 days.")
            return None

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total runs fetched: {len(runs)}. Starting data processing...")
        # Process runs into a timeline
        run_data = []
        latencies = []
        error_counts = []
        total_runs_per_day = {}

        for i, run in enumerate(runs):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing run {i+1}/{len(runs)}...")
            # Calculate latency (in seconds) if not directly available
            latency = getattr(run, "latency", None)
            if latency is None and hasattr(run, "start_time") and hasattr(run, "end_time"):
                latency = (run.end_time - run.start_time).total_seconds()
            if latency is not None:
                latencies.append(latency)

            # Check for errors
            is_error = getattr(run, "error", None) is not None

            # Group by date
            date_key = run.start_time.strftime("%Y-%m-%d")
            if date_key not in total_runs_per_day:
                total_runs_per_day[date_key] = {"runs": 0, "errors": 0}
            total_runs_per_day[date_key]["runs"] += 1
            if is_error:
                total_runs_per_day[date_key]["errors"] += 1

            run_data.append({
                "date": date_key,
                "latency": latency if latency is not None else 0,
                "is_error": is_error
            })

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Data processing completed. Total unique dates: {len(total_runs_per_day)}")

        # Compute single metrics
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Computing single metrics...")
        total_traces = len(runs)
        ai_latency_p50 = quantiles(latencies, n=100)[49] if latencies else 0  # Median (p50) in seconds
        error_rate = sum(1 for run in runs if getattr(run, "error", None) is not None) / len(runs) if runs else 0
        acceptance_rate = 0.187  # Placeholder; adjust based on your definition (e.g., feedback stats)

        # Compute daily metrics for plots
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Computing daily trends...")
        processing_trends = []
        error_trends = []
        for i, (date, data) in enumerate(total_runs_per_day.items()):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing day {i+1}/{len(total_runs_per_day)}: {date}")
            total_runs = data["runs"]
            error_count = data["errors"]
            error_rate_daily = error_count / total_runs if total_runs > 0 else 0
            daily_latencies = [r["latency"] for r in run_data if r["date"] == date and r["latency"] > 0]
            p50_latency = quantiles(daily_latencies, n=100)[49] if daily_latencies else 0
            p99_latency = quantiles(daily_latencies, n=100)[98] if daily_latencies else 0

            processing_trends.append({
                "date": date,
                "p50_latency": p50_latency,
                "p99_latency": p99_latency
            })
            error_trends.append({
                "date": date,
                "error_count": error_count,
                "error_rate": error_rate_daily
            })

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Daily trends computation completed. Processing trends: {len(processing_trends)}, Error trends: {len(error_trends)}")

        # Return all metrics
        metrics = {
            "single_metrics": {
                "ai_latency_p50": ai_latency_p50,
                "total_traces": total_traces,
                "acceptance_rate": acceptance_rate,
                "error_rate": error_rate
            },
            "processing_trends": processing_trends,
            "error_trends": error_trends
        }
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Metrics computed successfully. Returning metrics...")
        return metrics

    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error fetching data: {str(e)}")
        return None

# Flask app to serve metrics
app = Flask(__name__)

@app.route("/api/wabi-fastapi-metrics", methods=["GET"])
def get_metrics():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Received GET request for /api/wabi-fastapi-metrics")
    metrics = fetch_langsmith_metrics()
    if metrics is None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No metrics available, returning error response")
        return jsonify({"error": "Failed to fetch data"}), 500
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Metrics fetched, returning JSON response")
    return jsonify(metrics)

# Main execution
if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting main execution...")
    metrics = fetch_langsmith_metrics()
    if metrics:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Metrics fetched successfully: {metrics}")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No data to return")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Flask server on port 5000...")
    app.run(debug=True, port=5000)
