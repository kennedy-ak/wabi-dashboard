import os
import asyncio
from datetime import datetime, timedelta
from statistics import quantiles
from dotenv import load_dotenv
from langsmith.async_client import AsyncClient

# Load environment variables
print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Loading environment variables...")
load_dotenv()
api_key = os.getenv("LANGSMITH_API_KEY")
project_name = "wabi-fastapi"
print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Environment variables loaded. API Key and project name set.")

async def fetch_langsmith_metrics_async():
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Starting async fetch_langsmith_metrics function...")
    try:
        # Use AsyncClient for non-blocking API calls
        async with AsyncClient(api_key=api_key) as client:
            start_time_filter = datetime.now() - timedelta(days=1)
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Time range start: {start_time_filter}")

            runs = []
            offset = 0
            batch_size = 100

            while True:
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Fetching batch at offset {offset}...")

                batch = []
                async for run in client.list_runs(
                    project_name=project_name,
                    start_time=start_time_filter,
                    limit=batch_size,
                    offset=offset
                ):
                    batch.append(run)

                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Retrieved {len(batch)} runs.")
                runs.extend(batch)

                if len(batch) < batch_size:
                    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] No more runs to fetch.")
                    break
                offset += batch_size


            if not runs:
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] No runs found.")
                return None

            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Processing {len(runs)} runs...")
            run_data, latencies, daily = [], [], {}

            for i, run in enumerate(runs, start=1):
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Processing run {i}/{len(runs)}...")
                start, end = getattr(run, "start_time", None), getattr(run, "end_time", None)
                latency = getattr(run, "latency", None)
                if latency is None and start and end:
                    latency = (end - start).total_seconds()
                if latency is not None:
                    latencies.append(latency)

                is_error = getattr(run, "error", None) is not None
                date_key = start.strftime("%Y-%m-%d") if start else datetime.now().strftime("%Y-%m-%d")
                if date_key not in daily:
                    daily[date_key] = {"runs": 0, "errors": 0}
                daily[date_key]["runs"] += 1
                if is_error:
                    daily[date_key]["errors"] += 1

                run_data.append({"date": date_key, "latency": latency or 0, "is_error": is_error})

            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Computing metrics...")
            total_traces = len(runs)
            ai_latency_p50 = quantiles(latencies, n=100)[49] if latencies else 0
            error_rate = sum(r["is_error"] for r in run_data) / total_traces if total_traces else 0
            acceptance_rate = 0.187  # placeholder

            processing_trends, error_trends = [], []
            for date, data in daily.items():
                total, errs = data["runs"], data["errors"]
                daily_latencies = [r["latency"] for r in run_data if r["date"] == date and r["latency"] > 0]
                p50 = quantiles(daily_latencies, n=100)[49] if daily_latencies else 0
                p99 = quantiles(daily_latencies, n=100)[98] if daily_latencies else 0
                processing_trends.append({"date": date, "p50_latency": p50, "p99_latency": p99})
                error_trends.append({"date": date, "error_count": errs, "error_rate": errs / total if total else 0})

            metrics = {
                "single_metrics": {
                    "ai_latency_p50": ai_latency_p50,
                    "total_traces": total_traces,
                    "acceptance_rate": acceptance_rate,
                    "error_rate": error_rate,
                },
                "processing_trends": processing_trends,
                "error_trends": error_trends,
            }
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Metrics ready.")
            return metrics

    except Exception as e:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S'}] Error: {e}")
        return None

if __name__ == "__main__":
    result = asyncio.run(fetch_langsmith_metrics_async())
    print(result)
