import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import requests

# ==== Fetch Metrics ====
url = "http://localhost:8000/langsmith/metrics"
try:
    response = requests.get(url)  # no timeout
    response.raise_for_status()   # raises HTTPError if status != 200
    try:
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Response is not a JSON object")
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        data = {}

    single_metrics = data.get("single_metrics", {"ai_latency_p50": 0, "total_traces": 0, "acceptance_rate": 0, "error_rate": 0})
    df_proc = pd.DataFrame(data.get("processing_trends", []))
    df_errors = pd.DataFrame(data.get("error_trends", []))

except Exception as e:
    print(f"Error fetching metrics: {e}")
    # fallback if request failed
    single_metrics = {"ai_latency_p50": 0, "total_traces": 0, "acceptance_rate": 0, "error_rate": 0}
    df_proc = pd.DataFrame(columns=["date", "p50_latency", "p99_latency"])
    df_errors = pd.DataFrame(columns=["date", "error_count", "error_rate"])



# ==== Dash App ====
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Analytics Dashboard", style={"textAlign": "center"}),

    # Single Metrics
    html.Div([
        html.Div(f"AI Latency (P50): {single_metrics['ai_latency_p50']:.4f}s"),
        html.Div(f"Total Traces: {single_metrics['total_traces']}"),
        html.Div(f"Acceptance Rate: {single_metrics['acceptance_rate']*100:.2f}%"),
        html.Div(f"Error Rate: {single_metrics['error_rate']*100:.2f}%"),
    ], style={"display": "flex", "justifyContent": "space-around", "padding": "10px"}),

    # Processing Trends
    dcc.Graph(
        figure={
            "data": [
                go.Scatter(x=df_proc["date"], y=df_proc["p50_latency"], mode="lines+markers", name="P50 Latency") if not df_proc.empty else None,
                go.Scatter(x=df_proc["date"], y=df_proc["p99_latency"], mode="lines+markers", name="P99 Latency") if not df_proc.empty else None,
            ],
            "layout": go.Layout(title="Processing Time Trends", xaxis={"title": "Date"}, yaxis={"title": "Latency (s)"})
        }
    ),

    # Error Trends
    dcc.Graph(
        figure={
            "data": [
                go.Bar(x=df_errors["date"], y=df_errors["error_count"], name="Error Count", yaxis="y1") if not df_errors.empty else None,
                go.Scatter(x=df_errors["date"], y=df_errors["error_rate"], name="Error Rate", yaxis="y2", mode="lines+markers") if not df_errors.empty else None,
            ],
            "layout": go.Layout(
                title="Error Trends",
                xaxis={"title": "Date"},
                yaxis={"title": "Error Count", "side": "left"},
                yaxis2={"title": "Error Rate", "overlaying": "y", "side": "right"},
                legend={"x": 0, "y": 1.1, "orientation": "h"}
            )
        }
    )
])

if __name__ == "__main__":
    app.run(debug=True)
