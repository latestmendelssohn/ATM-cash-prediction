"""Small Gradio demo for the bundled synthetic ATM dataset.

The Space deliberately skips the Gemini/RAG layer. It demonstrates the part of the
project that is safe to publish without secrets: a 14-day Holt-Winters forecast,
its interval, and the cash load recommended from measured cycle-total error.
"""
from __future__ import annotations

import gradio as gr

from demo_logic import ATMS, run_forecast


def run_gradio(atm_id, horizon, service_level):
    summary, rows = run_forecast(atm_id, horizon, service_level)
    return summary, [[r["date"], r["forecast (Rs)"], r["lower (Rs)"], r["upper (Rs)"]]
                     for r in rows]


with gr.Blocks(title="ATM cash forecasting demo") as demo:
    gr.Markdown(
        """# ATM cash forecasting demo

Choose one of the five synthetic ATMs. The app forecasts daily cash demand and turns
the forecast into a replenishment amount. The 14-day default is the project's main
example, but you can inspect a shorter or longer window too.
"""
    )
    with gr.Row():
        atm = gr.Dropdown(ATMS, value=ATMS[0], label="ATM")
        horizon = gr.Slider(1, 30, value=14, step=1, label="Forecast horizon (days)")
        service = gr.Slider(0.80, 0.99, value=0.95, step=0.01, label="Service level")
    run = gr.Button("Run forecast", variant="primary")
    summary = gr.Markdown()
    table = gr.Dataframe(
        headers=["date", "forecast (Rs)", "lower (Rs)", "upper (Rs)"],
        datatype=["str", "number", "number", "number"],
        label="Daily forecast and interval",
        wrap=True,
    )
    run.click(run_gradio, [atm, horizon, service], [summary, table])
    demo.load(run_gradio, [atm, horizon, service], [summary, table])


if __name__ == "__main__":
    demo.launch()
