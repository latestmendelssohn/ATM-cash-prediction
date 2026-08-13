"""Streamlit entry point for the ATM forecasting demo."""
from __future__ import annotations

import streamlit as st

from demo_logic import ATMS, run_forecast

st.set_page_config(page_title="ATM cash forecasting", page_icon="🏧", layout="wide")
st.title("ATM cash forecasting demo")
st.caption(
    "Public 2009-2010 ATM dataset. The source CSV does not identify a currency, "
    "so numbers are shown in the source unit."
)

with st.sidebar:
    st.header("Forecast settings")
    atm_id = st.selectbox("ATM", ATMS)
    horizon = st.slider("Forecast horizon (days)", 1, 30, 14)
    service_level = st.slider("Service level", 0.80, 0.99, 0.95, 0.01)
    run = st.button("Run forecast", type="primary", use_container_width=True)

st.markdown(
    "Choose an ATM and run a Holt-Winters forecast. The cash plan uses the measured "
    "spread of cycle-total forecast errors for safety stock."
)

if run:
    with st.spinner("Fitting the model and measuring cycle uncertainty..."):
        summary, rows = run_forecast(atm_id, horizon, service_level)
    st.markdown(summary)
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.info("Choose settings in the sidebar, then click **Run forecast**.")
