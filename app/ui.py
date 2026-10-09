"""Shared UI blocks for the guided demo app."""

import data_io
import streamlit as st


def get_ticker() -> str:
    """Resolve the active company (sidebar selector when several are available)."""
    tickers = data_io.available_tickers()
    if not tickers:
        st.warning(
            "No data found. Run the pipeline for a company, then "
            "`uv run python scripts/build_demo_bundle.py <TICKER>`."
        )
        st.stop()
    if len(tickers) == 1:
        st.sidebar.markdown(f"**Company:** {tickers[0]}")
        return tickers[0]
    return st.sidebar.selectbox("Company", tickers, key="ticker")


def step_header(step: int, total: int, title: str, subtitle: str) -> None:
    """Consistent header for the guided steps."""
    st.markdown(f"##### Step {step} of {total}")
    st.title(title)
    st.markdown(subtitle)
    st.progress(step / total)


def link(page: str, label: str, icon: str = "➡️") -> None:
    """Page link that degrades gracefully outside the multipage runtime."""
    try:
        st.page_link(page, label=label, icon=icon)
    except Exception:
        st.markdown(f"{icon} **{label}** — use the sidebar to navigate.")


def next_step(page: str, label: str) -> None:
    st.divider()
    link(page, f"Next: {label} →")
