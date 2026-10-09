"""Shared palette and Plotly layout defaults for the Signal Intelligence app.

Colors follow a validated accessible palette: categorical hues are assigned by
entity (never cycled), sequential is one hue light->dark, and text/grid use
dedicated ink tokens rather than series colors.
"""

# Chart chrome (light mode — the app pins the light theme)
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots (fixed order — color follows the entity)
BLUE = "#2a78d6"
GREEN = "#008300"
MAGENTA = "#e87ba4"
YELLOW = "#eda100"
ORANGE = "#eb6834"
RED = "#e34948"

# Entity assignments, consistent across every page
CORPUS_PRESENCE_COLORS = {
    "triangulated": BLUE,
    "insider_only": GREEN,
    "outsider_only": MAGENTA,
}
CORPUS_COLORS = {
    "management": BLUE,
    "press": GREEN,
}
POLARITY_COLORS = {1: BLUE, -1: RED}  # diverging pair: blue <-> red

PRESENCE_LABELS = {
    "triangulated": "Triangulated (management + press)",
    "insider_only": "Insider only (management)",
    "outsider_only": "Outsider only (press)",
}


def plotly_layout(**overrides) -> dict:
    """Base Plotly layout: recessive grid, ink-token text, no chart junk."""
    layout = {
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {
            "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
            "color": INK_PRIMARY,
            "size": 13,
        },
        "xaxis": {
            "gridcolor": GRIDLINE,
            "linecolor": BASELINE,
            "zerolinecolor": BASELINE,
            "tickfont": {"color": INK_MUTED},
            "title_font": {"color": INK_SECONDARY},
        },
        "yaxis": {
            "gridcolor": GRIDLINE,
            "linecolor": BASELINE,
            "zerolinecolor": BASELINE,
            "tickfont": {"color": INK_MUTED},
            "title_font": {"color": INK_SECONDARY},
        },
        "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
        "hovermode": "closest",
        "legend": {"font": {"color": INK_SECONDARY}},
    }
    layout.update(overrides)
    return layout
