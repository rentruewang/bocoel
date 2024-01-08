import numpy as np
from dash.dash_table import DataTable
from dash.html import P
from pandas import DataFrame
from plotly.graph_objects import Figure, Indicator, Scatter, Surface
from scipy import interpolate, stats

from . import constants, utils


def control_text_1(slider_value: float):
    return P(
        f"#Prompt size: {slider_value}",
        style={"font-size": "16px", "align": "center"},
    )


def control_text_2(slider_value: float) -> Figure:
    fig = Figure(
        Indicator(
            mode="gauge+number",
            value=slider_value,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"font": {"color": constants.FONT_COLOR}},
            title={
                "text": "Confidence Interval",
                "font": {"color": constants.FONT_COLOR},
            },
            gauge={
                "axis": {"range": [None, 1]},
                "steps": [
                    {"range": [0, 0.9], "color": "lightgray"},
                    {"range": [0.9, 0.95], "color": "gray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 0.95,
                },
            },
        )
    )

    fig.update_layout(paper_bgcolor=constants.BLOCK_COLOR)
    return fig


@utils.copy_inputs
def table(slider_value: float, df: DataFrame) -> DataTable:
    df = df[df["sample_size"] <= slider_value]
    df = df[["scores", "Description"]]
    df["scores"] = df["scores"].apply(lambda x: round(x, 3))
    df = df.tail(10)

    table = DataTable(
        id="table",
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("records"),
        style_cell=dict(textAlign="left"),
        style_header=dict(backgroundColor="paleturquoise"),
        style_data=dict(backgroundColor="lavender"),
        style_cell_conditional=[
            {"if": {"column_id": "scores"}, "width": "15%"},
            {"if": {"column_id": "Description"}, "width": "85%"},
        ],
    )
    return table


@utils.copy_inputs
def two_d(slider_value: float, df: DataFrame) -> Figure:
    df = df[df["sample_size"] <= slider_value]
    df = df.sort_values(by="x")

    fig = Figure()
    # fig.add_trace(Scatter(x=df['x'], y=df['scores'],
    #                     mode='lines',
    #                     name='lines'))
    # fig.add_trace(Scatter(x=df['y'], y=df['scores'],
    #                     mode='lines+markers',
    #                     name='lines+markers'))
    fig.add_trace(
        Scatter(
            x=df["x"], y=df["y"], mode="markers", name="markers", text=df["Description"]
        )
    )
    fig.update_layout(title="Prompt Embedding 2D Mapping", template=constants.TEMPLATE)
    return fig


@utils.copy_inputs
def x_splines(slider_value: float, df: DataFrame) -> Figure:
    df = df[df["sample_size"] <= slider_value]
    df = df.sort_values(by="x")

    score_upper = [(y + np.std(df["scores"])) for y in df["scores"]]
    score_lower = [(y - np.std(df["scores"])) for y in df["scores"]]
    score_lower = score_lower[::-1]

    fig = Figure()

    # standard deviation area
    fig.add_traces(
        Scatter(
            x=df["x"],
            y=score_upper + score_lower,
            fill="tozerox",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            name="x-std",
        )
    )

    # line trace
    fig.add_traces(
        Scatter(
            x=df["x"],
            y=df["scores"],
            line=dict(color="blue", width=2.5),
            mode="lines",
            name="x-line",
        )
    )
    fig.update_layout(title="X-axis Optimization Spline")
    return fig


@utils.copy_inputs
def y_splines(slider_value: float, df: DataFrame) -> Figure:
    df = df[df["sample_size"] <= slider_value]
    df = df.sort_values(by="y")

    score_upper = [(y + np.std(df["scores"])) for y in df["scores"]]
    score_lower = [(y - np.std(df["scores"])) for y in df["scores"]]
    score_lower = score_lower[::-1]

    fig = Figure()

    # standard deviation area
    fig.add_traces(
        Scatter(
            x=df["y"],
            y=score_upper + score_lower,
            fill="tozerox",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            name="x-std",
        )
    )

    # line trace
    fig.add_traces(
        Scatter(
            x=df["y"],
            y=df["scores"],
            line=dict(color="blue", width=2.5),
            mode="lines",
            name="x-line",
        )
    )
    fig.update_layout(title="Y-axis Optimization Spline")
    return fig


@utils.copy_inputs
def three_d(slider_value: float, ci: float, df: DataFrame) -> Figure:
    if not ci:
        ci = 0.95
    ci = float(ci)
    print(ci)
    # print(sep)
    df = df[df["sample_size"] <= slider_value]
    # print(df.head())
    x = df["x"]
    y = df["y"]
    z = df["scores"]

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)

    X, Y = np.meshgrid(xi, yi)

    Z = interpolate.griddata((x, y), z, (X, Y), method="cubic", fill_value=0)
    Z_up = Z + stats.norm.ppf(ci) * np.std(z) * 10
    Z_low = Z - stats.norm.ppf(ci) * np.std(z) * 10

    fig = Figure(
        data=[
            Surface(
                z=Z,
                x=xi,
                y=yi,
                contours={
                    "x": {
                        "show": True,
                        "start": 1.5,
                        "end": 2,
                        "size": 0.04,
                        "color": "white",
                    },
                    "z": {"show": True, "start": 0.5, "end": 10, "size": 0.05},
                },
                name="optimized surface",
            ),
            Surface(
                z=Z_up, x=xi, y=yi, showscale=False, opacity=0.9, name="upper bound"
            ),
            Surface(
                z=Z_low, x=xi, y=yi, showscale=False, opacity=0.9, name="lower bound"
            ),
        ]
    )
    fig.update_layout(
        title="Optimization Surface",
        autosize=True,
        width=800,
        height=500,
        margin=dict(l=65, r=50, b=65, t=90),
    )

    fig.update_layout(
        scene={
            "xaxis": {"nticks": 20},
            "yaxis": {"nticks": 4},
            "zaxis": {"nticks": 4},
            "camera_eye": {"x": 0, "y": -1, "z": 0.5},
            "aspectratio": {"x": 0.8, "y": 0.8, "z": 0.2},
        },
        template=constants.TEMPLATE,
    )
    return fig
