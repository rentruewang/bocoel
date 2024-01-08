import numpy as np
from dash import Dash, Input, Output
from dash.dash_table import DataTable
from dash.html import P
from pandas import DataFrame
from plotly.graph_objects import Figure, Indicator, Scatter, Surface
from scipy import interpolate, stats

from . import app
from .processors import PCAPreprocessor, Processor


def process_data(processor: Processor) -> DataFrame:
    # TODO: Use real data.
    X = np.random.rand(100, 512) * 100

    df = processor.process(X)
    return df


#### Initialize app ####
processor = PCAPreprocessor()
data: DataFrame = process_data(processor)

dash_app = Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
        }
    ],
)
dash_app.title = (
    "Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models"
)
dash_app.config["suppress_callback_exceptions"] = True

server = dash_app.server

hi_data = [
    {"dropout": 0.1, "lr": 0.001, "loss": 10.0, "optimizer": "SGD"},
    {"dropout": 0.15, "lr": 0.01, "loss": 3.5, "optimizer": "Adam"},
    {"dropout": 0.3, "lr": 0.1, "loss": 4.5, "optimizer": "Adam"},
]
hi_html = None
# hi_html = Experiment.to_html()

dash_app.layout = app.layout()

#### CALLBACKS ####
"""Reads control input and update div generators"""


@dash_app.callback(Output("data_card_1", "children"), [Input("slider", "value")])
def update_control_text_1(slider_value: float):
    return P(
        "#Prompt size: {}".format(slider_value),
        style={"font-size": "16px", "align": "center"},
    )


@dash_app.callback(Output("data_card_2", "figure"), [Input("CI", "value")])
def update_control_text_2(slider_value: float) -> Figure:
    fig = Figure(
        Indicator(
            mode="gauge+number",
            value=slider_value,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"font": {"color": app.FONT_COLOR}},
            title={"text": "Confidence Interval", "font": {"color": app.FONT_COLOR}},
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

    fig.update_layout(paper_bgcolor=app.BLOCK_COLOR)
    return fig


@dash_app.callback(Output("table_out", "children"), Input("slider", "value"))
def update_table(slider_values: float) -> DataTable:
    sep = slider_values
    df = data.copy()
    df = df[df["sample_size"] <= sep]
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


@dash_app.callback(Output("2D-plane", "figure"), Input("slider", "value"))
def update_2D(slider_values: float) -> Figure:
    sep = slider_values
    df = data.copy()
    df = df[df["sample_size"] <= sep]
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
    fig.update_layout(title="Prompt Embedding 2D Mapping", template=app.TEMPLATE)
    return fig


# Unused
@dash_app.callback(Output("x-splines", "figure"), Input("slider", "value"))
def update_X_splines(slider_values: float) -> Figure:
    sep = slider_values
    df = data.copy()
    df = df[df["sample_size"] <= sep]
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


# Unused
@dash_app.callback(Output("y-splines", "figure"), Input("slider", "value"))
def update_Y_splines(slider_values: float) -> Figure:
    sep = slider_values
    df = data.copy()
    df = df[df["sample_size"] <= sep]
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


@dash_app.callback(
    Output("3D-plane", "figure"), [Input("slider", "value"), Input("CI", "value")]
)
def update_3D_plot(slider_values: float, ci: float = 0) -> Figure:
    df = data.copy()
    sep = slider_values
    if not ci:
        ci = 0.95
    ci = float(ci)
    print(ci)
    # print(sep)
    df = df[df["sample_size"] <= sep]
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
        template=app.TEMPLATE,
    )
    return fig


dash_app.run_server(debug=True)
