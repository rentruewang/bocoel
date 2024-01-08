import abc
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from dash import Dash, Input, Output
from dash.dash_table import DataTable
from dash.dcc import Graph, Slider
from dash.html import H1, H2, H3, B, Div, Img, P
from hiplot import Experiment
from numpy.typing import NDArray
from pandas import DataFrame
from plotly.graph_objects import Figure, Indicator, Scatter, Surface
from scipy import interpolate, stats
from sklearn.decomposition import PCA

# Layout Parameters
BG_COLOR = "#040720"
BLOCK_COLOR = "#0C090A"
TEMPLATE = "plotly_dark"
FONT_COLOR = "LightBlue"


#### Process Data ####
class DataProcessor(Protocol):
    @abc.abstractmethod
    def process(self) -> DataFrame:
        ...


# TODO: Attach real data
class PCAPreprocessor(DataProcessor):
    def __init__(
        self,
        X: NDArray = np.random.rand(100, 512) * 100,
        scores: NDArray = np.random.rand(100),
        size: float = 0.5,
        sample_size: Sequence = tuple(range(1, 101)),
        desc: Sequence = (),
        algo: str = "PCA",
    ):
        self._X = X
        self._scores = scores
        self._size = size
        self._sample_size = sample_size
        self._algo = algo
        self._description = (
            desc if desc else ["Fake prompt number {}".format(i) for i in range(1, 101)]
        )

    def _dim_reduce(self, X: NDArray) -> NDArray:
        func: Callable
        # match algo:
        #     case "PCA":
        #         func = PCA(n_components=2, svd_solver='full').fit_transform
        #     case _:
        #         raise ValueError("Not supported.")
        func = PCA(n_components=2, svd_solver="full").fit_transform
        return func(X)

    def process(self) -> DataFrame:
        df = DataFrame()

        df["size"] = self._size
        df["std"] = np.std(self._scores)
        df["sample_size"] = self._sample_size
        df["scores"] = self._scores
        df["Description"] = self._description

        x_reduced = self._dim_reduce(self._X)
        df["x"] = x_reduced[:, 0]
        df["y"] = x_reduced[:, 1]

        return df


def process_data(processor: DataProcessor) -> DataFrame:
    df = processor.process()
    return df


#### DIV GENERATORS ####
"""Generate html.div and elements with function, arranged in app.layout"""


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return Div(
        id="description-card",
        children=[
            Div(
                id="intro",
                children="Explore Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models.",
                style={
                    "background-color": BLOCK_COLOR,
                    "color": FONT_COLOR,
                    "margin": "10px 5px",
                },
            ),
            Div(
                id="control-card",
                children=[
                    B("Select Sample Size"),
                    Slider(
                        min=0,
                        max=100,
                        marks={i: "{}".format(i) for i in range(0, 110, 10)},
                        value=50,
                        id="slider",
                    ),
                    B("Select Confidence Interval"),
                    Slider(
                        min=0,
                        max=1,
                        marks={i / 100: "{}%".format(i) for i in range(0, 105, 5)},
                        value=0.95,
                        id="CI",
                    ),
                    Div(
                        [P(id="data_card_1"), Graph(id="data_card_2")],
                    ),
                ],
                style={
                    "background-color": BLOCK_COLOR,
                    "color": FONT_COLOR,
                    "margin": "5px 10px",
                },
            ),
        ],
        style={"width": "50%"},
    )


def generate_2D():
    return Div(
        id="2D",
        children=[
            Graph(id="2D-plane"),
        ],
        style={"width": "50%", "display": "inline-block"},
    )


def generate_table():
    return Div(
        id="table",
        children=[
            H3("Prompt table"),
            P(id="table_out"),
            DataTable(
                id="table",
            ),
        ],
        style={"width": "40%", "display": "inline-block"},
    )


def generate_3D():
    return Div(
        id="3D-plot",
        children=[Graph(id="3D-plane")],
        style={"width": "60%", "display": "inline-block"},
    )


# Unused
def generate_splines():
    return Div(
        id="xy-splines",
        children=[
            Div(
                id="splines",
                children=[
                    Div(
                        Graph(id="x-splines", style={"height": "300px"}),
                        style={"width": "49%", "display": "inline-block"},
                    ),
                    Div(
                        Graph(id="y-splines", style={"height": "300px"}),
                        style={"width": "49%", "display": "inline-block"},
                    ),
                ],
                style={"display": "flex"},
            ),
        ],
    )


#### Initialize app ####
processor = PCAPreprocessor()
data: DataFrame = process_data(processor)

app = Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
        }
    ],
)
app.title = (
    "Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models"
)
app.config["suppress_callback_exceptions"] = True

server = app.server

hi_data = [
    {"dropout": 0.1, "lr": 0.001, "loss": 10.0, "optimizer": "SGD"},
    {"dropout": 0.15, "lr": 0.01, "loss": 3.5, "optimizer": "Adam"},
    {"dropout": 0.3, "lr": 0.1, "loss": 4.5, "optimizer": "Adam"},
]
hi_html = None
# hi_html = Experiment.to_html()

#### app Layout ####
app.layout = Div(
    id="app-container",
    children=[
        # Banner
        Div(
            id="banner",
            className="banner",
            children=[
                H1(
                    "BoCoEl:Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models",
                    style={"color": FONT_COLOR},
                ),
                Img(src="banner.png", alt="banner"),
            ],
            style={"display": "flex", "gap": "20px"},
        ),
        # Upper column
        Div(
            id="left-column",
            className="three columns",
            children=[description_card(), generate_table()],
            style={
                "display": "flex",
                "gap": "20px",
                "align-items": "flex-stretch",
                "width": "100%",
            },
        ),
        # Lower column
        Div(H2("ParBayesian Optimization in Action", style={"color": FONT_COLOR})),
        Div(
            id="right-column",
            className="eight columns",
            children=[
                Div(
                    children=[generate_2D()] + [generate_3D()],
                    style={
                        "display": "flex",
                        "gap": "20px",
                        "align-items": "flex-top",
                        "width": "100%",
                    },
                ),
            ],
        ),
        Div(hi_html),
    ],
    style={"background-color": BG_COLOR},
)


#### CALLBACKS ####
"""Reads control input and update div generators"""


@app.callback(Output("data_card_1", "children"), [Input("slider", "value")])
def update_control_text_1(slider_value):
    return P(
        "#Prompt size: {}".format(slider_value),
        style={"font-size": "16px", "align": "center"},
    )


@app.callback(Output("data_card_2", "figure"), [Input("CI", "value")])
def update_control_text_2(slider_value) -> Figure:
    fig = Figure(
        Indicator(
            mode="gauge+number",
            value=slider_value,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"font": {"color": FONT_COLOR}},
            title={"text": "Confidence Interval", "font": {"color": FONT_COLOR}},
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

    fig.update_layout(paper_bgcolor=BLOCK_COLOR)
    return fig


@app.callback(Output("table_out", "children"), Input("slider", "value"))
def update_table(slider) -> DataTable:
    sep = slider
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


@app.callback(Output("2D-plane", "figure"), Input("slider", "value"))
def update_2D(slider) -> Figure:
    sep = slider
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
    fig.update_layout(title="Prompt Embedding 2D Mapping", template=TEMPLATE)
    return fig


# Unused
@app.callback(Output("x-splines", "figure"), Input("slider", "value"))
def update_X_splines(slider) -> Figure:
    sep = slider
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
@app.callback(Output("y-splines", "figure"), Input("slider", "value"))
def update_Y_splines(slider) -> Figure:
    sep = slider
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


@app.callback(
    Output("3D-plane", "figure"), [Input("slider", "value"), Input("CI", "value")]
)
def update_3D_plot(slider, CI) -> Figure:
    df = data.copy()
    sep = slider
    if not CI:
        CI = 0.95
    CI = float(CI)
    print(CI)
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
    Z_up = Z + stats.norm.ppf(CI) * np.std(z) * 10
    Z_low = Z - stats.norm.ppf(CI) * np.std(z) * 10

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
        template=TEMPLATE,
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
