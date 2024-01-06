import typing

import numpy as np
from dash import Dash, Input, Output
from dash.dash_table import DataTable
from dash.dcc import Graph, Slider
from dash.html import H1, H2, H3, H4, B, Div, Img, P
from numpy import random
from pandas import DataFrame
from plotly.graph_objects import Figure, Scatter, Surface
from scipy import interpolate
from sklearn.decomposition import PCA


def _dummy_process():
    # Candidates ((embeddings): X-axis
    X = random.rand(100, 512) * 100
    # Scores: y-axis
    score = random.rand(100)
    # Description
    D = [f"Fake prompt embedding {p}" for p in range(1, 101)]

    df = DataFrame({"scores": score, "Description": D})
    df["size"] = 0.5
    df["sample_size"] = [i for i in range(1, 101)]
    df["std"] = np.std(list(df["scores"]))
    return df, X


def init_process():
    state, X = _dummy_process()

    pca = PCA(n_components=2, svd_solver="full")
    x_reduced = pca.fit_transform(X)
    state["x"] = x_reduced[:, 0]
    state["y"] = x_reduced[:, 1]

    return state


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return Div(
        id="description-card",
        children=[
            H3("Welcome to the visualization dashboard of BoCoEl"),
            Div(
                id="intro",
                children="Explore Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models.",
            ),
        ],
    )


def generate_2D():
    return Div(
        id="2D",
        children=[
            Graph(id="2D-plane"),
        ],
        # style={'width': '49%', 'display': 'inline-block'}
    )


def generate_control():
    """
    :return: A Div containing controls for graphs.
    """
    return Div(
        id="control-card",
        children=[
            B("Select Sample Size"),
            Slider(
                min=0,
                max=100,
                marks={i: "{}".format(i) for i in range(0, 100, 10)},
                value=50,
                id="slider",
            ),
            P(id="slider-selection"),
        ],
        style={"width": "79%", "display": "inline-block"},
    )


def generate_table():
    return Div(
        id="table",
        children=[
            H4("Prompt table"),
            P(id="table_out"),
            DataTable(
                id="table",
            ),
        ],
        # style={'width': '49%', 'display': 'inline-block'}
    )


def generate_3D():
    return Div(
        id="3D-plot",
        children=[Graph(id="3D-plane")],
        style={"width": "49%", "display": "inline-block"},
    )


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
DATA = init_process()

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


app.layout = Div(
    id="app-container",
    children=[
        # Banner
        Div(
            id="banner",
            className="banner",
            children=[
                H1("BoCoEl"),
                Img(src=app.get_asset_url("plotly_logo.png")),
            ],
            style={"display": "flex", "gap": "20px"},
        ),
        # Upper column
        Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control(), generate_table()],
            style={"display": "flex", "gap": "20px", "align-items": "flex-top"},
        ),
        Div(
            children=[generate_splines()],
        ),
        # Lower column
        Div(H2("ParBayesian Optimization in Action")),
        Div(
            id="right-column",
            className="eight columns",
            children=[
                Div(
                    children=[generate_2D()] + [generate_3D()],
                    style={"display": "flex", "gap": "20px", "align-items": "flex-top"},
                ),
            ],
        ),
    ],
)


@app.callback(Output("slider-selection", "children"), Input("slider", "value"))
def update_output(value):
    return "Selecting {} samples.".format(value)


@app.callback(Output("table_out", "children"), Input("slider", "value"))
def update_table(slider):
    sep = slider

    df: DataFrame = DATA.copy()
    df = typing.cast(DataFrame, df[df["sample_size"] <= sep])
    df = typing.cast(DataFrame, df[["scores", "Description"]])
    df["scores"] = df["scores"].apply(lambda x: round(x, 3))
    df = df.tail(3)

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
def update_2D(slider):
    sep = slider

    df: DataFrame = DATA.copy()
    df = typing.cast(DataFrame, df[df["sample_size"] <= sep])
    df: DataFrame = df.sort_values(by="x")

    fig = Figure()
    fig.add_trace(
        Scatter(
            x=df["x"], y=df["y"], mode="markers", name="markers", text=df["Description"]
        )
    )
    fig.update_layout(title="Prompt Embedding 2D Mapping")
    return fig


@app.callback(Output("x-splines", "figure"), Input("slider", "value"))
def update_X_splines(slider):
    sep = slider

    df = DATA.copy()
    df = typing.cast(DataFrame, df[df["sample_size"] <= sep])
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


@app.callback(Output("y-splines", "figure"), Input("slider", "value"))
def update_Y_splines(slider):
    sep = slider
    df: DataFrame = DATA.copy()
    df = typing.cast(DataFrame, df[df["sample_size"] <= sep])
    df: DataFrame = df.sort_values(by="y")

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


@app.callback(Output("3D-plane", "figure"), Input("slider", "value"))
def update_3D_plot(slider):
    sep = slider

    df = DATA.copy()
    df = df[df["sample_size"] <= sep]

    x = df["x"]
    y = df["y"]
    z = df["scores"]

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)

    X, Y = np.meshgrid(xi, yi)

    Z = interpolate.griddata((x, y), z, (X, Y), method="cubic")

    fig = Figure(data=[Surface(z=Z, x=xi, y=yi)])
    fig.update_layout(
        title="Optimization surface",
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=65, r=50, b=65, t=90),
    )

    fig = Figure(
        Surface(
            contours={
                "x": {
                    "show": True,
                    "start": 1.5,
                    "end": 2,
                    "size": 0.04,
                    "color": "white",
                },
                "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05},
            },
            x=xi,
            y=yi,
            z=Z,
        )
    )
    fig.update_layout(
        scene={
            "xaxis": {"nticks": 20},
            "yaxis": {"nticks": 4},
            "zaxis": {"nticks": 4},
            "camera_eye": {"x": 0, "y": -1, "z": 0.5},
            "aspectratio": {"x": 0.8, "y": 0.8, "z": 0.2},
        },
        title="Optimization Plane",
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
