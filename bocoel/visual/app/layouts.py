from dash.dash_table import DataTable
from dash.dcc import Graph, Slider
from dash.html import H1, H2, H3, B, Div, Img, P

from . import constants


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
                    "background-color": constants.BLOCK_COLOR,
                    "color": constants.FONT_COLOR,
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
                    "background-color": constants.BLOCK_COLOR,
                    "color": constants.FONT_COLOR,
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


#### app Layout ####
def layout():
    return Div(
        id="app-container",
        children=[
            # Banner
            Div(
                id="banner",
                className="banner",
                children=[
                    H1(
                        "BoCoEl:Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models",
                        style={"color": constants.FONT_COLOR},
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
            Div(
                H2(
                    "ParBayesian Optimization in Action",
                    style={"color": constants.FONT_COLOR},
                )
            ),
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
            # FIXME: Hiplot not working yet.
            # Div(hi_html),
        ],
        style={"background-color": constants.BG_COLOR},
    )
