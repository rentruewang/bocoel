# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from dash.dash_table import DataTable
from dash.dcc import Checklist, Graph, Slider
from dash.html import H1, H2, H3, B, Div, Img, Li, P, Ul

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
                children=[
                    P(
                        "Explore Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models. \
                            Bayesian optimization is a sequential design strategy for global optimization of black-box functions that does not assume any functional forms.\
                             It is usually employed to optimize expensive-to-evaluate functions."
                    ),
                    H3("Members"),
                    Ul([Li(author) for author in constants.AUTHORS]),
                ],
                style={
                    "background-color": constants.BLOCK_COLOR,
                    "color": constants.FONT_COLOR,
                    "margin": "10px 5px",
                    "width": "25%",
                    "display": "inline-block",
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
                        [P(id="data-card-1"), Graph(id="data-card-2")],
                    ),
                ],
                style={
                    "background-color": constants.BLOCK_COLOR,
                    "color": constants.FONT_COLOR,
                    "margin": "5px 10px",
                    "width": "25%",
                    "display": "inline-block",
                },
            ),
            generate_table(),
        ],
        style={"display": "flex"},
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
            H3("Prompts 2D Mapping", style={"color": constants.FONT_COLOR}),
            P(id="table-content"),
            DataTable(
                id="prompt-table",
            ),
            Graph(id="2D-plane"),
        ],
        style={"width": "40%", "display": "inline-block"},
    )


def generate_3D():
    return Div(
        id="3D-plot",
        children=[
            # Div(children=[Graph(id="3D-plane")]),
            Div(
                [],
                id="3D-plane",
                # style={"width": "25%", "display": "flex"}
            ),
        ],
        style={"width": "100%", "display": "flex"},
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
                id="upper-column",
                className="two columns",
                children=[description_card()],
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
                id="lower-column",
                className="two columns",
                children=[
                    Div(
                        children=[
                            Div(
                                style={"display": "flex"},
                                children=[
                                    Div(
                                        children=[
                                            H3(
                                                "Select Models:",
                                                style={
                                                    "border-bottom": "0.5px lightcyan solid",
                                                    "margin-left": "5px",
                                                },
                                            ),
                                            Checklist(
                                                ["GPT-3", "BERT"],
                                                ["GPT-3"],
                                                # inline=True,
                                                id="LLM-dropdown",
                                                style={
                                                    "background-color": constants.BLOCK_COLOR,
                                                    "color": constants.FONT_COLOR,
                                                    "width": "90%",
                                                    "margin": "0 auto",
                                                    "display": "inline-block",
                                                },
                                            ),
                                        ],
                                        style={
                                            "background-color": constants.BLOCK_COLOR,
                                            "color": constants.FONT_COLOR,
                                            "border-right": "0.5px lightcyan solid",
                                        },
                                    ),
                                    Div(
                                        children=[
                                            H3(
                                                "Select Corpus:",
                                                style={
                                                    "border-bottom": "0.5px lightcyan solid",
                                                    "margin-left": "5px",
                                                },
                                            ),
                                            Checklist(
                                                ["Corpus-1", "Corpus-2"],
                                                ["Corpus-1"],
                                                id="Corpus-dropdown",
                                                # inline=True,
                                                style={
                                                    "background-color": constants.BLOCK_COLOR,
                                                    "color": constants.FONT_COLOR,
                                                    "width": "90%",
                                                    "margin": "0 auto",
                                                    "display": "inline-block",
                                                },
                                            ),
                                        ],
                                        style={
                                            "background-color": constants.BLOCK_COLOR,
                                            "color": constants.FONT_COLOR,
                                        },
                                    ),
                                    Div(
                                        children=[generate_3D()],
                                        style={
                                            "display": "flex",
                                            "gap": "20px",
                                            "align-items": "flex-top",
                                            "width": "100%",
                                        },
                                    ),
                                ],
                            ),
                        ],
                        style={
                            "gap": "20px",
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
