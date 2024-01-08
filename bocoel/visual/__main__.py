import numpy as np
from dash import Dash, Input, Output
from dash.dash_table import DataTable
from pandas import DataFrame
from plotly.graph_objects import Figure

from . import app
from .app import updates
from .reducers import PCAReducer, Reducer


def process_data(reducer: Reducer) -> DataFrame:
    # TODO: Use real data.
    X = np.random.rand(100, 512) * 100

    df = reducer.process(X)
    return df


#### Initialize app ####
processor = PCAReducer()
data: DataFrame = process_data(processor)

APP = Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
        }
    ],
)
APP.title = (
    "Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models"
)
APP.config["suppress_callback_exceptions"] = True


# hi_data = [
#     {"dropout": 0.1, "lr": 0.001, "loss": 10.0, "optimizer": "SGD"},
#     {"dropout": 0.15, "lr": 0.01, "loss": 3.5, "optimizer": "Adam"},
#     {"dropout": 0.3, "lr": 0.1, "loss": 4.5, "optimizer": "Adam"},
# ]
# hi_html = None
# hi_html = Experiment.to_html()

APP.layout = app.layout()

#### CALLBACKS ####
"""Reads control input and update div generators"""


@APP.callback(Output("data_card_1", "children"), [Input("slider", "value")])
def update_control_text_1(slider_value: float):
    return updates.control_text_1(slider_value=slider_value)


@APP.callback(Output("data_card_2", "figure"), [Input("CI", "value")])
def update_control_text_2(slider_value: float) -> Figure:
    return updates.control_text_2(slider_value=slider_value)


@APP.callback(Output("table_out", "children"), Input("slider", "value"))
def update_table(slider_value: float) -> DataTable:
    return updates.table(slider_value=slider_value, df=data)


@APP.callback(Output("2D-plane", "figure"), Input("slider", "value"))
def update_2D(slider_value: float) -> Figure:
    return updates.two_d(slider_value=slider_value, df=data)


# Unused
@APP.callback(Output("x-splines", "figure"), Input("slider", "value"))
def update_X_splines(slider_value: float) -> Figure:
    return updates.x_splines(slider_value=slider_value, df=data)


# Unused
@APP.callback(Output("y-splines", "figure"), Input("slider", "value"))
def update_Y_splines(slider_value: float) -> Figure:
    return updates.y_splines(slider_value=slider_value, df=data)


@APP.callback(
    Output("3D-plane", "figure"), [Input("slider", "value"), Input("CI", "value")]
)
def update_3D_plot(slider_value: float, ci: float = 0) -> Figure:
    return updates.three_d(slider_value=slider_value, ci=ci, df=data)


APP.run_server(debug=True)
