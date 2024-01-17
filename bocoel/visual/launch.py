from dash import Dash, Input, Output
from dash.dash_table import DataTable
from numpy.typing import NDArray
from pandas import DataFrame
from plotly.graph_objects import Figure

from . import app
from .app import updates
from .reducers import Reducer


def main(*, debug: bool = True, X: NDArray, reducer: Reducer) -> None:
    #### Initialize app ####
    data: DataFrame = reducer.process(X)

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
    def update_2D_plot(slider_value: float) -> Figure:
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

    APP.run_server(debug=debug)
