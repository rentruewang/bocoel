from dash import Dash, Input, Output, no_update, State
from dash.dash_table import DataTable
from dash.dcc import Graph
from dash.html import H1
from numpy.typing import NDArray
from pandas import DataFrame
from plotly.graph_objects import Figure
from dash.html import Div, H3
from math import ceil, floor
from json import dumps

from . import app
from .app import updates
from .reducers import Reducer
from .app import constants


def main(*, debug: bool = True, X: NDArray, reducer: Reducer) -> None:
    #### Initialize app ####
    #TODO: Input Real Data
    data: DataFrame = reducer.process(X)
    data_2: DataFrame = reducer.process(X)
    data_3: DataFrame = reducer.process(X)
    data_4: DataFrame = reducer.process(X)
    Data = [data, data_2, data_3, data_4]


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

    @APP.callback(Output("data-card-1", "children"), [Input("slider", "value")])
    def update_control_text_1(slider_value: float):
        return updates.control_text_1(slider_value=slider_value)

    @APP.callback(Output("data-card-2", "figure"), [Input("CI", "value")])
    def update_control_text_2(slider_value: float) -> Figure:
        return updates.control_text_2(slider_value=slider_value)

    @APP.callback(Output("table-content", "children"), Input("slider", "value"))
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
        [Output("3D-plane", "children")], 
        [Input("slider", "value"), Input("CI", "value"), Input("LLM-dropdown","value"), Input("Corpus-dropdown","value")],
        #[State("3D-plane","children")]
    )
    def update_3D_plot(slider_value: float, ci: float = 0, llm:list=[], corpus:list=[], layout_children:list=[]):
        #TODO: have llm and corpus name follow dataframe
        #print("the type of children is",type(children))
    
        show_1 = "GPT-3" in llm and "Corpus-1" in corpus
        show_2 = "GPT-3" in llm and "Corpus-2" in corpus
        show_3 = "BERT" in llm and "Corpus-1" in corpus
        show_4 = "BERT" in llm and "Corpus-2" in corpus
        show = [show_1,show_2,show_3,show_4]
        count = sum(show)
        
        
        models = ["GPT-3","BERT"]
        corpus = ["corpus-1","corpus-2"]
        names = []
        for m in models:
            for c in corpus:
                names.append(m+" on "+c +"-μ ")
                names.append(m+" on "+c +"-std.dev")

        Data_show = [Data[i] for i in range(len(Data)) if show[i]]
        names_show = [names[i] for i in range(len(Data)*2) if show[floor(i/2)]]


        graph_children = []

        if len(Data_show) != 0:
            graph_children.append(Graph(figure=updates.three_d(slider_value=slider_value, ci=ci, dfs=Data_show, 
                                                           row=ceil(len(Data_show)), col=2, 
                                                           names=names_show)))
        else:
            graph_children.append(Div(children=[H1("Select Models & Corpus...",
                                                   style={"color": constants.FONT_COLOR,
                                                          "justify":"center",
                                                           "align":"center"})],
                                                   style={"width":"1200px","height":"650px",
                                                          "border":"1.5px lightcyan solid",
                                                          "border-style": "dashed"}))
        


        layout_children = [Div(
            children=graph_children,
            style={"width": "100%", "displsy":"inline-block","height": "100%","justify":"center", "align":"center"}
        ),
        ]


        return layout_children

        

    APP.run_server(debug=debug, dev_tools_hot_reload=True)
