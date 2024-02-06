import numpy as np
from dash.dash_table import DataTable
from dash.html import P
from pandas import DataFrame
from plotly.graph_objects import Figure, Indicator, Scatter, Surface, Contour
from scipy import interpolate, stats
from plotly import subplots

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
    #df = df.tail(10)

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
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'},
        fixed_rows={'headers': True},
        style_data_conditional=[                
                {
                    "if": {"state": "selected"},              # 'active' | 'selected'
                    "backgroundColor": "rgba(0, 231, 233, 0.8)",
                    "border": "1px solid blue",
                },
            ]
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
    fig.update_layout(title="On 2D Plane", template=constants.TEMPLATE)
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
def three_d(slider_value: float, ci: float, dfs: list, row:int, col:int, names:list) -> Figure:
    if not ci:
        ci = 0.95
    ci = float(ci)
    #print(ci)
    #print(sep)

    # Initialize X,Y,Z
    X_list = []
    Y_list = []
    Z_list = []
    Z_up_list = []
    Z_low_list = []

    # Process Dataframes
    for df in dfs:
        df = df[df["sample_size"] <= slider_value]
        x = df["x"]
        sorted_x = sorted(list(x))
        y = df["y"]
        sorted_y = sorted(list(y),reverse=True)
        z = df["scores"]
        z_std = df["std"]


        xi = np.linspace(x.min(), x.max(), df.shape[0])
        yi = np.linspace(y.min(), y.max(), df.shape[0])

        X, Y = np.meshgrid(xi, yi)

        Z = interpolate.griddata((x, y), z, (X, Y), method="cubic", fill_value=0)
        Z_std = interpolate.griddata((x, y), z_std, (X, Y), method="cubic", fill_value=0)
        # Z_up = Z + stats.norm.ppf(ci) * np.std(z) * 10
        # Z_low = Z - stats.norm.ppf(ci) * np.std(z) * 10

        # X_list.append(xi)
        # Y_list.append(yi)
        # Z_list.append(Z)
        # Z_up_list.append(Z_up)
        # Z_low_list.append(Z_low)
    
    specs = [[{"type": "contour"},{"type": "contour"}]]*row

    fig = subplots.make_subplots(
        rows=row, cols=col, subplot_titles=names,
        specs=specs, shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.1
    )

    # process hover text
    hover_texts = []
    hover_std = []
    for i in range(df.shape[0]):
        score_temp = list(z)[i]
        std_temp = list(df["std"])[i]
        text_temp = list(df["Description"])[i]
        hover_texts.append(f"Score: {score_temp:.3f} || Prompt: {text_temp}")
        hover_std.append(f"Std: {std_temp:.3f} || Prompt: {text_temp}")


        
    for i in range(1,row+1):
        for j in range(1,col+1,2):
            #print(i,j)

            cbarlocs = [.23, .78]

            # avg. contour
            fig.add_trace(
                Contour(
                    z = Z, 
                    x = sorted_x,
                    y = sorted_y,
                    name="μ surface",
                    colorbar=dict(len=0.3, x=cbarlocs[0], y=1.05, orientation='h', thickness=15),          
                ),
                row=i, col=j
            )

            # std.dev contour
            fig.add_trace(
                Contour(
                    z = Z_std, 
                    x = sorted_x, 
                    y = sorted_y,
                    name="std.dev surface",
                    colorbar=dict(len=0.3, x=cbarlocs[1], y=1.05, orientation='h', thickness=15), 
                ),
                row=i, col=j+1
            )

            # sampled corpus points
            fig.add_trace(Scatter(
                    x=df['x'],
                    y=df['y'],
                    text=hover_texts,
                    hovertemplate=
                    "X: %{x:.3f}<br>" +
                    "Y: %{y:.3f}<br>" +
                    #"Score: %{text}" +
                    "%{text}" +
                    "<extra></extra>",
                    mode="markers",
                    marker=dict(color='Green')
                    #marker_size=,
                    ),
                row=i, col=j
            )

            
            # sampled corpus std.dev
            fig.add_trace(Scatter(
                    x=df['x'],
                    y=df['y'],
                    text=hover_std,
                    hovertemplate=
                    "X: %{x:.3f}<br>" +
                    "Y: %{y:.3f}<br>" +
                    #"Score: %{text}" +
                    "%{text}" +
                    "<extra></extra>",
                    mode="markers",
                    marker=dict(color='LightSkyBlue'),
                    #marker_size=,
                    ),
                row=i, col=j+1
            )

            """ fig.add_trace(Surface(
                z=Z_list[i+j-2],
                x=X_list[i+j-2],
                y=Y_list[i+j-2],
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
                name="optimized surface"
                ), row=i, col=j
            )
            fig.add_trace(Surface(
                z=Z_up_list[i+j-2], x=X_list[i+j-2], y=Y_list[i+j-2], showscale=False, opacity=0.9, name="upper bound"
                ), row=i, col=j
            )
            fig.add_trace(Surface(
                z=Z_low_list[i+j-2], x=X_list[i+j-2], y=Y_list[i+j-2], showscale=False, opacity=0.9, name="lower bound"
                ), row=i, col=j
            ) """
            
    # fig LAYOUT
    fig.update_layout(
        title="Plot Title",
        showlegend=False,
        # xaxis_title="X Axis Title",
        # yaxis_title="Y Axis Title",
        width=1200, height=580*row,
        template="plotly_dark"
    )




    return fig
