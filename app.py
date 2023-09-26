"""Dash file parquet visualization app example with a coarse and fine-grained view.

In this use case, we have dropdowns which allows end-users to select multiple
parquet files, which are visualized using FigureResampler after clicking on a button.

There a two graphs displayed; a coarse and a dynamic graph. Interactions with the
coarse graph will affect the dynamic graph it's shown range. Note that the autosize
of the coarse graph is not linked.

TODO: add an rectangle on the coarse graph

"""

__author__ = "Jonas Van Der Donckt"

from pathlib import Path
from typing import List
import dash_design_kit as ddk
import dash_mantine_components as dmc
import dash
import plotly_express as px
import os
from dotenv import load_dotenv


# Load the .env file
load_dotenv()


from typing import List
from uuid import uuid4


import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, no_update, MATCH
from dash_extensions.enrich import (
    DashProxy,
    ServersideOutput,
    ServersideOutputTransform,
    Trigger,
)
from trace_updater import TraceUpdater
from utils.callback_helpers import (
    get_selector_states,
    multiple_folder_file_selector,
    multiple_folder_file_selector1,
    get_selector_states1,
)
from utils.graph_construction import visualize_multiple_files

from plotly_resampler import FigureResampler
from sqlalchemy import create_engine
import pandas as pd
import time


# Database connection

token = os.getenv("DATABRICKS_TOKEN")
host = os.getenv("DATABRICKS_HOST")
path = os.getenv("DATABRICKS_PATH")
engine_url = f"databricks://token:{token}@{host}/?http_path={path}&catalog=main&schema=information_schema"
engine = create_engine(engine_url)
# tables_stmt = (
#     f"SELECT * FROM main.resamplerdata.auto_iot_bronze_sensors_optimized LIMIT 100000;"
# )
# start_time = time.time()
# tables_in_db = pd.read_sql_query(tables_stmt, engine)
# end_time = time.time()
# elapsed_time = end_time - start_time

# # Fetch data from the database
# import polars as pl
import gc
from sqlalchemy import create_engine
import os

directory = "backend-data"
if not os.path.exists(directory):
    os.makedirs(directory)


# SQL statements (unchanged)

engine_temp_stmt = "SELECT Timestamp, EngineTemperature_C FROM main.resamplerdata.auto_iot_data_bronze_sensors LIMIT 50000;"
oil_pressure_stmt = "SELECT Timestamp, OilPressure_psi FROM main.resamplerdata.auto_iot_data_bronze_sensors LIMIT 50000;"
speed_stmt = "SELECT Timestamp, Speed_kmh FROM main.resamplerdata.auto_iot_data_bronze_sensors LIMIT 50000;"
tire_pressure_stmt = "SELECT Timestamp, TirePressure_psi FROM main.resamplerdata.auto_iot_data_bronze_sensors LIMIT 50000;"
battery_voltage_stmt = "SELECT Timestamp, BatteryVoltage_V FROM main.resamplerdata.auto_iot_data_bronze_sensors LIMIT 50000;"


# Read data from SQL queries into Polars dataframes
query_start_time = time.time()
engine_temp_df = pd.read_sql(engine_temp_stmt, engine)
oil_pressure_df = pd.read_sql(oil_pressure_stmt, engine)
speed_df = pd.read_sql(speed_stmt, engine)
tire_pressure_df = pd.read_sql(tire_pressure_stmt, engine)
battery_voltage_df = pd.read_sql(battery_voltage_stmt, engine)
query_end_time = time.time()

print(f"Query time: {query_end_time - query_start_time} seconds")

# Save dataframes as Parquet files
parquet_start_time = time.time()

engine_temp_df.to_parquet("backend-data/engine_temp_df.parquet")
oil_pressure_df.to_parquet("backend-data/oil_pressure_df.parquet")
speed_df.to_parquet("backend-data/speed_df.parquet")
tire_pressure_df.to_parquet("backend-data/tire_pressure_df.parquet")
battery_voltage_df.to_parquet("backend-data/battery_voltage_df.parquet")

battery_voltage_df.to_parquet(f"backend-data/battery_voltage_df.parquet")

parquet_end_time = time.time()

print(f"Parquet time: {parquet_end_time - parquet_start_time} seconds")

del engine_temp_df
del oil_pressure_df
del speed_df
del tire_pressure_df
del battery_voltage_df

# Collect garbage to reclaim memory
gc.collect()


# --------------------------------------Globals ---------------------------------------
app = DashProxy(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUX],
    transforms=[ServersideOutputTransform()],
)

server = app.server


# --------- File selection configurations ---------
name_folder_list = [
    {
        # the key-string below is the title which will be shown in the dash app
        "auto iot data": {
            "folder": Path("./.backend-data").parent.parent.joinpath("backend-data")
        },
    },
    # NOTE: A new item om this level creates a new file-selector card.
    # { "PC data": { "folder": Path("/home/jonas/data/wesad/empatica/") } }
    # TODO: change the folder path above to a location where you have some
    # `.parquet` files stored on your machine.
]


app.layout = ddk.App(
    [
        ddk.Header(
            [
                ddk.Logo(src=app.get_asset_url("plotly_logo.png")),
                ddk.Title("Plotly Resampler Dynamic Demo"),
                ddk.Menu(
                    [
                        ddk.CollapsibleMenu(
                            title="Resampler",
                            children=[
                                dcc.Link("Coarse & Dynamic", href="/coarse-dynamic"),
                                dcc.Link("MultiFile", href="/multifile"),
                                dcc.Link("All-in-One", href="/aio"),
                                # dcc.Link("Borrowings", href="/borrowings"),
                            ],
                        ),
                        dcc.Link("Datashader", href="/datashader"),
                        # dcc.Link("Explorer", href="/explorer"),
                        dcc.Link("Settings", href="/settings"),
                    ]
                ),
            ]
        ),
        ddk.Card(
            children=[
                dcc.Location(id="url", refresh=False),
                html.Div(
                    id="page-content",
                ),
            ],
        ),
    ]
)


multifile_layout = dbc.Container(
    [
        dbc.Container(
            html.H1("Comparative View of Multiple Sensors - 10000000 Points per Trace"),
            style={"textAlign": "center"},
        ),
        html.Hr(),
        dbc.Row(
            [
                # Add file selection layout (+ assign callbacks)
                dbc.Col(multiple_folder_file_selector1(app, name_folder_list), md=2),
                # Add the graph, the dcc.Store (for serialization) and the
                # TraceUpdater (for efficient data updating) components
                dbc.Col(
                    [
                        dmc.LoadingOverlay(
                            ddk.Graph(id="graph-id1", figure=go.Figure()),
                            overlayOpacity=0.25,
                            overlayColor="#292943",
                            loaderProps={"color": "#F5F7FA"},
                        ),
                        dcc.Store(id="store1"),
                        TraceUpdater(id="trace-updater1", gdID="graph-id1"),
                    ],
                    md=10,
                ),
            ],
            align="center",
        ),
    ],
    fluid=True,
)

coarse_dynamic_layout = dbc.Container(
    [
        dbc.Container(
            html.H1("Fine Grained Control Using Coarse Graph- 10000000_Points "),
            style={"textAlign": "center"},
        ),
        html.Hr(),
        dbc.Row(
            [
                # Add file selection layout (+ assign callbacks)
                dbc.Col(
                    multiple_folder_file_selector(app, name_folder_list, multi=False),
                    md=2,
                ),
                # Add the graphs, the dcc.Store (for serialization) and the
                # TraceUpdater (for efficient data updating) components
                dbc.Col(
                    [
                        # The coarse graph whose updates will fetch data for the
                        dmc.LoadingOverlay(
                            ddk.Graph(
                                id="coarse-graph",
                                figure=go.Figure(),
                                config={"modeBarButtonsToAdd": ["drawrect"]},
                            ),
                            overlayOpacity=0.25,
                            overlayColor="#292943",
                            loaderProps={"color": "#F5F7FA"},
                        ),
                        html.Br(),
                        dmc.LoadingOverlay(
                            ddk.Graph(
                                id="plotly-resampler-graph",
                                figure=go.Figure(),
                            ),
                            overlayOpacity=0.25,
                            overlayColor="#292943",
                            loaderProps={"color": "#F5F7FA"},
                        ),
                        dcc.Store(id="store"),
                        TraceUpdater(
                            id="trace-updater",
                            gdID="plotly-resampler-graph",
                        ),
                    ],
                    md=10,
                ),
            ],
            align="center",
        ),
    ],
    fluid=True,
)

aio_layout = html.Div(
    [
        html.Div(children=[html.Button("Load Data", id="add-chart", n_clicks=0)]),
        html.Div(id="container", children=[]),
    ]
)


# ------------------------------------ DASH logic -------------------------------------
# --------- graph construction logic + callback ---------
@app.callback(
    [
        Output("coarse-graph", "figure"),
        Output("plotly-resampler-graph", "figure"),
        ServersideOutput("store", "data"),
    ],
    [Input("plot-button", "n_clicks"), *get_selector_states(len(name_folder_list))],
    prevent_initial_call=True,
)
def construct_plot_graph(n_clicks, *folder_list):
    it = iter(folder_list)
    file_list: List[Path] = []
    for folder, files in zip(it, it):
        if not all((folder, files)):
            continue
        else:
            files = [files] if not isinstance(files, list) else file_list
            for file in files:
                file_list.append((Path(folder).joinpath(file)))

    ctx = callback_context
    if len(ctx.triggered) and "plot-button" in ctx.triggered[0]["prop_id"]:
        if len(file_list):
            # Create two graphs, a dynamic plotly-resampler graph and a coarse graph
            dynamic_fig: FigureResampler = visualize_multiple_files(file_list)
            coarse_fig: go.Figure = go.Figure(
                FigureResampler(dynamic_fig, default_n_shown_samples=3_000)
            )

            coarse_fig.update_layout(title="<b>coarse view</b>", height=250)
            coarse_fig.update_layout(margin=dict(l=0, r=0, b=0, t=40, pad=10))
            coarse_fig.update_layout(showlegend=False)
            coarse_fig._config = coarse_fig._config.update(
                {"modeBarButtonsToAdd": ["drawrect"]}
            )

            dynamic_fig._global_n_shown_samples = 1000
            dynamic_fig.update_layout(title="<b>dynamic view<b>", height=450)
            dynamic_fig.update_layout(margin=dict(l=0, r=0, b=40, t=40, pad=10))
            dynamic_fig.update_layout(
                legend=dict(
                    orientation="h", y=-0.11, xanchor="right", x=1, font_size=18
                )
            )

            return coarse_fig, dynamic_fig, dynamic_fig
    else:
        return no_update


# Register the graph update callbacks to the layout
@app.callback(
    Output("trace-updater", "updateData"),
    Input("coarse-graph", "relayoutData"),
    Input("plotly-resampler-graph", "relayoutData"),
    State("store", "data"),
    prevent_initial_call=True,
)
def update_dynamic_fig(coarse_grained_relayout, fine_grained_relayout, fr_fig):
    if fr_fig is None:  # When the figure does not exist -> do nothing
        return no_update

    ctx = callback_context
    trigger_id = ctx.triggered[0].get("prop_id", "").split(".")[0]

    if trigger_id == "plotly-resampler-graph":
        return fr_fig.construct_update_data(fine_grained_relayout)
    elif trigger_id == "coarse-graph":
        return fr_fig.construct_update_data(coarse_grained_relayout)

    return no_update


# ------------------------------------ DASH logic -------------------------------------
@app.callback(
    [Output("graph-id1", "figure"), ServersideOutput("store1", "data")],
    [Input("plot-button", "n_clicks"), *get_selector_states1(len(name_folder_list))],
    prevent_initial_call=True,
)
def plot_graph(n_clicks, *folder_list):
    it = iter(folder_list)
    file_list: List[Path] = []
    for folder, files in zip(it, it):
        if not all((folder, files)):
            continue
        else:
            for file in files:
                file_list.append((Path(folder).joinpath(file)))

    ctx = callback_context
    if len(ctx.triggered) and "plot-button" in ctx.triggered[0]["prop_id"]:
        if len(file_list):
            fig: FigureResampler = visualize_multiple_files(file_list)
            return fig, fig
    else:
        return no_update


# --------- Figure update callback ---------
@app.callback(
    Output("trace-updater1", "updateData"),
    Input("graph-id1", "relayoutData"),
    State("store1", "data"),  # The server side cached FigureResampler per session
    prevent_initial_call=True,
)
def update_fig(relayoutdata, fig):
    if fig is None:
        return no_update
    return fig.construct_update_data(relayoutdata)


# ------------------------------------ DASH logic -------------------------------------
# This method adds the needed components to the front-end, but does not yet contain the
# FigureResampler graph construction logic.
@app.callback(
    Output("container", "children"),
    Input("add-chart", "n_clicks"),
    State("container", "children"),
    prevent_initial_call=True,
)
def add_graph_div(n_clicks: int, div_children: List[html.Div]):
    uid = str(uuid4())
    new_child = html.Div(
        children=[
            # The graph and its needed components to serialize and update efficiently
            # Note: we also add a dcc.Store component, which will be used to link the
            #       server side cached FigureResampler object
            dmc.LoadingOverlay(
                ddk.Graph(
                    id={"type": "dynamic-graph", "index": uid},
                    figure=go.Figure(),
                ),
                overlayOpacity=0.25,
                overlayColor="#292943",
                loaderProps={"color": "#F5F7FA"},
            ),
            dcc.Store(id={"type": "store", "index": uid}),
            TraceUpdater(id={"type": "dynamic-updater", "index": uid}, gdID=f"{uid}"),
            # This dcc.Interval components makes sure that the `construct_display_graph`
            # callback is fired once after these components are added to the session
            # its front-end
            dcc.Interval(
                id={"type": "interval", "index": uid}, max_intervals=1, interval=1
            ),
        ],
    )
    div_children.append(new_child)
    return div_children


# Specify the path to the folder containing Parquet files
parquet_folder_path = "backend-data"
# Get a list of all Parquet file paths in the folder
parquet_file_paths = [
    os.path.join(parquet_folder_path, file)
    for file in os.listdir(parquet_folder_path)
    if file.endswith(".parquet")
]

# Read all Parquet files into a single pandas DataFrame
dfs = [pd.read_parquet(file) for file in parquet_file_paths]
start_time = time.time()  # Record start time

combined_df = pd.concat(dfs, ignore_index=True)


# This method constructs the FigureResampler graph and caches it on the server side
@app.callback(
    ServersideOutput({"type": "store", "index": MATCH}, "data"),
    Output({"type": "dynamic-graph", "index": MATCH}, "figure"),
    State("add-chart", "n_clicks"),
    Trigger({"type": "interval", "index": MATCH}, "n_intervals"),
    prevent_initial_call=True,
)
def construct_display_graph(n_clicks, n_intervals) -> FigureResampler:
    fig = FigureResampler(go.Figure(), default_n_shown_samples=2_000)

    # Figure construction logic based on a state variable, in our case n_clicks
    # sigma = n_clicks * 1e-6
    fig.add_trace(
        dict(name="Temperature (C)"),
        hf_x=combined_df["Timestamp"],
        hf_y=combined_df["EngineTemperature_C"],
    )
    fig.add_trace(
        dict(name="Oil Pressure (psi)"),
        hf_x=combined_df["Timestamp"],
        hf_y=combined_df["OilPressure_psi"],
    )
    fig.add_trace(
        dict(name="Speed (kmh)"),
        hf_x=combined_df["Timestamp"],
        hf_y=combined_df["Speed_kmh"],
    )
    fig.add_trace(
        dict(name="Tire Pressure (psi)"),
        hf_x=combined_df["Timestamp"],
        hf_y=combined_df["TirePressure_psi"],
    )
    fig.add_trace(
        dict(name="Battery Voltage (V)"),
        hf_x=combined_df["Timestamp"],
        hf_y=combined_df["BatteryVoltage_V"],
    )
    fig.update_layout(
        title=f"<b>Temperature, Oil Pressure, Tire Pressure, and Battery Voltage - {n_clicks}</b>",
        title_x=0.5,
    )

    return fig, fig


@app.callback(
    Output({"type": "dynamic-updater", "index": MATCH}, "updateData"),
    Input({"type": "dynamic-graph", "index": MATCH}, "relayoutData"),
    State({"type": "store", "index": MATCH}, "data"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata: dict, fig: FigureResampler):
    if fig is not None:
        return fig.construct_update_data(relayoutdata)
    return no_update


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    if pathname == "/coarse-dynamic":
        return coarse_dynamic_layout
    elif pathname == "/multifile":
        return multifile_layout
    elif pathname == "/aio":
        return aio_layout
    else:
        return html.Div("404")


# --------------------------------- Running the app ---------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=9023)
