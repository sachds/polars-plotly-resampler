"""Dash helper functions for constructing a file seelector
"""

__author__ = "Jonas Van Der Donckt"

import itertools
from pathlib import Path
from typing import Dict, List

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from functional import seq
import dash_design_kit as ddk
import dash_mantine_components as dmc

from pathlib import Path


def _update_file_widget(folder):
    if folder is None:
        return []

    arrow_files = [
        file.name
        for file in Path(folder).iterdir()
        if file.is_file() and file.name.endswith(".arrow")
    ]

    sorted_files = sorted(set(arrow_files))

    return [{"label": filename, "value": filename} for filename in sorted_files]


def _register_selection_callbacks(app, ids=None):
    if ids is None:
        ids = [""]

    for id in ids:
        app.callback(
            Output(f"file-selector{id}", "options"),
            [Input(f"folder-selector{id}", "value")],
        )(_update_file_widget)


def _register_selection_callbacks1(app, ids=None):
    if ids is None:
        ids = [""]

    for id in ids:
        app.callback(
            Output(f"file-selector+{id}", "options"),
            [Input(f"folder-selector+{id}", "value")],
        )(_update_file_widget)


def multiple_folder_file_selector(
    app, name_folders_list: List[Dict[str, dict]], multi=True
) -> dbc.Card:
    """Constructs a folder user date selector

    Creates a `dbc.Card` component which can be

    Parameters
    ----------
    app:
        The dash application.
    name_folders_list:List[Dict[str, Union[Path, str]]]
         A dict with key, the display-key and values the correspondign path.

    Returns
    -------
    A bootstrap card component
    """
    selector = dbc.Card(
        [
            dbc.Card(
                [
                    dbc.Col(
                        [
                            dbc.Label("folder"),
                            dcc.Dropdown(
                                id=f"folder-selector{i}",
                                options=[
                                    {"label": l, "value": str(f["folder"])}
                                    for (l, f) in name_folders.items()
                                ],
                                clearable=False,
                            ),
                            dbc.Label("file"),
                            dcc.Dropdown(
                                id=f"file-selector{i}",
                                options=[],
                                clearable=True,
                                multi=multi,
                            ),
                            html.Br(),
                        ]
                    ),
                ]
            )
            for i, name_folders in enumerate(name_folders_list, 1)
        ]
        + [
            dbc.Card(
                dbc.Col(
                    [
                        html.Br(),
                        dbc.Button(
                            "create figure",
                            id="plot-button",
                            color="primary",
                        ),
                        # dmc.Space(h=10),
                        # dbc.Button("pull new data"),
                    ],
                    style={"textAlign": "center"},
                ),
            )
        ],
        body=True,
    )

    _register_selection_callbacks(app=app, ids=range(1, len(name_folders_list) + 1))
    return selector


def multiple_folder_file_selector1(
    app, name_folders_list: List[Dict[str, dict]], multi=True
) -> dbc.Card:
    """Constructs a folder user date selector

    Creates a `dbc.Card` component which can be

    Parameters
    ----------
    app:
        The dash application.
    name_folders_list:List[Dict[str, Union[Path, str]]]
         A dict with key, the display-key and values the correspondign path.

    Returns
    -------
    A bootstrap card component
    """
    selector = dbc.Card(
        [
            dbc.Card(
                [
                    dbc.Col(
                        [
                            dbc.Label("folder"),
                            dcc.Dropdown(
                                id=f"folder-selector+{i}",
                                options=[
                                    {"label": l, "value": str(f["folder"])}
                                    for (l, f) in name_folders.items()
                                ],
                                clearable=False,
                            ),
                            dbc.Label("file"),
                            dcc.Dropdown(
                                id=f"file-selector+{i}",
                                options=[],
                                clearable=True,
                                multi=multi,
                            ),
                            html.Br(),
                        ]
                    ),
                ]
            )
            for i, name_folders in enumerate(name_folders_list, 1)
        ]
        + [
            dbc.Card(
                dbc.Col(
                    [
                        html.Br(),
                        dbc.Button(
                            "create figure",
                            id="plot-button",
                            color="primary",
                        ),
                    ],
                    style={"textAlign": "center"},
                ),
            )
        ],
        body=True,
    )

    _register_selection_callbacks1(app=app, ids=range(1, len(name_folders_list) + 1))
    return selector


def get_selector_states(n: int) -> List[State]:
    """Return a list of all the folder-file selector fields, which are used as State

    Parameters
    ----------
    n: int
        The number of folder selectors

    """
    return list(
        itertools.chain.from_iterable(
            [
                (
                    State(f"folder-selector{i}", "value"),
                    State(f"file-selector{i}", "value"),
                )
                for i in range(1, n + 1)
            ]
        )
    )


def get_selector_states1(n: int) -> List[State]:
    """Return a list of all the folder-file selector fields, which are used as State

    Parameters
    ----------
    n: int
        The number of folder selectors

    """
    return list(
        itertools.chain.from_iterable(
            [
                (
                    State(f"folder-selector+{i}", "value"),
                    State(f"file-selector+{i}", "value"),
                )
                for i in range(1, n + 1)
            ]
        )
    )
