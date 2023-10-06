import dash
from dash.dependencies import Input, Output, State
import dash_daq as daq
from dash_daq import DarkThemeProvider
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from scipy import signal
from preprocessor import Preprocessor
from tensorflow.keras.models import load_model
import joblib

# Load and preprocess DF
preprocessor = Preprocessor()
#df = preprocessor.df
#summary = preprocessor.summary(df)
#date = preprocessor.last_recorded_date
#df = df [['Date', 'CV1', 'MV1', 'MV2', 'MV3', 'MV4']]

# Load ML tools
model_path = 'ml_tools/P075_24h_3h_CNNLSTM.h5'
scaler1_path = 'ml_tools/scaler1.pkl'
scaler2_path = 'ml_tools/scaler2.pkl'
       
model = load_model(model_path)

with open(scaler1_path, 'rb') as file1:
    sc1 = joblib.load(file1)

with open(scaler2_path, 'rb') as file2:
    sc2 = joblib.load(file2)

# Define function to predict 3h
def predict_3h(df, date, features):
    #date = df[['Date']].max()
    X_test = df[features][:1440]
    X_test_sc = sc1.transform(X_test)
    X_test_sc = X_test_sc[np.newaxis, :, :]
    y = model.predict(X_test_sc)
    y_pred_reshaped = y.reshape(-1, y.shape[-2])
    y_pred = sc2.inverse_transform(y_pred_reshaped)

    date_rng = pd.date_range(start=date, periods=180, freq='1min')
    # Create a DataFrame with the array and the DatetimeIndex
    df_pred = pd.DataFrame(y_pred.T, index=date_rng)

    # Rename the columns to 'date' and 'prediction'
    df_pred = df_pred.reset_index()
    df_pred.columns = ['Date', 'prediction']

    return df_pred


# Setting up Dash app
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
app.config["suppress_callback_exceptions"] = True
app.title = "Farm Dashboard"
server = app.server

# Define light and dark thems colors
marker_color = {"dark": "#FFD60A", "light": "#FFB703"}
axis_color = {"dark": "#f2f5fa", "light": "#2a3f5f"}

theme = {
    "dark": False,
    "primary": "red",
    "secondary": "#86D9C7",
    "detail": "#1F7A8C",
}

# Function to display the date and time of the last record
def last_record_div():
    _, date = preprocessor.request_24()
    return html.Div(
        className="row power-settings-tab",
        children=[
            # Title
            html.Div(
                className="Title",
                children=[html.H3(
                    f"Last Record : {date.year}-{date.strftime('%B')}-{date.day} {date.hour}:{date.minute}", id="power-title", #className='title-line'
                    )]),
        ],
    )

# Create an empty list to hold the components for each column
column_components = []

columns = ['CV1', 'MV1', 'MV2', 'MV3', 'MV4']

# Create a row for each column in the dataset
for index, column in enumerate(columns):

    column_components.append(
        html.Div([
            html.Div(
                html.Div(
                    html.A(
                        html.H6(f"{column}"),
                        href='#',
                        id={'type': 'column-link', 'index': column},
                        n_clicks=0,
                        style={'text-decoration': 'none', 'text-align': 'center'}
                    )
                ),
                className="two columns",
                style={'text-align': 'center', 'flex': '2',}
                ),
            html.Div([
            dcc.Graph(id=f"column-graph-{column}", figure={})
                ], className="four columns", style={'text-align': 'center', 'flex': '10',}),
        ], className="row", style = {"width":"100%", 'display': 'flex', 'width': '95%', 'padding':'15px'}))
    

# Call back to update the colors of the graphs in the table after changing the theme
@app.callback(
    [Output(f"column-graph-{column}", "figure") for column in columns],
    [Input("toggleTheme", "value"), Input('interval-component', 'n_intervals')],
)
def update_column_graphs(theme_value, interval):
    theme_select = "dark" if theme_value else "light"
    axis = axis_color[theme_select]
    marker = marker_color[theme_select]
    
    df, _ = preprocessor.request_24()
    graph_outputs = []

    for column in columns:
        base_figure = dict(
            data=[dict(x=df["Date"], y=df[column], marker={"color": marker})],
            layout=dict(
                xaxis=dict(
                    #title="Date",
                    color=axis,
                    titlefont=dict(family="Gabarito", size=13),
                    #gridcolor="#61626370",
                    showgrid=False,
                ),
                yaxis=dict(
                    #title=column,
                    color=axis,
                    titlefont=dict(family="Gabarito", size=13),
                    #gridcolor="#61626370",
                    showgrid=False,
                ),
                margin={"l": 0, "b": 0, "t": 0, "r": 0},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=70
            ),
        )
        graph_outputs.append(base_figure)
    return graph_outputs

# Define list of ids for each variable
link_ids = [{'type': 'column-link', 'index': column} for column in columns]

# Call back to update the main graph after select a variable or changing the theme
@app.callback(
    Output("main-graph", "figure"),
    [Input("toggleTheme", "value")] + [Input('interval-component', 'n_intervals')]
    + [Input({'type': 'column-link', 'index': column}, 'n_clicks') for column in columns],
    #prevent_initial_call=True
)
def update_selected_column(theme_value, interval , *n_clicks):
    selected_index = n_clicks.index(max(n_clicks[1:]))
    selected_column = columns[selected_index]

    if selected_column is None:
        # Default to "CV1" if no link is clicked
        selected_column = "CV1"

    theme_select = "dark" if theme_value else "light"
    selected_column = "prediction"
    column = 'CV1'
    axis = axis_color[theme_select]
    marker = marker_color[theme_select]
    df, date = preprocessor.request_24()
    df_3 = predict_3h(df, date, ['MV1', 'CV1'])
    # Create a trace for the selected column
    trace = go.Scatter(
        x=df_3["Date"],
        y=df_3[selected_column],
        mode='lines',
        marker={"color": marker},
        line={'width': 3},  # Set the line thickness
        name=selected_column
    )

    # Create horizontal lines for min and max expected values
    min_expected = go.Scatter(
        x=df_3["Date"],
        y=[preprocessor.thresholds[column][0]] * len(df),
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Min Expected'
    )
    max_expected = go.Scatter(
        x=df_3["Date"],
        y=[preprocessor.thresholds[column][1]] * len(df),
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Max Expected'
    )
    # Create the layout for the graph
    layout = go.Layout(
        xaxis=dict(
            title="Date",
            color=axis,
            titlefont=dict(family="Gabarito", size=13),
            #gridcolor='#61626370',
            showgrid=False,
            fixedrange=True,
        ),
        yaxis=dict(
            title=selected_column,
            color=axis,
            titlefont=dict(family="Gabarito", size=13),
            #gridcolor='#61626370',
            showgrid=False,
            fixedrange=True,
        ),
        margin={"l": 0, "b": 0, "t": 0, "r": 0},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
    )
    # Create the figure
    figure = go.Figure(data=[trace, min_expected, max_expected], layout=layout)

    return figure

# Call back to change gauges values
@app.callback(
    [Output('frozen-gauge', 'value'),
     Output('missing-gauge', 'value'),
     Output('outlier-gauge', 'value'),
     Output('frozen-gauge', 'label'),
     Output('missing-gauge', 'label'),
     Output('outlier-gauge', 'label')],
    [Input({'type': 'column-link', 'index': column}, 'n_clicks') for column in columns]+[Input('interval-component', 'n_intervals')],
)
def update_gauges_and_column(*n_clicks):
    selected_index = n_clicks.index(max(n_clicks[1:]))
    selected_column = columns[selected_index]

    if selected_column is None:
        # Default to "CV1" if no link is clicked
        selected_column = "CV1"
    df, _ = preprocessor.request_24()
    summary = preprocessor.summary(df)
    frozen = int(summary[summary['Variable'] == selected_column]['Frozen'].item())
    missing = int(summary[summary['Variable'] == selected_column]['Missing'].item())
    outlier = int(summary[summary['Variable'] == selected_column]['Outlier'].item())
    # Format the labels with the displayed values
    frozen_label = f"Frozen {frozen}%"
    missing_label = f"Missing {missing}%"
    outlier_label = f"Outlier {outlier}%"
    
    return frozen, missing, outlier, frozen_label, missing_label, outlier_label

# Main App
app.layout = html.Div(
    id="main-page",
    className='full-screen-div',
    children=[
        # Header
        html.Div(
            id="header",
            className="banner row",
            children=[
                # Logo and Title
                html.Div(
                    className="banner-logo-and-title",
                    children=[
                        html.H2(
                            "AGRODARAA FARM DASHBOARD", style={"margin-left":"0px"}
                        ),
                    ],#style={"text-align":"center"}
                ),
                # Toggle
                html.Div(
                    className="row toggleDiv",
                    children=daq.ToggleSwitch(
                        id="toggleTheme", size=40, value=True, color="#86D9C7"
                    ),
                ),
            ],
        ),
        html.Div(
            className="row",
            children=[
                # LEFT PANEL -  graphs
                html.Div(
                    className="eight columns",
                    children=[
                        html.Div(
                            id="right-panel",
                            className="right-panel",
                            children=[
                                html.Div(
                                    id="card-right-panel-info",
                                    className="light-card",
                                    children=[
                                        html.Div(
                                            [html.Div([
                                            html.Div(
                                                html.H6("CVs"), className="two columns title-line", style={'text-align': 'left', 'flex': '2'}),
                                            html.Div(
                                                html.H6("Last 24 hours"), className="four columns title-line",style={'text-align': 'left', 'flex': '10'}),
                                            
                                        ], className="row", style = {'display': 'flex', 'width': '95%', 'padding-left':'15px'})],
                                           # className="Title",
                                        ),
                                        html.Div([    
                                        # Display each column in its own row
                                        dbc.Row([*column_components])], style= {"overflow": "scroll",
                                                "height": "200px"})
                                    ], 
                                ),
                                html.Div(
                                    id="card-graph",
                                    className="light-card",
                                    children=[html.H6("PREDICTIONS", className='title-line'),
                                              dcc.Graph(id="main-graph", figure={})],
                                ),
                            ],
                        )
                    ],
                ),
                # RIGHT PANEL - NOTIFICATIONS
                html.Div(
                    className="four columns",
                    children=[
                        html.Div(
                            id="left-panel",
                            children=[
                                html.Div(
                                    id="dark-theme-components",
                                    className="left-panel-controls",
                                    style={"height": 500},
                                    children=DarkThemeProvider(
                                        theme=theme,
                                        children=[
                                            html.Div([html.H6("INDICATORS")],className="title-line",),
                                            #gauges('CV1'),
                                            html.Div([html.Div(
                                                        [
                                                            daq.Gauge(
                                                                id='frozen-gauge',
                                                                max=100,
                                                                min=0,
                                                                size=72,
                                                                scale={"labelInterval": 100},
                                                                color=theme["primary"],
                                                                className="four columns",
                                                                style={'text-align': 'center', 'flex': '4'}
                                                                ),
                                                            daq.Gauge(
                                                                id='missing-gauge',
                                                                max=100,
                                                                min=0,
                                                                size=72,
                                                                scale={"labelInterval": 100},
                                                                color=theme["primary"],
                                                                className="four columns",
                                                                style={'text-align': 'center', 'flex': '4'}
                                                                ),
                                                            daq.Gauge(
                                                                id='outlier-gauge',
                                                                max=100,
                                                                min=0,
                                                                size=72,
                                                                scale={"labelInterval": 100},
                                                                color=theme["primary"],
                                                                className="four columns",
                                                                style={'text-align': 'center', 'flex': '4'}
                                                                ),
                                                        ], id='gauges-div',
                                                        className="row title-column", 
                                                        style = {'display': 'flex', 'width': '100%','font-size': '16px', 'padding': '10px'},
                                                        #className="knobs",
                                                    )], ),
                                            last_record_div(),
                                            html.Div(
                                                className="row power-settings-tab",
                                                children=[
                                                    html.Div(
                                                        className="Title",
                                                        children=[html.H3("NOTIFICATIONS", id="function-title", className='title-line')]
                                                    ),
                                                    html.Div(
                                                        children=[]
                                                    ),
                                                ],),
                                        ],
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        dcc.Interval(
        id='interval-component',
        interval=60 * 1000,  # in milliseconds (1 minute)
        n_intervals=0
        ),
        dcc.Store(id="control-inputs", data={}),
    ],
)

# Callback updating backgrounds
@app.callback(
    [
        Output("main-page", "className"),
        Output("left-panel", "className"),
        Output("card-right-panel-info", "className"),
        Output("card-graph", "className"),
    ],
    [Input("toggleTheme", "value")],
)
def update_background(turn_dark):

    if turn_dark:
        return ["dark-main-page", "dark-card", "dark-card", "dark-card"]
    else:
        return ["light-main-page", "light-card", "light-card", "light-card"]

if __name__ == "__main__":
    app.run_server(host= '0.0.0.0', debug=True)
