import dash
from dash import dcc, html, dash_table
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import ml as ml 
from sklearn.preprocessing import StandardScaler
import scanfile as scan
from dash.dependencies import Input, Output, State
import datetime
import plotly.graph_objs as go


# Suppress warnings
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

# Load data
data = pd.read_csv("data/pretrained_data.csv")
data_original = data[data['File Size'] <= 50000000]

# Define ransomware types
ransomware_types = data['Ransomware Type'].dropna().unique()
data['Date'] = pd.to_datetime(data['Date'])

# Standardize data
scaler = StandardScaler()
data[['File Size', 'Entropy']] = scaler.fit_transform(data[['File Size', 'Entropy']])

# Load data
x = data[['File Size', 'Entropy']]
y = data['Ransomware']

# Run classifiers
knn_accuracy, _, _, knn_confmatrix = ml.knn_evaluate(x, y)
nb_accuracy, _, _, nb_confmatrix = ml.nb_evaluate(x, y)
rf_accuracy, _, _, rf_confmatrix = ml.rf_evaluate(x, y)
gb_accuracy, _, _, gb_confmatrix = ml.gb_evaluate(x, y)

# Create Dash app
#app = dash.Dash(__name__)

# Create Dash app
external_stylesheets = ['styles.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define header layout
header_layout = html.Header([
    html.H1("AI-powered Ransomware Detection System", style={'text-align': 'center','color': '#21130d'}),
    html.Hr()
])

# Define footer layout
footer_layout = html.Footer([
    html.Hr(),
    html.P("© 2024 Saskatchewan Polytechnic Applied Research", style={'text-align': 'center','color': '#21130d'})
])

# Define combined layout for dataset attribute and machine learning training section
data_and_ml_training_layout = html.Div([
    html.H3("Dataset Detail and Machine Learning Training", style={'text-align': 'center', 'color': '#21130d'}),
    html.Div([
        dcc.Graph(id='file-size-histogram', figure=px.histogram(data_original, x='File Size', nbins=10, title='Distribution of File Size', color_discrete_sequence=['#778da9'])),
        dcc.Graph(id='entropy-histogram', figure=px.histogram(data_original, x='Entropy', nbins=10, title='Distribution of Entropy', color_discrete_sequence=['#778da9'])),
        dcc.Graph(id='ransomware-count', figure=px.bar(data_original['Ransomware'].value_counts(), x=data_original['Ransomware'].value_counts().index, y=data_original['Ransomware'].value_counts(), title='Ransomware Count', color_discrete_sequence=['#778da9']))
    ], style={'display': 'flex', 'justify-content': 'space-around', 'flex-wrap': 'wrap', 'margin': '20px'}),
    html.Div([
        html.Div([
            html.H4("kNN", style={'text-align': 'center', 'margin-bottom': '10px', 'color': '#21130d'}),
            html.P(f"Accuracy: {knn_accuracy * 100:.2f}%", style={'text-align': 'center', 'color': '#21130d'}),
            dcc.Graph(id='knn_confusionmatrix', figure=ff.create_annotated_heatmap(z=knn_confmatrix, x=['0', '1'], y=['0', '1'], colorscale='teal', reversescale=False), config={'displayModeBar': False})
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.H4("Gradient Boosting", style={'text-align': 'center', 'margin-bottom': '10px', 'color': '#21130d'}),
            html.P(f"Accuracy: {gb_accuracy * 100:.2f}%", style={'text-align': 'center', 'color': '#21130d'}),
            dcc.Graph(id='gb_confusionmatrix', figure=ff.create_annotated_heatmap(z=nb_confmatrix, x=['0', '1'], y=['0', '1'], colorscale='teal', reversescale=False), config={'displayModeBar': False})
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.H4("Random Forest", style={'text-align': 'center', 'margin-bottom': '10px', 'color': '#21130d'}),
            html.P(f"Accuracy: {rf_accuracy * 100:.2f}%", style={'text-align': 'center', 'color': '#21130d'}),
            dcc.Graph(id='rf_confusionmatrix', figure=ff.create_annotated_heatmap(z=rf_confmatrix, x=['0', '1'], y=['0', '1'], colorscale='teal', reversescale=False), config={'displayModeBar': False})
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ], style={'display': 'flex', 'justify-content': 'space-around', 'flex-wrap': 'wrap', 'margin': '20px'})
], style={'background-color': '#f0f0f0', 'padding': '20px'})

# Adjust the size of each graph
graph_style = {'width': '30%', 'height': '400px'}

# Apply the size to each graph
for graph_id in ['file-size-histogram', 'entropy-histogram', 'ransomware-count']:
    data_and_ml_training_layout[graph_id].style = graph_style

# Define layout for ransomware prediction section
ransomware_prediction_layout = html.Div([
    html.H3("Ransomware Detection (Best Model)", style={'text-align': 'center', 'margin-bottom': '20px','color': '#21130d'}),
    html.Div([
        dcc.Dropdown(
            id='folder-dropdown',
            options=[
                {'label': 'test/benign', 'value': '/Users/admin/11.SaskPoly/4.capstone/test/benign'},
                {'label': 'test/malware', 'value': '/Users/admin/11.SaskPoly/4.capstone/test/malware'},
                {'label': 'test/misc', 'value': '/Users/admin/11.SaskPoly/4.capstone/test/misc'}
            ],
            value='/Users/admin/11.SaskPoly/4.capstone/test/misc',
            style={'width': '200px', 'margin-right': '10px'}
        ),
        html.Button('Predict', id='predict-button', n_clicks=0, style={'background-color': '#21130d', 'width': '100px', 'border': 'none', 'color': 'white',  'text-align': 'center', 'text-decoration': 'none', 'display': 'inline-block', 'cursor': 'pointer', 'border-radius': '4px'}),
    ], style={'display': 'flex', 'justify-content': 'center', 'margin-bottom': '20px'}),
    html.Div(id='prediction-output', style={'width': '80%', 'margin': 'auto', 'text-align': 'center'})
], style={'background-color': '#f0f0f0', 'padding': '20px'})

# Callback to handle predicting ransomware for files in the specified folder
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('folder-dropdown', 'value')]
)
def predict_ransomware(n_clicks, folder_path):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    try:
        # Calculate file size, entropy, and write to CSV
        scan.scan_file_csv(folder_path)

        # Load the data from files_scan.csv
        new_data = pd.read_csv("predict/files_scan.csv")

        # Use the existing scaler instance fitted on the training data to scale the new data
        new_data_transformed = scaler.transform(new_data[['File Size', 'Entropy']])

        # Make predictions using Random Forest model
        bestAccuracy = max(rf_accuracy, knn_accuracy, nb_accuracy)
        if bestAccuracy == rf_accuracy:
            predictions = ml.rf_predict(x, y, new_data_transformed)
        elif bestAccuracy == knn_accuracy:
            predictions = ml.knn_predict(x, y, new_data_transformed)
        else:
            predictions = ml.nb_predict(x, y, new_data_transformed)

        # Write the results back to the CSV file
        new_data['Ransomware'] = predictions
        new_data.to_csv("predict/ransomware_predicted.csv", index=False)

        # Read the predicted data
        predicted_data = pd.read_csv("predict/ransomware_predicted.csv")

        table = dash_table.DataTable(
            id='data-table',
            columns=[{'name': col, 'id': col} for col in predicted_data.columns],
            data=predicted_data.to_dict('records'),
            style_table={'overflowX': 'auto'},  # Horizontal scroll
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                {'if': {'filter_query': '{Ransomware} = 1'}, 'color': 'red', 'fontWeight': 'bold'}
            ]
        )
        return table
    except Exception as e:
        return html.Div(f'Error: {e}', style={'color': 'red'})

# Define the layout for the Dashboard tab
# Define the layout for the Dashboard tab
dashboard_layout = html.Div([
    html.Div([
        html.Label('Date Range Selection:'),
        dcc.Dropdown(
            id='date-range-dropdown',
            options=[
                {'label': 'Last Week', 'value': 'last_week'},
                {'label': 'Last Month', 'value': 'last_month'},
                {'label': 'Last Year', 'value': 'last_year'},
                {'label': 'All', 'value': 'all'}
            ],
            value='last_month',
            className="dcc_control",
        ),
        html.Label('Ransomware Types:'),
        dcc.Dropdown(
            id='ransomware-dropdown',
            options=[{'label': ransomware, 'value': ransomware} for ransomware in ransomware_types],
            value=list(ransomware_types),
            className="dcc_control",
            multi=True
        )
    ], className='pretty_container four columns'),
    html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.H6(id="well_text"), html.P("No. of Ransomware")],
                        id="ransomware_count",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(id="gasText"), html.P("Gas")],
                        id="gas",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(id="oilText"), html.P("Oil")],
                        id="oil",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(id="waterText"), html.P("Water")],
                        id="water",
                        className="mini_container",
                    ),
                ],
                id="info-container",
                className="row container-display",
            ),
            html.Div(
                [dcc.Graph(id='ransomware-trend')],
                id="countGraphContainer",
                className="pretty_container",
            ),

        ],
        id="right-column",
    ),
    html.Div([
        dcc.Graph(id='ransomware-pie-chart')
    ], className='pretty_container four columns')
])


# Define callback to update the trend graph and pie chart based on filter selections
@app.callback(
    [Output('ransomware-trend', 'figure'),
     Output('ransomware-pie-chart', 'figure')],
    [Input('date-range-dropdown', 'value'),
     Input('ransomware-dropdown', 'value')]
)
def update_ransomware_trend_and_pie(date_range, selected_ransomware):
    # Filter data based on date range selection
    if date_range == 'last_week':
        start_date = datetime.datetime.now() - datetime.timedelta(days=7)
    elif date_range == 'last_month':
        start_date = datetime.datetime.now() - datetime.timedelta(days=30)
    elif date_range == 'last_year':
        start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    else:
        start_date = datetime.datetime.min

    filtered_data = data[data['Date'] >= start_date]

    # Filter data based on selected ransomware types
    filtered_data = filtered_data[filtered_data['Ransomware Type'].isin(selected_ransomware)]

    # Group data by date and ransomware type
    grouped_data = filtered_data.groupby(['Date', 'Ransomware Type']).size().reset_index(name='Count')

    # Create traces for each ransomware type for the trend graph
    traces = []
    for ransomware in selected_ransomware:
        ransomware_data = grouped_data[grouped_data['Ransomware Type'] == ransomware]
        trace = go.Scatter(x=ransomware_data['Date'], y=ransomware_data['Count'], mode='lines+markers', name=ransomware)
        traces.append(trace)

    # Create the trend figure
    trend_figure = {
        'data': traces,
        'layout': go.Layout(
            title='Ransomware Trend',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Count'},
            hovermode='closest'
        )
    }

    # Create the pie chart
    ransomware_counts = filtered_data['Ransomware Type'].value_counts().reset_index()
    ransomware_counts.columns = ['Ransomware Type', 'Count']
    pie_figure = px.pie(ransomware_counts, values='Count', names='Ransomware Type', title='Ransomware Type Percentage')

    return trend_figure, pie_figure

# Define the layout with tabs
app.layout = html.Div(
    [
    header_layout,
    dcc.Tabs([
        dcc.Tab(label='Data & ML Training', children=[data_and_ml_training_layout]),
        dcc.Tab(label='Ransomware Prediction', children=[ransomware_prediction_layout]),
        dcc.Tab(label='Dashboard', children=[dashboard_layout])
    ]),
    footer_layout
    ]   
)

# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8060, debug=True)
