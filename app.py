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

scan_data = pd.read_csv("predict/ransomware_detection.csv")

# Define ransomware types
ransomware_types = scan_data['Ransomware Type'].dropna().unique()
scan_data['Date'] = pd.to_datetime(scan_data['Date'])

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
header_layout = html.Header(
    className='header',
    children=[
        html.H1("AI-powered Ransomware Detection System")
    ]
)

# Define footer layout
footer_layout = html.Footer(
    className='footer',
    children=[
        html.P("© 2024 Saskatchewan Polytechnic")
    ]
)

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
                        [html.H6(id="filesNoText"), html.P("No. of Scan Files")],
                        id="scan_count",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(id="ransomwareText"), html.P("No. of Ransomware")],
                        id="ransomware_count",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(id="ransomwareTypeText"), html.P("No. of Ransomware Types")],
                        id="ransomwareType_count",
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

    filtered_data = scan_data[scan_data['Date'] >= start_date]

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
# Define the layout
app.layout = html.Div(
    [
        header_layout,
        html.Div(
            className="tab-container",
            children=[
                dcc.Tabs(
                    id="tabs",
                    value="tab-1",
                    children=[
                        dcc.Tab(label='Data & ML Training', children=[data_and_ml_training_layout], className="tab-style", selected_className="tab-style--selected"),
                        dcc.Tab(label='Ransomware Dashboard', children=[dashboard_layout], className="tab-style", selected_className="tab-style--selected")
                    ]
                )
            ]
        ),
        footer_layout
    ]
)

# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8060, debug=True)
