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
import time  # Import the time module
import plotly.figure_factory as ff


# Suppress warnings
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

# Load data
data = pd.read_csv("data/pretrained_data.csv")
data_original = data[data['File Size'] <= 50000000]
train_count = len(data_original)
benign_count = len(data_original[data_original['Ransomware'] == 0])
ransomware_count = len(data_original[data_original['Ransomware'] == 1])
ransomware_percent = ransomware_count / train_count * 100
entropy_mean = data_original['Entropy'].mean()    
file_size_mean = data_original['File Size'].mean() / 1000000
  
scan_data = pd.read_csv("predict/ransomware_detection.csv")
scan_data['Date'] = pd.to_datetime(scan_data['Date'], format='mixed')

# Filter scan_data
ransomware_data = scan_data[(scan_data['Ransomware'] == 1) & (scan_data['Ransomware Type'] != 'unknow') & (scan_data['Ransomware Type'] != 'MAZE')]
ransomware_types = ransomware_data['Ransomware Type'].dropna().unique()

# Mapping for x-axis labels
label_mapping = {0: 'Benign', 1: 'Ransomware'}
data_original['Ransomware_Label'] = data_original['Ransomware'].map(label_mapping)


# Standardize data
scaler = StandardScaler()
data[['File Size', 'Entropy']] = scaler.fit_transform(data[['File Size', 'Entropy']])

# Load data
x = data[['File Size', 'Entropy']]
y = data['Ransomware']

# Run classifiers
knn_accuracy, _, _, knn_confmatrix, knn_total_train_files, knn_total_test_files = ml.knn_evaluate(x, y)
nb_accuracy, _, _, nb_confmatrix, nb_total_train_files, nb_total_test_files = ml.nb_evaluate(x, y)
rf_accuracy, _, _, rf_confmatrix, rf_total_train_files, rf_total_test_files = ml.rf_evaluate(x, y)
gb_accuracy, _, _, gb_confmatrix, gb_total_train_files, gb_total_test_files = ml.gb_evaluate(x, y)

# Create figure
ransomware_benign_fig = px.bar(
    data_original['Ransomware_Label'].value_counts(),
    x=data_original['Ransomware_Label'].value_counts().index,
    y=data_original['Ransomware_Label'].value_counts(),
    title='Benign vs. Ransomware',
    color_discrete_sequence=['#778da9']
)

# Update layout to add axis titles
ransomware_benign_fig.update_layout(
    xaxis_title="Type",
    yaxis_title="Count"
)

# Create Dash app
external_stylesheets = ['styles.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define header layout
header_layout = html.Header(
    className='header',
    children=[
        html.H1("Machine Learning System for Ransomware Detection")
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

    html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.H6(f"{train_count}"), html.P("No. of Files")],
                        id="trained_count",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(f"{benign_count}"), html.P("No. of Benign Files")],
                        id="benign_count",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(f"{ransomware_count}"), html.P("No. of Ransomware Files")],
                        id="trained_ransomware_count",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(f"{entropy_mean:.2f}"), html.P("Entropy Mean")],
                        id="entropy_count",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(f"{file_size_mean:.2f} Mb"), html.P("File Size Mean")],
                        id="file_size_count",
                        className="mini_container",
                    ),
                ],
                id="info-container-2",
                className="row container-display",
            ),
        ],
    ),
   
    html.Div([
        dcc.Graph(id='ransomware-begign-fig', figure=ransomware_benign_fig),
        dcc.Graph(id='entropy-histogram', figure=px.histogram(data_original, x='Entropy', nbins=10, title='Distribution of Entropy', color_discrete_sequence=['#778da9'])),
        dcc.Graph(id='file-size-histogram', figure=px.histogram(data_original, x='File Size', nbins=10, title='Distribution of File Size', color_discrete_sequence=['#778da9'])), 
    ], style={'display': 'flex', 'justify-content': 'space-around', 'flex-wrap': 'wrap', 'margin': '20px'}),
    
    html.Div([
         html.Div([
            html.Button('Start Machine Learning', id='train-button', n_clicks=0),
        ], style={'text-align': 'center', 'margin': '20px'}),
    ], style={'display': 'flex', 'justify-content': 'space-around', 'flex-wrap': 'wrap','padding-top':'20px', 'margin': '20px'}),

    html.Div([
        dcc.Loading(
            id="loading-training-results",
            type="default",
            children=[
                html.Div([
                    html.Div(id='rf-training-results', style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div(id='knn-training-results', style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div(id='gb-training-results', style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
                ])
            ]
        ),
    ], style={'justify-content': 'space-around', 'flex-wrap': 'wrap','margin-left': '45px','margin-right': '45px'})

])

# Adjust the size of each graph
graph_style = {'width': '30%', 'height': '400px'}

# Apply the size to each graph
for graph_id in ['file-size-histogram', 'entropy-histogram', 'ransomware-begign-fig']:
    data_and_ml_training_layout[graph_id].style = graph_style

@app.callback(
    [Output('rf-training-results', 'children'),
    Output('knn-training-results', 'children'),
    Output('gb-training-results', 'children')],
    [Input('train-button', 'n_clicks')]
)
def machine_learning_train(n_clicks):
    if n_clicks > 0:
        # Add a delay of 2 seconds
        time.sleep(2)

        # Train the Random Forest model
        rf_accuracy, rf_report, rf_f1, rf_confmatrix, rf_total_train_files, rf_total_test_files = ml.rf_evaluate(x, y)
        # Train the KNN model
        knn_accuracy, knn_report, knn_f1, knn_confmatrix, knn_total_train_files, knn_total_test_files = ml.knn_evaluate(x, y)
        # Train the Gradient Boosting model
        gb_accuracy, gb_report, gb_f1, gb_confmatrix, gb_total_train_files, gb_total_test_files = ml.gb_evaluate(x, y)

        # Custom labels
        prediction_x_labels = ['Benign', 'Ransomware']
        prediction_y_labels = ['Benign', 'Ransomware']

        # Create the confusion matrix heatmap
        rf_conf_matrix_fig = ff.create_annotated_heatmap(
            z=rf_confmatrix,
            x=prediction_x_labels,
            y=prediction_y_labels,
            colorscale='teal',
            reversescale=False,
            annotation_text=[[str(y) for y in x] for x in rf_confmatrix],
            hoverinfo="z"
        )
        # Customize layout
        rf_conf_matrix_fig.update_layout(
            xaxis=dict(title='Predicted'),
            yaxis=dict(title='Actual')
        )

        # Create the confusion matrix heatmap
        knn_conf_matrix_fig = ff.create_annotated_heatmap(
            z=knn_confmatrix,
            x=prediction_x_labels,
            y=prediction_y_labels,
            colorscale='teal',
            reversescale=False,
            annotation_text=[[str(y) for y in x] for x in knn_confmatrix],
            hoverinfo="z"
        )
        # Customize layout
        knn_conf_matrix_fig.update_layout(
            xaxis=dict(title='Predicted'),
            yaxis=dict(title='Actual')
        )

        # Create the confusion matrix heatmap
        gb_conf_matrix_fig = ff.create_annotated_heatmap(
            z=gb_confmatrix,
            x=prediction_x_labels,
            y=prediction_y_labels,
            colorscale='teal',
            reversescale=False,
            annotation_text=[[str(y) for y in x] for x in gb_confmatrix],
            hoverinfo="z"
        )
        # Customize layout
        gb_conf_matrix_fig.update_layout(
            xaxis=dict(title='Predicted'),
            yaxis=dict(title='Actual')
        )

        # Create a summary of results
        rf_results = html.Div([
            html.H4("Random Forest Model", style={'text-align': 'center', 'margin-bottom': '10px', 'color': '#21130d'}),
            html.P(f"Accuracy: {rf_accuracy * 100:.2f}%", style={'text-align': 'center','color': '#21130d'}),
            dcc.Graph(figure=rf_conf_matrix_fig)
        ])

        knn_results = html.Div([
            html.H4("kNN Model", style={'text-align': 'center', 'margin-bottom': '10px', 'color': '#21130d'}),
            html.P(f"Accuracy: {knn_accuracy * 100:.2f}%", style={'text-align': 'center','color': '#21130d'}),
            dcc.Graph(figure=knn_conf_matrix_fig)
        ])

        gb_results = html.Div([
            html.H4("Gradient Boosting Model", style={'text-align': 'center', 'margin-bottom': '10px', 'color': '#21130d'}),
            html.P(f"Accuracy: {gb_accuracy * 100:.2f}%", style={'text-align': 'center','color': '#21130d'}),
            dcc.Graph(figure=gb_conf_matrix_fig)
        ])

        return rf_results,knn_results,gb_results

    return html.Div(),html.Div(),html.Div()




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
            value='last_week',
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
                        [html.H6(id="dataCountText"), html.P("No. of Scan Files")],
                        id="scan_count",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(id="ransomwareCountText"), html.P("No. of Ransomware")],
                        id="ransomware_count",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(id="ransomwarePercentText"), html.P("Ransomware Percent")],
                        id="ransomware_percent",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(id="ransomwareTypeCountText"), html.P("Ransomware Types")],
                        id="ransomwareType_count",
                        className="mini_container",
                    ),
                    
                ],
                id="info-container",
                className="row container-display",
            ),
            html.Div(
                [dcc.Graph(id='ransomware-trend', style={'height': '380px'})],
                id="countGraphContainer",
                className="pretty_container",
            )
        ],
        id="right-column",
    ),
    html.Div(
        [
            html.Div(
                [dcc.Graph(id="ransomware-bar-chart")],
                className="pretty_container six columns",
            ),
            html.Div(
                [dcc.Graph(id="ransomware-pie-chart")],
                className="pretty_container six columns",
            ),
        ],
        className="row flex-display",
    )
    
])


# Define callback to update the trend graph and pie chart based on filter selections
@app.callback(
    [Output('ransomware-trend', 'figure'),
     Output('ransomware-pie-chart', 'figure'),
     Output('ransomware-bar-chart', 'figure'),
     Output('dataCountText', 'children'),
     Output('ransomwareCountText', 'children'),
     Output('ransomwarePercentText', 'children'),
     Output('ransomwareTypeCountText', 'children')],
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

    total_data = scan_data[scan_data['Date'] >= start_date]
    filtered_data = ransomware_data[ransomware_data['Date'] >= start_date]

    # Filter data based on selected ransomware types
    filtered_data = filtered_data[filtered_data['Ransomware Type'].isin(selected_ransomware)]

    # Group data by date and ransomware type
    grouped_data = filtered_data.groupby(['Date', 'Ransomware Type']).size().reset_index(name='Count')

    # Create traces for each ransomware type for the trend graph
    traces = []
    for ransomware in selected_ransomware:
        ransomware_trend_data = grouped_data[grouped_data['Ransomware Type'] == ransomware]
        trace = go.Scatter(
            x=ransomware_trend_data['Date'],
            y=ransomware_trend_data['Count'],
            mode='lines',
            name=ransomware,
            line_shape='spline'  # Use 'spline' for smoothing
        )
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
    pie_figure = px.pie(ransomware_counts, values='Count', names='Ransomware Type', title='Ransomware Types')

    
    # Calculate counts
    data_count = len(total_data)
    ransomware_count = filtered_data['Ransomware'].sum()
    ransomware_percent = '{1:.{0}f}%'.format(1, (ransomware_count / data_count * 100) / 10 ** 1) 
    ransomware_types_count = len(filtered_data['Ransomware Type'].unique())

   
    # Create the bar chart
    ransomware_type_counts = filtered_data['Ransomware Type'].value_counts().reset_index()
    ransomware_type_counts.columns = ['Ransomware Type', 'Count']

    bar_data = [
        go.Bar(name='All Types', x=['All Types'], y=[data_count])
    ]

    for index, row in ransomware_type_counts.iterrows():
        bar_data.append(go.Bar(name=row['Ransomware Type'], x=[row['Ransomware Type']], y=[row['Count']]))

    bar_figure = {
        'data': bar_data,
        'layout': go.Layout(
            title='Ransomware Count by Type',
            xaxis={'title': 'Type'},
            yaxis={'title': 'Count'},
            barmode='group'
        )
    }

    return trend_figure, pie_figure,bar_figure, data_count,ransomware_count,ransomware_percent,ransomware_types_count


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
                        dcc.Tab(label='Data & Machine Learning Training', children=[data_and_ml_training_layout], className="tab-style", selected_className="tab-style--selected"),
                        dcc.Tab(label='Ransomware Prediction', children=[dashboard_layout], className="tab-style", selected_className="tab-style--selected")
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
