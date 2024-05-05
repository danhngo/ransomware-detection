import dash
from dash import dcc, html, dash_table
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import ml as ml 
from sklearn.preprocessing import StandardScaler
import scanfile as scan
from dash.dependencies import Input, Output, State

# Suppress warnings
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

# Load data
data = pd.read_csv("data/pretrained_data.csv")
data_original = data[data['File Size'] <= 50000000]

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
app = dash.Dash(__name__)

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

# Define layout for dataset attribute section
data_training_layout = html.Div([
    html.H3("Dataset Attribute", style={'text-align': 'center','color': '#21130d'}),
    html.Div([
        dcc.Graph(id='file-size-histogram', figure=px.histogram(data_original, x='File Size', nbins=10, title='Distribution of File Size', color_discrete_sequence=['#778da9'])),
        dcc.Graph(id='file-type-histogram', figure=px.histogram(data_original, x='File Type', nbins=10, title='Distribution of File Type', color_discrete_sequence=['#778da9'])),
        dcc.Graph(id='entropy-histogram', figure=px.histogram(data_original, x='Entropy', nbins=10, title='Distribution of Entropy', color_discrete_sequence=['#778da9'])),
        dcc.Graph(id='ransomware-count', figure=px.bar(data_original['Ransomware'].value_counts(), x=data_original['Ransomware'].value_counts().index, y=data_original['Ransomware'].value_counts(), title='Ransomware Count', color_discrete_sequence=['#778da9']))
    ], style={'display': 'flex', 'justify-content': 'space-around', 'flex-wrap': 'wrap', 'margin': '20px'})
], style={'background-color': '#f0f0f0', 'padding': '20px'})

# Adjust the size of each graph
graph_style = {'width': '25%', 'height': '400px'}

# Apply the size to each graph
for graph_id in ['file-size-histogram', 'file-type-histogram', 'entropy-histogram', 'ransomware-count']:
    data_training_layout[graph_id].style = graph_style

# Define layout for machine learning training section
machine_learning_training_layout = html.Div([
    html.H3("Machine Learning Training", style={'text-align': 'center', 'margin-bottom': '20px','color': '#21130d'}),
    html.Div([
        html.Div([
            html.H4("kNN Training", style={'text-align': 'center', 'margin-bottom': '10px','color': '#21130d'}),
            html.P(f"kNN Accuracy: {knn_accuracy * 100 :.2f}%", style={'text-align': 'center','color': '#21130d'}),
            dcc.Graph(id='knn_confusionmatrix', figure=ff.create_annotated_heatmap(z=knn_confmatrix, x=['0', '1'], y=['0', '1'], colorscale='teal', reversescale=False), config={'displayModeBar': False})
        ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.H4("Naive Bayes Training", style={'text-align': 'center', 'margin-bottom': '10px','color': '#21130d'}),
            html.P(f"Naive Bayes Accuracy: {nb_accuracy * 100 :.2f}%", style={'text-align': 'center','color': '#21130d'}),
            dcc.Graph(id='nb_confusionmatrix', figure=ff.create_annotated_heatmap(z=nb_confmatrix, x=['0', '1'], y=['0', '1'], colorscale='teal', reversescale=False), config={'displayModeBar': False})
        ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.H4("Gradient Boosting Training", style={'text-align': 'center', 'margin-bottom': '10px','color': '#21130d'}),
            html.P(f"Gradient Boosting Accuracy: {gb_accuracy * 100 :.2f}%", style={'text-align': 'center','color': '#21130d'}),
            dcc.Graph(id='gb_confusionmatrix', figure=ff.create_annotated_heatmap(z=nb_confmatrix, x=['0', '1'], y=['0', '1'], colorscale='teal', reversescale=False), config={'displayModeBar': False})
        ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.H4("Random Forest Training", style={'text-align': 'center', 'margin-bottom': '10px','color': '#21130d'}),
            html.P(f"Random Forest Accuracy: {rf_accuracy * 100 :.2f}%", style={'text-align': 'center','color': '#21130d'}),
            dcc.Graph(id='rf_confusionmatrix', figure=ff.create_annotated_heatmap(z=rf_confmatrix, x=['0', '1'], y=['0', '1'], colorscale='teal', reversescale=False), config={'displayModeBar': False})
        ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ], style={'display': 'flex', 'justify-content': 'space-around', 'flex-wrap': 'wrap', 'margin': '20px'})
])

# Define layout for ransomware prediction section
ransomware_prediction_layout = html.Div([
    html.H3("Ransomware Prediction (Best Model)", style={'text-align': 'center', 'margin-bottom': '20px','color': '#21130d'}),
    html.Div([
        dcc.Dropdown(
            id='folder-dropdown',
            options=[
                {'label': 'test/benign', 'value': '/Users/admin/11.SaskPoly/4.Innovation/test/benign'},
                {'label': 'test/malware', 'value': '/Users/admin/11.SaskPoly/4.Innovation/test/malware'},
                {'label': 'test/misc', 'value': '/Users/admin/11.SaskPoly/4.Innovation/test/misc'}
            ],
            value='/Users/admin/11.SaskPoly/4.Innovation/test/misc',
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

# Combine header, content, and footer layouts
app.layout = html.Div([
    header_layout,
    data_training_layout,
    machine_learning_training_layout,
    ransomware_prediction_layout,
    footer_layout
])

# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)
