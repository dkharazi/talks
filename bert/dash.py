from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.io as pio


pio.templates.default = 'plotly_white'


tokenizer = AutoTokenizer.from_pretrained("lvwerra/bert-imdb")
model = AutoModelForSequenceClassification.from_pretrained("lvwerra/bert-imdb")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL])


app.layout = dbc.Container([
	html.H4('Movie Review Sentiment', className='mt-4 mb-2'),

	html.P('Write your own movie review and predict its sentiment using a BERT model trained on 50K IMDB movie reviews.'),

    dbc.Textarea(
    	id='review-input',
    	placeholder='Enter a movie review...',
    	value='',
    	className='mt-4 mb-3'
    ),

    dbc.Button(
    	'What is the Sentiment?',
    	id='review-button',
    	color='primary',
    	n_clicks=0,
    	className='mb-5'
    ),

    dcc.Graph(id='review-chart', config={'displayModeBar': False}),

    dbc.Table(
    	[],
        id='review-table',
        hover=True
    ),

    html.Div(id='table-length', children=0, style={'display': 'none'})
])


@app.callback(
	[Output('review-table', 'children'),
	Output('table-length', 'children')],
    [Input('review-button', 'n_clicks')],
    [State('review-input', 'value'),
    State('review-table', 'children'),
    State('table-length', 'children')])
def predict_review(n_clicks, review, table, table_len):
	if n_clicks == 0:
		row_list = [html.Tr([html.Td('I love this movie'), html.Td(1), html.Td(0)])]
		table_body = [html.Tbody(row_list)]
		table_header = [
		    html.Thead(html.Tr([
		    	html.Th('Movie Review'),
		    	html.Th('Positive Sentiment'),
		    	html.Th('Negative Sentiment')
		    ]))
		]
	else:
		test_tensor = tokenizer.encode(review, return_tensors="pt")
		outputs = model.forward(test_tensor)
		probs = tf.nn.softmax(outputs.logits[0].detach())
		neg_prob = round(float(probs[0].numpy()),2)
		pos_prob = round(float(probs[1].numpy()), 2)
		row = html.Tr([html.Td(review), html.Td(pos_prob), html.Td(neg_prob)])
		table.append(html.Tbody([row]))
		return [table, table_len+1]
	return [table_header + table_body, table_len+1]


@app.callback(
    Output('review-chart', 'figure'),
    [Input('table-length', 'children')],
    [State('review-table', 'children'),
    State('review-input', 'value')])
def update_chart(table_len, table, review):
	sentiments = []
	for i in table[1:]:
		sentiments.append(i['props']['children'][0]['props']['children'][1]['props']['children'])
	pos_len = len([i for i in sentiments if i >= 0.5])
	neg_len = len([i for i in sentiments if i < 0.5])
	pos_df = pd.DataFrame({'ID': [1], 'Sentiment': ['Positive'], 'Frequency': [pos_len]})
	neg_df = pd.DataFrame({'ID': [1], 'Sentiment': ['Negative'], 'Frequency': [neg_len]})
	df = pd.concat([pos_df, neg_df], ignore_index=True)
	fig = px.bar(
		df,
		x='Frequency',
		y='ID',
		color='Sentiment',
		text='Sentiment',
		orientation='h',
		color_discrete_map={'Positive': '#22b24c', 'Negative': '#f57a00'}
		)
	fig.update_layout(height=200, margin={'t': 10}, xaxis={'visible': False}, yaxis={'visible': False}, hovermode=False, clickmode='none', dragmode=False, showlegend=False)
	return fig


if __name__ == '__main__':
    app.run_server(debug=True)