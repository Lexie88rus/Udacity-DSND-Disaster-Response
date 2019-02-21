import sys
sys.path.insert(0, '../models/')

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine

#import custom transformers
from train_classifier import AddGenre, TokensNumber


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    #plot 1: count of messages per genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #plot 2: count of messages for each category
    categories = df.columns.values[4:]

    message_counts = []
    for category in categories:
        message_counts.append(df[category].sum())
        
    #plot 3: count number of tokens per message
    df_wc = df.copy()
    df_wc['tokens_num'] = df_wc.apply(lambda row: len(tokenize(row['message'])), axis=1)
    
    #plot 4: count number of categories per message
    df_cat = df.copy()
    df_cat['categories_num'] = df_cat.iloc[:, 3:].sum(1)
    
    # create visuals
    graphs = [
            
        #plot 1
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color='#68B6AF')
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
           
        #plot 2
        {
            'data': [
                Bar(
                    x=categories,
                    y=message_counts,
                    marker=dict(color='#7FDBE2')
                )
            ],

            'layout': {
                'title': 'Distribution of Message Сategories',
                'yaxis': {
                    'title': "Count"
                }
            }
        }, 
                
        #plot 3
        {
            'data': [
                Histogram(
                    x=df_wc['tokens_num'],
                    marker=dict(color='#82C5A0')
                )
            ],

            'layout': {
                'title': 'Distribution of number of tokens per message',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Number of tokens per message"
                }
            }
        },
                
        #plot 4
        {
            'data': [
                Histogram(
                    x = df_cat['categories_num'],
                    marker=dict(color='#EED2BB')
                )
            ],

            'layout': {
                'title': 'Distribution of number of categories per message',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Number of categories per message"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()