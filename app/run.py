import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    Remove stopwords, lemmatize, lower, strip and tokenize text data.
    :param text - string:
        string to tokenize
    :return:
        tokenized string
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('YourTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl", 'rb')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_list = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']

    category_sum = df[category_list].sum()


    # create visuals
    graphs = [

        {


            'data': [
                Bar(
                    x=category_list,
                    y=category_sum,

                )
            ],

            'layout': {
                'title': 'Number of Messages per Classification in the Training Data',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Training Classes"
                }
            }
        },
             {

                 'data': [
                     Bar(
                         x=genre_names,
                         y=genre_counts
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
    '''run web app'''
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()