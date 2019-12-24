import json
import plotly
import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    tokenize the text by removing stopwords,removing short words and lemmetization  
    """
    
    tokens = word_tokenize(text)
    
    STOPWORDS = list(set(stopwords.words('english')))
    # remove short words
    tokens = [token for token in tokens if len(token) > 2]
    # remove stopwords
    tokens = [token for token in tokens if token not in STOPWORDS]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # for graph one
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    ### for graph two
    df2 = df.iloc[:,-36:]
    category_counts = list(df2.sum(axis = 0, skipna = True))
    category_names = list(df2.columns)
    
    ### for graph three
    top_N = 20
    #if not necessary all lower
    a = df['message'].str.lower().str.cat(sep=' ')
    words = tokenize(a)
    word_dist = nltk.FreqDist(words)
    rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency'])
    word = list(rslt['Word'])
    word_count = list(rslt['Frequency'])
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
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
        },
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Target variable',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=word,
                    y=word_count
                )
            ],

            'layout': {
                'title': 'Top 20 most frequently used words',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Word"
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