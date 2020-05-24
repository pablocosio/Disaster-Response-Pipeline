import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

app = Flask(__name__)

def tokenize(text):
    """ Tokenizes text data"""
    
    # Separate the sentences in words
    tokens = word_tokenize(text)
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lowercase, eliminate blank spaces and findin the root form of the words
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='v').lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens
    
# load data
try:
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('messages_categories', con = engine)
except:
    print('If load data from database failed, try to run it from the app folder')
    
# load model
model = joblib.load("../models/classifier_model.pkl")

# Function for first plot
def first_plot(df):
    """Create first plot TOP 10 categories """
    
    # Define counts
    categories = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum().sort_values(ascending=False)

    # Get top 10
    categories = categories[0:10]
    
    # Json for plotting
    data = [Bar(
        x = categories.index,
        y = categories.values,
    )]
    
    layout = {
        "title": "Messages per category",
        "xaxis": {
               "title": 'Categories'
        },
        "yaxis": {
               "title": 'No. of messages'
        }
    }
    
    return {"data": data, "layout": layout}
    
# Function for second plot
def second_plot(df):
    """Create second plot with message genres """
    
    # Define counts
    genres = df.groupby('genre').count()['message']
    
    # Json for plotting
    data = [Bar(
        x = genres.index,
        y = genres.values,
    )]
    
    layout = {
        "title": "Messages per genre",
        "xaxis": {
               "title": 'Genres'
        },
        "yaxis": {
               "title": 'No. of messages'
        }
    }
    
    return {"data": data, "layout": layout}
    
# Function for second plot
def third_plot(df):
    """Create third plot with 0/1 per category """
    
    # Define top-10 categories by amount of messages
    categories = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum().sort_values(ascending=False)
    categories = list(categories[0:10].index)
    
    # Define loop
    graphs = []
    for cat in categories:
        counts = df[cat].value_counts()
        # Json for plotting
        data = [Bar(
            x = counts.index,
            y = counts.values,
        )]
    
        layout = {
            "title": "Classification messages of category {}".format(cat),
            "xaxis": {
                   "title": 'Assigned to the category or not',
                   "tickmode": "array",
                   "tickvals": [0,1],
                   "ticktext":["no", "yes"]
            },
            "yaxis": {
                   "title": 'No. of messages'
            }
        }
        
        graphs.append({"data": data, "layout": layout})
    
    return graphs
    
# call function
fig1 = first_plot(df)
fig2 = second_plot(df)
figs = third_plot(df)
    
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
 
    # encode plotly graphs in JSON
    graphs = [ fig1, fig2 ] + figs
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