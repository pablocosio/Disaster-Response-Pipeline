import nltk
nltk.download('wordnet')
nltk.download('stopwords')

import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    ''' 
    Load data from the database
    
    Args:
    database_filepath: SQL databse file
    
    Return:
    X: features dataframe
    y: target labels
    '''
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categories', engine) 

    # Define category names
    category_names = list(df.drop(['id', 'message', 'original', 'genre'], axis = 1).columns)
    
    # Define features and target variables
    X = df.message.values
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1).values

    return X, y, category_names
    

def tokenize(text):
    ''' 
    Tokenize the text data
    
    Args:
    text: message to be tokenized
    
    Return:
    clean_tokens: list of cleaned words
    '''
    
    # Separate the sentences in words
    tokens = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in tokens if w not in stopwords.words("english")]
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lowercase, eliminate blank spaces and findin the root form of the words
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok, pos='v').lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens
    

def build_model():
    ''' 
    Build the model including grid search
     
    Return:
    pipeline: model list of cleaned words
    '''
    
    # Define pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Define parameters
    parameters = {
        'clf__estimator__n_estimators': [50, 100]#, 200],
        #'clf__estimator__min_samples_leaf': [1, 5, 10],
        #'clf__estimator__max_depth': [10, 20, 50, 100, 200],
        #'tfidf__smooth_idf': (True, False),
        #'tfidf__use_idf': (True, False)
        }
    
    # Define grid search to find the optimal parameters
    model = GridSearchCV(pipeline, param_grid=parameters, verbose = 3, return_train_score = True)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    ''' 
    Evaluate model quality. Print different kpis.
    
    Args:
    model: model to predict with
    X_test: features of the testing set
    Y_test: target values of the testing set
    category_names: target labels
    '''
    
    # Use the model to predict
    y_pred = model.predict(X_test)

    # Evaluate results with the classification report
    for col in np.arange(Y_test.shape[1]):
        print("Column name: {}".format(category_names[col]))
        print(classification_report(Y_test[:,col], y_pred[:,col], labels = labels))
        print("-----------------------------------------------------------------")
    
    # Estimate accuracy
    print('Accuracy: {}'.format(np.mean(Y_test == y_pred)))
    print("-----------------------------------------------------------------")
    
    # Estimate f1-score
    for col in np.arange(len(col_names)):
        score = f1_score(y_test[:,col], y_pred[:,col], average='weighted')
        scores.append(score)

    print("TOTAL F1-SCORE: {}".format(np.mean(scores)))
    print("-----------------------------------------------------------------")
    
    
def save_model(model, model_filepath):
    '''
    Saves the model in a python pickle file    
    
    Args:
    model: trained model
    model_filepath: file to save the model
    '''
    
    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    ''' 
    Executes the whole pipeline: from loading the 
    data till saving the trained model
    '''
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()