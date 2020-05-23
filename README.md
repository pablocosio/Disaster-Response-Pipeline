# Disaster Response Pipeline Project

### Motivation

During a disaster, emergency services personal have no time to classify messages received and to analyze them. Consequently, this project classifies these messages through machine learning. Thus, emergency personal can concentrate on saving lifes while the understanding and classification of the received messages are automated.

### Required installation

The following packages/libraries are required the data loading, preprocessing and machine learning algorithm:

- numpy
- pandas
- sqlalchemy
- sys
- nltk: wordnet and stopwords
- sklearn

### Content

- Data
    - process_data.py: reads the csv files with the messages and categories, merges both dataframes and cleans the resulting df. Finally, the resulting dataset is saved in a SQL database.
    - categories.csv: contains categories for the messages classification. It is used to trained and test the model.
    - messages.csv: contains the messages to be classified.
    - DisasterResponse.db: created database with the cleaned data.

- Models
    - train_classifier.py: loads data from the database, uses the nltk package to transform the text into tokens which can be used to feed a ML algorithm. Finally, a machine learning algorithm is defined, trained with Grid Search and finally, tested. This model is saved in a pickle file.

- App: 
    - run.py: runs a flask app creating a user interface to predict results and display them.
    - templates: contains the html templates.

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Authors

Pablo Cosio, pca_92@hotmail.com

### Acknowledgements

Thanks to Udacity and Figure Eight for providing the data.
