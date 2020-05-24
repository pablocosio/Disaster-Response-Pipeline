# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    '''
    Loads data and merge the dataframes    
    
    Args:
    messages_filepath: path of the messages csv file
    categories_filepath: path of the categories csv file
    
    Return:
    df: resulting dataframe from the merge of the categories and messages datasets
    '''
    
    try:
        # load messages dataset
        messages = pd.read_csv(messages_filepath)
        # load categories dataset
        categories = pd.read_csv(categories_filepath)

        # merge datasets
        df = messages.merge(categories, on='id', how='inner')
   
        return df
  	
    except:
        sys.exit("Exception occurred while loading data.")


def clean_data(df):
    '''
    Cleans data for applying the ML algorithm: separate categories in columns, convert 
    variables to binary and drop duplicates.
    
    Args:
    df: uncleaned dataframe
       
    Return:
    df: cleaned dataframe
    '''
    
    try:
        # create a dataframe of the 36 individual category columns
        categories = df.categories.str.split(";", expand = True)

        # select the first row of the categories dataframe and extract column names
        row = pd.Series(categories.iloc[0])
        category_colnames = row.str.slice(0, -2)

        # rename the columns of `categories`	
        categories.columns = category_colnames

        for column in categories:
            # set each value to be the last character of the string
            categories[column] = categories[column].str.slice(-1)
            # convert column from string to numeric
            categories[column] = pd.to_numeric(categories[column])

        # Convert the first column to binary
        categories.related = categories.related.apply(lambda row: 1 if row == 2 else row)
        
        # drop the original categories column from `df`
        df.drop('categories', axis=1, inplace=True)

        # concatenate the original dataframe with the new `categories` dataframe
        df = pd.merge(df, categories, left_index = True, right_index = True, how ='inner')

        # drop duplicates
        df.drop_duplicates(inplace=True)
            
        return df
            
    except:
        sys.exit("Exception occurred while cleaning data.")


def save_data(df, database_filename):
    '''
    Saves the data into a SQL database.
    
    Args:
    df: cleaned dataframe
    database_filename: path of the databse where the data is saved
    '''
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages_categories', engine, index=False)  


def main():
    '''
    Reads user input, prepares the data cleaning it and saves it on a database
    '''
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()