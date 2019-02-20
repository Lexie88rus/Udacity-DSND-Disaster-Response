# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function loads messages and categories data from csv datasets and merges
    into one pandas dataframe
    
    INPUTS:
        1. messages_filepath - path to csv file containing messages
        2. categories_filepath - path to csv file containing message categories
        
    OUTPUT:
        1. df - merged dataset containing data from both messages and categories
    '''
    
    # load data from csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets on 'id'
    df = messages.merge(categories, on = 'id')
    
    return df


def clean_data(df):
    '''
    Function for cleaning data in dataframe, containing data loaded from
    messages and message categories csv: function splits 'Categories' column into
    several columns for each category and removes duplicated rows.
    
    INPUTS:
        1. df - pandas dataframe, containing raw data
        
    OUTPUTS:
        1. df - cleaned pandas dataframe
    
    '''
    
    # Split the values in the categories column on the ; character so that each value becomes a separate column
    categories = df['categories'].str.split(pat = ';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc(0)[0]

    # use row to extract a list of new column names for categories
    category_colnames = [x.split('-')[0] for x in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        # convert column values to binary
        categories.loc[categories[column] > 1, column] = 1

    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicated rows from resulting dataset
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    Functions saves cleaned dataframe into sqlite database
    Rewrites data if the table already exists
    
    INPUTS:
        1. df - cleaned dataframe to be saved into database
        2. database_filename - database filename for the dataframe to be saved to
        
    OUTPUTS: none
    '''
    
    conn_string = 'sqlite:///{database_filename}'.format(database_filename = database_filename)
    engine = create_engine(conn_string)
    df.to_sql('DisasterMessages', engine, index=False, if_exists = 'replace')


def main():
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