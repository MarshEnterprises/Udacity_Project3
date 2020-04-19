import sys
import nltk
import pandas as pd
import sqlalchemy as sql

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



def load_data(messages_filepath, categories_filepath):
    '''
    imports two csv files, preforms inner union and returns dataframe.
    :param messages_filepath - string:
        path to csv file containing messages
    :param categories_filepath - string:
        path to csv containing categories
    :return:
        pandas.DataFrame
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    '''
    removes redundant string and converts category data into int
    :param df - pandas.Dataframe:
        dataframe containing categorical data as string
    :return:
        pandas.DataFrame with categorical data as int
    '''
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = categories.loc[0, :].apply(lambda x: x[0:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    df = df.drop(labels='categories', axis=1)
    df = pd.concat((df, categories), axis=1)
    
    return df
    
    
def save_data(df, database_filename):
    '''
    Saves dataframe as a table in a sqlite database.
    :param df - pandas.DataFrame:
        cleaned data
    :param database_filename - string:
        path for file save
    '''
    engine = sql.create_engine('sqlite:///' + database_filename)
    df.to_sql('YourTableName', engine, index=False, if_exists='replace')
    


def main():
    '''
    run.
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
    