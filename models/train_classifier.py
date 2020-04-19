import sys
import pandas as pd
import sqlalchemy as sql
import nltk
import re
from sklearn.multioutput import MultiOutputClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    '''
    Load data from a table called `YourTableName` in a specified database.
    :param database_filepath - string:
        path to the sqlite database
    :return:
        X - message data
        y - categories
        categories - names of categories
    '''

    engine = sql.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('YourTableName', engine)
    
    categories = ['related', 'request', 'offer', 
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    
    X = df['message']
    y = df[categories].values
    
    return X, y, categories
        


def tokenize(text):
    '''
    Remove stopwords, lemmatize, lower, strip and tokenize text data.
    :param text - string:
        string to tokenize
    :return:
        tokenized string
    '''


    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    #text = re.sub("  ", " ", text)
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    '''
    define pipeline
    :return:
        returns pipeline
    '''

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])
    return model


def gridsearch(model):
    '''
    preforms gridsearch to optimize parameters of pipeline.
    :param model - sklearn pipeline object:
        takes the result of build_model as argument
    :return:
        optimized pipeline object
    '''
    '''preforms gridsearch to optimize parameters of pipeline.
    model - sklearn pipeline object
        takes the result of build_model as argument'''

    parameters = {
        # 'clf__estimator__min_samples_split': [2, 3],
        # 'clf__estimator__min_samples_leaf':[1, 2],
        'tfidf__smooth_idf': [True, False]#,
        #'clf__estimator__n_estimators': [5, 10, 20]
    }

    cv = GridSearchCV(model, param_grid=parameters, n_jobs=1)

    return cv






def evaluate_model(model, X_test, Y_test, category_names):
    '''
    prints precision, recall and fscore for all target classes.
    :param model - array:
        the test component of the training data
    :param X_test - array:
        the test component of the training data
    :param Y_test - array:
        the test component of the target categories
    :param category_names - list of strings:
        the names of the target categories
    '''

    Y_pred = model.predict(X_test)
    
    results = pd.DataFrame({'precision': [], 'recall': [], 'fscore': []})
    index = []
    
    for i, n in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(Y_test[i], Y_pred[i], average='micro')
        results = results.append({'precision': precision, 'recall': recall, 'fscore': fscore}, ignore_index=True)
        index = index + [n]
    
    print(results)
    print(results.mean())

def save_model(model, model_filepath):
    '''
    saves the pipeline object as a pickle file.
    :param model - sklearn pipeline object:
        sklearn pipeline object
    :param model_filepath - string:
         path to save pipeline object
    '''

    #np.save(model_filepath, model)
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''run code. Trains, optimizes and saves the pipeline'''
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

        print('Optimising model...')
        cv = gridsearch(model)
        cv.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(cv, X_test, Y_test, category_names)
        print('Best Parameters: \n', cv.best_params_)
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(cv, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()