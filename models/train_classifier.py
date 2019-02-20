#import common libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

#import nltk for nlp
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# import machine learning
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier

# import xgboost
import xgboost as xgb

# import pickle to save the model
import pickle


class AddGenre(BaseEstimator, TransformerMixin):
    '''
    Custom transformer class to add genre label to features to train the model
    '''
    
    def __init__(self, df):
        '''
        Initialization
        
        INPUTS:
            1. df - original dataframe loaded from sqlite database
        '''
        self.df = df
        
    def get_genre_label(self, genre):
        '''
        Function returns genre label depending on genre name
        
        INPUTS:
            1. genre - string containing genre name
            
        RETURNS:
            1. genre label - 0 if genre is 'direct', 1 if genre is 'social'
            2 - otherwise
        '''
        if genre == 'direct':
            return 0
        if genre == 'social':
            return 1
        else:
            return 2
    
    def add_genre(self, text):
        '''
        Function returns genre depending on message
        
        INPUTS:
            1. text - message text
            
        RETURNS:
            1. genre label - 0 if genre is 'direct', 1 if genre is 'social'
            2 - otherwise
        '''
        try:
            genre = self.df.loc[self.df['message'] == text, 'genre'].values[0]
            return self.get_genre_label(genre)
        except:
            return -1

    def fit(self, X, y=None):
        '''
        Transformer fit function, no custom logic
        '''
        return self

    def transform(self, X):
        '''
        Transform function, applies transformation for the dataset: adds
        genre label to features
        '''
        X_tagged = pd.Series(X).apply(self.add_genre)
        return pd.DataFrame(X_tagged)
    
    
class TokensNumber(BaseEstimator, TransformerMixin):
    '''
    Custom transformer class which adds number of tokens in message to features
    '''
    
    def get_tokens_number(self, text):
        '''
        Function which returns the number of tokens
        
        INPUTS:
            1. text - message text
            
        OUTPUT:
            1. number of tokens in provided message text
        '''
        tokens = nltk.word_tokenize(text)
        return len(tokens)

    def fit(self, X, y=None):
        '''
        Transformer fit function, no custom logic
        '''
        return self

    def transform(self, X):
        '''
        Transform function, applies transformation for the dataset: adds
        number of tokens to features
        '''
        X_tagged = pd.Series(X).apply(self.get_tokens_number)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    '''
    Function loads clean datasets ready for machine learning from sqlite database
    
    INPUTS:
        1. database_filepath - path to sqlite database .db file
        
    OUTPUTS:
        1. df - full dataframe loaded from sqlite database
        2. X - list of messages to be marked with message categories
        3. Y - dataframe containing message categories for each message in X
        4. category_names - list of category names
    '''
    # load data from database
    engine = create_engine('sqlite:///{database_filepath}'.format(database_filepath = database_filepath))
    df = pd.read_sql("SELECT * FROM DisasterMessages", engine)
    
    # split into X and Y datasets
    X = df.message.values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    #find list of category names as list of Y columns
    category_names = Y.columns.values
    
    return df, X, Y, category_names


def tokenize(text):
    '''
    Function to be used to tokenize messages in machine learning pipeline
    
    INPUTS:
        1. text - message text
        
    OUTPUTS:
        1. clean_tokens - list of tokens
    '''
    
    #use nltk.word_tokenize to perform tokenization
    tokens = word_tokenize(text)
    
    #initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    #lemmatize, convert to lower case and remove spaces
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(df, X_train, Y_train):
    '''
    Function builds machine learning pipeline
    
    INPUTS:
        1. df - original dataframe loaded from sqlite database
        2. X_train - training dataset to perform grid search for parameters
        3. Y_train - response dataset to perform grid search for parameters
        
    OUTPUTS:
        1. model - model to be used for message categories prediction
    '''
    
    #create model pipeline
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('genre', AddGenre(df)),
        ('tokens_num', TokensNumber())
    ])),
    ('clf', MultiOutputClassifier(xgb.XGBClassifier()))
    ])
    
    #initialize parameters dictionary to perform grid search
    parameters = {
    'features__text_pipeline__vect__max_df': (0.5, 1.0),
    'features__text_pipeline__tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators': [50, 100],
    'clf__estimator__learning_rate': [0.8, 1.0]}
    
    #use grid search to find best hyperparameters for the model
    cv = GridSearchCV(pipeline, param_grid=parameters, cv = 3, n_jobs = -1)
    cv.fit(X_train, Y_train)
    
    #initialize optimized pipeline
    model = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('genre', AddGenre(df)),
        ('tokens_num', TokensNumber())
    ])),
    ('clf', MultiOutputClassifier(xgb.XGBClassifier()))
    ])

    #set best hyperparameters for the model
    model.set_params(**cv.best_params_)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function prints performance metrics for the model for
    each category in category_names
    
    INPUTS:
        1. model - model for evaluation
        2. X_test - test dataset
        3. Y_test - actual response for test dataset
        4. category_names - names of categories to print performance metrics
    '''
    
    #get predictions
    Y_pred = model.predict(X_test)
    
    #print performance metrics
    for i in range(0, len(category_names)):
        print('Category: {category}'.format(category = category_names[i]))
        print(classification_report(Y_test[category_names[i]], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    Function saves created model
    
    INPUTS:
        1. model - model to be saved
        2. model_filepath - filepath where to save the model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df, X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(df, X_train, Y_train)
        
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