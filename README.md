# Udacity DSND Disaster Response Pipeline Project
Web-app which uses NLP to categorize messages related to a disaster.

## Table of Contents
* [About the Project](#about-the-project)
* [Project Results](#project-results)
    - [Model](#model)
    - [Web-app](#web-app)
* [Repository Contents](#repository-contents)
* [Setup Instructions](#setup-instructions)
* [External Lobraries](#external-libraries)

## About the Project
During a disaster, it is crucial to respond quickly to people's needs, which are expressed in messages sent across various channels. Machine learning algorithms using NLP could help to categorize messages so that they can be sent to appropriate disaster relief agencies. 

This project uses a data set containing real messages that were sent during disaster events provided by [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/). The goal of the project is to create a machine learning pipeline to categorize these events so that they can be sent to appropriate disaster relief agency.

This project also includes a web app where an emergency worker can input a new message and get classification results in several categories.

## Project Results

### Model
The resulting model uses NLTK library to perform tokenization and lemmatization of each provided message and XGBoost classifier to classify messages. More detailed description of machine learning pipeline is provided in [Repository Contents](#repository-contents) section below within the description of `train_classifier.py` script.

The resulting model provides the message classification for 36 categories, such as 'Food', 'Water', 'Medical Products' etc. The training dataset is very imbalanced for some of the categories, and this fact affects the model performance metrics (precision and recall) greately. For example, the category 'child_alone' is never used in the whole training dataset. On the contrary, the category 'related' is used for almost all of the messages in the dataset. That is why expanding the training dataset in order to make it more balanced could improve the resulting model performance.

### Web-app
The main page of the web-app contains visualizations of the training dataset used to create the machine learning model:
![Web-app main page](https://github.com/Lexie88rus/Udacity-DSND-Disaster-Response/blob/master/screenshots/app_main_screenshot.png)

The user is also able to input message and get classification results by pressing the button 'Classify Message'. See the page with the classification results:
![results page](https://github.com/Lexie88rus/Udacity-DSND-Disaster-Response/blob/master/screenshots/app_classify_screenshot.png)

## Repository Contents
The repository has the following structure:
```
- app
| - templates
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
| - static
| |- githublogo.png  # github logo used in the main page
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # script containing ETL for the initial dataset processing
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py # script containing machine learning pipeline to create the classifier 
|- classifier.pkl  # saved model 

- screenshots
|- app_classify_screenshot.png # screenshot of the main page
|- app_main_screenshot.png # screenshot of the page containing classification results

- README.md
```
More details on the most important repository files and scripts:
1. __disaster_messages.csv__ - dataset in csv format, which contains columns for the ids of the messages, messages text in English and in the original language, genre for each message ('direct', 'social', 'news').
2. __disaster_categories.csv__ - dataset in csv format, which contains ids of the messages and categories for each message in the following format: `related-1;request-0;offer-0;aid_related-0;...`.
3. __process_data.py__ script contains ETL pipeline to take the following steps to process data from the initial datasets:
    - combine initial csv datasets into one,
    - split the column, which contains message categories into 36 separate columns for each category,
    - convert all categories to binary features,
    - load the resulting data into sqlite database `DisasterResponse.db`.
4. __train_classifier.py__ script contains machine learning pipeline to build and evaluate the model to categorize messages:
    - load prepared data from sqlite database `DisasterResponse.db`,
    - build the model using sklearn pipeline, which takes CountVectorizer, TfidfTransformer and XGBoost classifier wrapped in MultiOutputClassifier,
    - perform grid search to find the best hyperparameters for the pipeline,
    - print out the resulting performance scores for the model with the defined hyperparameters,
    - save resulting model into `classifier.pkl` file to be used by the app.
    
## Setup Instructions
Follow the instructions to run the web-app locally:
1. Install required external libraries (see the [External Libraries](#external-libraries) section below).
2. Clone the repository.
3. Navigate to the repository's root directory.
4. Run the following commands in the repository's root directory to set up database and model.

    - Run ETL pipeline that cleans data and stores in database
        <br>`$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - Run ML pipeline that trains and saves the classifier __please, note that building the model takes at least 30 minutes!__
        <br>`$ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

5. Navigate to `app` folder.
6. Run the following command in the app's directory to run the web app:
    <br>`$ python run.py`

4. Open http://0.0.0.0:3001/ in web browser.

## External Libraries
The following external libraries should be installed in order to run the app:
1. [SQLAlchemy](https://www.sqlalchemy.org) to store the preprocessed data,
2. [XGBoost](https://xgboost.readthedocs.io/en/latest/) library for machine learning model,
3. [NLTK](http://www.nltk.org) library for message text processing,
4. [Plot.ly](https://plot.ly/) library for visualization,
5. [Flask](http://flask.pocoo.org/docs/1.0/) to run the web-app locally.
