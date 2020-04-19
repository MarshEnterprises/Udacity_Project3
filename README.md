# Disaster Response Pipeline Project

### Background:
This project is a requirement for project 3 of the Udacity Data Science Nanodegree. The idea is to build a machine learning pipeline to evaluate plaintext in the context of disaster response. Training data is provided as classified plaintext in csv format. The outcome is a flask app which categorises plaintext messages into 36 applicable disaster response classes.

### Prerequisites
The following needs to be installed:
- python 3.7
- json
- plotly
- pandas
- nltk
- flask
- plotly
- sklearn
- sqlalchemy
- sys
- re
- pickle
- numpy

### Data
- `./data/disaster_messages.csv` gives messages in plaintext.
- `./data/disaster_categories.csv` gives the 36 target classes.
These csv files are imported and joined using an inner merge. 

### Running the model
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

