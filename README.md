# Disaster-response-pipeline
This project was made to fullfill the udacity data science nano degree 2nd project specs

This project is a fully functional machine learning pipeline from raw data to model deployment using a flask app. I used the Disaster responce massages data prodvided by Udacity for the data science nano degree course .

Files In thie repo:

-README explanation and intro to the project

-app  -run.py : the main app file that runs the fianl model and the app
      -templates: html templates for the web page 

-data -process_data.py: where cleaning the training takes places
      -Training data files in csv format
      -DisasterResponsedb: where teh cleaned data is stored for use by the model

-Model -train_classiffier.py: teh script responsible for training and storing the model
       -model_pkl: a trained model stored in pickle format which I was unable to upload because it was too large
       

Summary and findings:

The data pipeline works fine after trying a few different models I reached the conclusion that random forest calssifiers are the most suitable for this poject beacause they achieve a balance between low relative computational time and high scores in recall prescision and f1 score.

Finaly Acknowledgments:
disaster data from Appen (formally Figure 8).
