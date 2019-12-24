# Disaster Response Pipeline Project

Summary of the project
This project is mainly about building a ETL pipeline and a machine learning pipeline and a web app for better disaster response.


Explanation of the files in the repository
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
 
 An example of the web app
 
 ![Visualization1](/image/pic1.png)
 ![Visualization2](/image/pic2.png)
 ![Visualization3](/image/pic3.png)
