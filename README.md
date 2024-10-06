# SpamSMS


This project demonstrates the use of various machine learning models to classify SMS messages as either "Spam" or "Ham" (legitimate). The dataset used is a CSV file containing labeled SMS messages, which is preprocessed and transformed into TF-IDF features. Five different models are trained and evaluated for accuracy, allowing for easy comparison of their performance.

Models Used
Naive Bayes (MultinomialNB)
Logistic Regression
Support Vector Machine (SVC)
Random Forest
K-Nearest Neighbors (KNN)
Dataset
The dataset used for this project is the spam.csv, which contains SMS messages labeled as either spam or ham.

Columns:
v1: Label (either spam or ham)
v2: The message (text content)
Prerequisites
Ensure you have the following libraries installed:

pandas
numpy
scikit-learn
tqdm


1. Load Data
The project starts by loading the dataset and performing basic cleaning operations, such as removing duplicates.

2. Preprocessing
Text data is transformed into TF-IDF vectors using TfidfVectorizer, which converts the raw text into numerical features suitable for machine learning algorithms.

3. Model Training and Evaluation
Five different models are trained on the dataset:

Multinomial Naive Bayes
Logistic Regression
Support Vector Machines
Random Forest
K-Nearest Neighbors
Each model is trained using the fit() method and then tested on the test set. The accuracy_score is calculated for each model to evaluate its performance.

4. Results
The accuracy of each model is printed for comparison, and the model with the highest accuracy is highlighted as the best model.


Customization
Add New Models: You can easily add more models by extending the models dictionary.
Tuning Hyperparameters: The current models are run with default settings. You can tune the hyperparameters of each model to potentially improve performance.
Preprocessing: Additional text preprocessing techniques (like stemming, lemmatization, or removing punctuation) can be incorporated.

Conclusion
This project serves as a basic template for comparing different machine learning models on a text classification problem. By leveraging the power of multiple algorithms, it becomes easier to determine which model best fits your data.


