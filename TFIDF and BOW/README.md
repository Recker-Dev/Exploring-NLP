# Text Classification Model - BoW vs TF-IDF with Naive Bayes

## Project Description

This project implements a text classification model using the Naive Bayes algorithm to classify text data into different categories. Two vectorization techniques are employed: **Bag of Words (BoW)** and **Term Frequency-Inverse Document Frequency (TF-IDF)**. These techniques are used to convert raw text data into numerical features that can be fed into a machine learning model. The Naive Bayes classifier is trained on these features to make predictions.

- **Bag of Words (BoW)**: A simple technique where each word in the text corpus is treated as a feature.
- **TF-IDF**: A more advanced technique that weighs words based on their importance in the document relative to the corpus.

This project uses the **Naive Bayes** classifier to predict whether the text is categorized as positive or negative. However, due to the simplicity of the model and text data, the accuracy may not be very high.

## Model Accuracy

- **BoW Accuracy**: 33.86%
- **TF-IDF Accuracy**: 34.84%

Please note that the performance of the model is not optimal due to the limitations of the Naive Bayes algorithm and the choice of features. These models should be viewed as a basic starting point, and further improvements can be made by exploring more advanced algorithms and feature engineering techniques.

## Installation

To set up the project and install the required dependencies, follow these steps:

```bash
pip install -r requirements.txt
```

## Notebooks

### 1. `eda_and_preProcessing.ipynb`

This notebook contains the **Exploratory Data Analysis (EDA)** and **Preprocessing** steps for the dataset. It includes:

- Data inspection and basic analysis to understand the structure of the data.
- Handling missing values, text cleaning, and any necessary transformations.
- Visualization of data distributions and relationships to better understand the dataset.
- Preprocessing steps such as tokenization, stop-word removal, and vectorization using both **Bag of Words (BoW)** and **TF-IDF**.

### 2. `INFERENCE_CODE_bow_and_tfid.ipynb`

This notebook is used for **making predictions** with the trained models. It includes:

- Loading the saved models (`nb_model_bow.pkl` and `nb_model_tfidf.pkl`) and vectorizers (`bow_vectorizer.pkl` and `tfidf_vectorizer.pkl`).
- Transforming custom input text using the vectorizers into the correct format for prediction.
- Running predictions using both **BoW** and **TF-IDF** models and displaying the results.
- A simple interface to test the models on custom inputs.

### 3. `TRAINING_CODE_bow_and_tfid.ipynb`

This notebook contains the **training code** for both the BoW and TF-IDF models. It includes:

- Loading and splitting the dataset into training and testing sets.
- Preprocessing the data using **Bag of Words (BoW)** and **TF-IDF** vectorization.
- Training two Naive Bayes models: one with BoW features and one with TF-IDF features.
- Saving the trained models and vectorizers for later use in inference.
- Evaluation of model accuracy using the test set.

## License

This project is licensed under the MIT License.

## Folder Structure

```
.
├── artifacts
│   └── models
│       ├── nb_model_bow.pkl
│       └── nb_model_tfidf.pkl
│   └── vectorizer
│       ├── bow_vectorizer.pkl
│       └── tfidf_vectorizer.pkl
└── dataset
    ├── AmazonData.csv
    └── processed_reviews.csv
├── eda_and_preProcessing.ipynb
├── INFERENCE_CODE_bow_and_tfid.ipynb
├── TRAINING_CODE_bow_and_tfid.ipynb
└── README.md
```
