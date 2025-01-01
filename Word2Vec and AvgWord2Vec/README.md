# Project Title: Text Classification with Word2Vec and Random Forest Classifier

## Project Description

This project focuses on classifying text data using a machine learning pipeline. The pipeline leverages Word2Vec embeddings to represent text data and a Random Forest classifier for classification. The primary steps include text preprocessing, feature extraction, model training, and evaluation.

## Project Workflow

1. **Text Preprocessing**: The text data is tokenized, lemmatized, and converted into a corpus.
2. **Feature Extraction**: Word2Vec is used to generate vector representations for the text. To achieve sentence-level representation, the average of word vectors is computed for each sentence, resulting in a 100-dimensional vector per sentence for better generalization.
3. **Model Training**: A Random Forest classifier is trained using the generated feature vectors.
4. **Evaluation**: The model is evaluated using metrics such as accuracy, classification report, and confusion matrix.

## Installation

To set up the project environment, install the required dependencies:

```bash
pip install -r requirements.txt
```
## Project Usage Guide

**1. Data Preparation:**

- The raw data is located in the `dataset` folder as `AmazonData.csv`.

**2. Data Exploration and Preprocessing:**

- Execute the `eda_and_preProcessing.ipynb` notebook to perform exploratory data analysis and necessary preprocessing steps on the raw data. This may include:
    - Data cleaning (handling missing values, removing duplicates)
    - Feature engineering (creating new features)
    - Text preprocessing (lowercasing, removing punctuation, stop word removal, lemmatization)
    - Saving the preprocessed data to `processed_reviews.csv` for later use.

**3. Model Training:**

- Run the `TRAINING_CODE_Word2Vec and AvgWord2Vec.ipynb` notebook. This notebook will:
    - Train a Word2Vec model on the preprocessed reviews.
    - Generate average word embeddings for each review.
    - Train a machine learning model (e.g., Random Forest) using the average word embeddings as features.
    - Save the trained model to `artifacts/classifier.pkl` and the Word2Vec model to `artifacts/w2v_model.pkl`.

**4. Inference and Prediction:**

- Execute the `INFERENCE_CODE_Word2Vec and AvgWord2Vec.ipynb` notebook. This notebook will:
    - Load the trained model and Word2Vec model from the `artifacts` folder.
    - Load new, unseen data (if applicable).
    - Preprocess the new data using the same steps as in `eda_and_preProcessing.ipynb`.
    - Generate average word embeddings for the new data using the loaded Word2Vec model.
    - Use the trained model to make predictions on the new data.
    - Evaluate the model's performance on the new data (if available).

## File Descriptions

- **w2v_model.pkl**: The pickled Word2Vec model.
- **classifier.pkl**: The pickled Random Forest classifier.
- **requirements.txt**: Lists all the dependencies needed for the project.

## Dependencies

Dependencies are specified in the `requirements.txt` file and include libraries such as:

- NLTK
- Gensim
- Scikit-learn
- NumPy

## License

This project is licensed under the MIT License.

## Folder Structure

```
├── artifacts
│ ├── classifier.pkl
│ └── w2v_model.pkl
├── dataset
│ ├── AmazonData.csv
│ └── processed_reviews.csv
├── eda_and_preProcessing.ipynb
├── INFERENCE_CODE_Word2Vec and AvgWord2Vec.ipynb
├── TRAINING_CODE_Word2Vec and AvgWord2Vec.ipynb
├── README.md
└── requirements.txt
```