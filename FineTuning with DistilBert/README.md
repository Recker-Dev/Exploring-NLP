
# Project Description
This project uses the Amazon Customer Review dataset to fine-tune a DistilBERT model for sentiment analysis. The goal is to classify customer reviews into different sentiment categories, such as very positive, positive, neutral, negative, and very negative. By leveraging the DistilBERT transformer model, which is known for its efficiency and performance, we aim to create an accurate sentiment analysis tool for Amazon product reviews.


# Amazon Review Dataset for Multi-Label Classification

This repository contains code for handling an Amazon Review dataset for multi-label classification using `DistilBERT`. The `dataset` folder contains the training data for the model. The `artifacts` folder contains the trained model and the label-encoder needed to backtrack during inferencing.

The Training Code is provided in : `TRAINING_CODE_DistilBert-FineTuned-Amazon-Sentiment-Analysis.ipynb`

The Inference/Testing Code is provided in :
`INFERENCE_CODE_DistilBert-FineTuned-Amazon-Sentiment-Analysis.ipynb`

## Installation

To use this code, you need to install the following dependencies:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.

## Project Structure

```
├── artifacts
│   └── model/distilbert_amazon_review_model
│       ├── training_args.bin
│       ├── config.json
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── model.safetensors
│       ├── vocab.txt
│       └── label_encoder.pkl
│   └── dataset
│       └── AmazonData.csv
│   └── INFERENCE_CODE_DistilBert-FineTuned-Amazon-Senti...
│   └── TRAINING_CODE_DistilBert-FineTuned-Amazon-Senti...
│   └── README.md
│   └── requirements.txt

```
