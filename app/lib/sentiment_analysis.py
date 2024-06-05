# library for standardizing input for model predictions
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

class SentimentAnalysis:

    def __init__(self, max_review_length=400, pad_type='pre', trunc_type='pre'):
        """Initialize MachineLearning with optional values.

        Args:
            max_review_length (int): Value to initialize standard review length.
            pad_type (string): Value to initialize padding type (pre or post).
            trunc_type (string): Value to initialize truncating type (pre or post). 
        """
        print(os.getcwd())
        self.max_review_length = max_review_length
        self.pad_type = pad_type
        self.trunc_type = trunc_type
        self.model = load_model('./model.h5')
        self.load_tokenizer()

    def load_tokenizer(self):
        """Helper function to laod tokenizer
        """
        self.word_index = imdb.get_word_index()
        self.word_index = { k: (v+3) for k,v in self.word_index.items()}
        self.word_index["PAD"] = 0
        self.word_index["START"] = 1
        self.word_index["UNK"] = 2
        self.index_word = {v:k for k,v in self.word_index.items()}

    def tokenize(self, review, word_index):
        """Perform tokenization of input based on model tokenization
        Args:
            review string: raw review data
            word_index dict: tokenization for trained dataset

        Returns:
            List of integers: result of tokenization
        """
        return [word_index.get(word, 2) for word in review.lower().split()]
    

    def standardize(self, data):
        """Standerdizes the input data in the same format as the trained model.
        Args:
            data (list of str): A list of reviews to be analyzed.

        Returns:
            list of floats: Sentiment analysis of each review submitted.
        """
        tokenized_reviews = [self.tokenize(review, self.word_index) for review in data]

        return pad_sequences(tokenized_reviews, 
                            maxlen = self.max_review_length,
                            padding = self.pad_type,
                            truncating = self.trunc_type), tokenized_reviews
    

    def predict(self, raw, padded_inputs, tokenized):
        """Wrapper function to return the sentiment analysis on formatted reviews.
        Args:
            data: (list of floats): List of tokenized reviews

        Returns:
            list of floats: List of predicted sentiment
        """
        predictions = self.model.predict(padded_inputs)

        data = {
            'Review': raw,
            'Tokenized': tokenized,
            'Padded': [list(padded) for padded in padded_inputs],
            'Predicted Sentiment': [pred[0] for pred in predictions]
        }

        data_frame = pd.DataFrame(data)
        print(data_frame)

        for i, review in enumerate(raw):
            predicted_sentiment = predictions[i][0]
            print(f"Review: {review}")
            print(f"Predicted Sentiment: {predicted_sentiment:.4f}")
            print(f"Tokenized: {tokenized[i]}")
            print(f"Padded: {padded_inputs[i]}")
            print(f"Decoded: {' '.join(self.index_word.get(id, 'UNK') for id in padded_inputs[i] if id != self.word_index['PAD'])}")
            print()

        return data_frame