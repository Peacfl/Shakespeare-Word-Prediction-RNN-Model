# Shakespeare-Word-Prediction-RNN-Model

This application demonstrates how deep learning can be applied to generate text in the style of Shakespeare.
Users can input a line or phrase, and the trained model predicts the next word based on learned patterns from the training corpus.

The app uses a trained LSTM-based language model along with tokenizers to convert text to sequences and back.

![image1](/word-pred-1.png?raw=true "Optional Title")

![image2](/word-pred-2.png?raw=true "Optional Title")

## Dataset

The training corpus is sourced from:
Project Gutenberg – Shakespeare Collection
The text was cleaned, tokenised, and transformed into sequences used for next-word prediction.

## Model Details
- Embedding Layer (128 dims): Learns vector representations of words
- LSTM Layer (256 units): Captures long-term dependencies in Shakespearean language
- Dropout + Recurrent Dropout: Helps reduce overfitting
- Dense Output Layer: Predicts the probability distribution of the next word

## Streamlit App
Write some text and adjust the creativity and text size to predict.

## Run the App
streamlit run app.py
