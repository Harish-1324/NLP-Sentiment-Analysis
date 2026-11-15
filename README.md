# Tweet Sentiment Analysis Using Deep Learning (CNN + BiLSTM)

Project documentation based on the Sentiment140 Twitter dataset.

| | |
|---|---|
| **Dataset** | Sentiment140 (1.6 million tweets) |
| **Task** | Binary text classification (Positive / Negative) |
| **Implementation** | Python, TensorFlow / Keras, NLTK, scikit-learn |
| **Model** | CNN + Bidirectional LSTM with GloVe embeddings |
| **Document type** | Project README |

---

## 1. Project Overview

Tweet Sentiment Analysis is a deep learning and natural language processing project that classifies a tweet as **Positive** or **Negative**. The project is a Jupyter notebook (`Notebook_analyse_sentiment.ipynb`) that covers the complete workflow: importing the data, preprocessing the text, encoding and vectorising it, building an LSTM-based model, training it, and analysing its predictions.

The model combines a 1D convolution layer with a Bidirectional LSTM and uses pre-trained **GloVe** word vectors to represent words.

## 2. Problem Statement

Millions of opinions are shared on social media every day, and reading them manually is impossible. The objective is to build an automated system that reads a short, informal piece of text and predicts whether the sentiment is positive or negative, while measuring how reliably it performs on unseen tweets.

## 3. Dataset Description

The project uses the **Sentiment140** dataset (`training.1600000.processed.noemoticon.csv`).

- Total tweets: 1,600,000 (6 columns, no missing values)
- Negative tweets (label 0): 800,000
- Positive tweets (label 4): 800,000

Columns: `label`, `time`, `date`, `query`, `user_id`, `text`.

To make training faster, the notebook draws a **balanced sample of 100,000 tweets** (50,000 per class) and splits it randomly (80% / 20%, `random_state=7`):

| Set | Tweets |
|---|---|
| Training | 80,000 |
| Test | 20,000 (10,024 Negative, 9,976 Positive) |

- **Dataset source:** [Sentiment140](http://help.sentiment140.com/for-students)
- **Word embeddings:** [GloVe 6B (300-dimensional)](https://nlp.stanford.edu/projects/glove/), 400,000 word vectors

## 4. Project Objectives

- Import the dataset and the natural language processing modules.
- Clean and preprocess the tweet text.
- Encode and vectorise the data using tokenisation, padding and GloVe embeddings.
- Build a CNN + Bidirectional LSTM model.
- Train the model and monitor loss and accuracy.
- Predict sentiment on unseen tweets and analyse the results.

## 5. Technologies and Libraries

- **Programming language:** Python 3
- **Data handling:** pandas and NumPy
- **Text processing:** NLTK (stop words, tokenisation), `re`, `string`
- **Deep learning:** TensorFlow / Keras (`Embedding`, `Conv1D`, `Bidirectional LSTM`, `Dense`, `Dropout`)
- **Machine learning utilities:** scikit-learn (`train_test_split`, `LabelEncoder`, `classification_report`, `confusion_matrix`)
- **Visualisation:** Matplotlib
- **Development environment:** Jupyter Notebook

## 6. System Workflow

1. **Data loading:** read the Sentiment140 CSV with pandas and name the columns.
2. **Data preparation:** check for null values, draw a balanced sample of 100,000 tweets and map labels to Negative / Positive.
3. **Text cleaning:** convert to lower case and remove stop words, punctuation, repeated characters, @mentions, URLs and numbers.
4. **Tokenisation:** split the cleaned text into words with a regular-expression tokenizer.
5. **Dataset splitting:** 80% training and 20% test with a fixed random seed.
6. **Encoding and vectorisation:** convert tweets to padded sequences of 30 tokens (vocabulary of 84,758 words), encode the labels, and build a 300-dimensional GloVe embedding matrix.
7. **Model building:** SpatialDropout (0.2) → Conv1D (64 filters, kernel 5) → Bidirectional LSTM (64 units) → Dense 512 → Dropout (0.5) → Dense 512 → Dense 1 (sigmoid).
8. **Model training:** Adam optimiser (learning rate 0.01), binary cross-entropy loss, batch size 1,000, up to 20 epochs, with early stopping and learning-rate decay.
9. **Prediction and evaluation:** classify the test tweets and report accuracy, precision, recall, F1-score and a confusion matrix.

## 7. Project Structure

The notebook expects the following layout (paths used inside the notebook):

```text
NLP-Sentiment-Analysis/
├── data/
│   └── training.1600000.processed.noemoticon.csv   # Sentiment140 dataset
├── glove.6B/
│   └── glove.6B.300d.txt                           # GloVe word vectors
├── save/
│   └── model_avanced_lstm/                         # Saved model (.h5)
├── model_checkpoints/                              # Training checkpoints
├── Notebook_analyse_sentiment.ipynb
└── README.md
```

## 8. Installation and Setup

1. Install Python 3 and Jupyter Notebook.
2. Download the Sentiment140 dataset and place the CSV file in the `data` folder.
3. Download `glove.6B.zip` from the GloVe website and extract `glove.6B.300d.txt` into the `glove.6B` folder.
4. Clone the repository and open a terminal in the project directory.
5. Install the required libraries.

```bash
git clone https://github.com/Harish-1324/NLP-Sentiment-Analysis.git
cd NLP-Sentiment-Analysis

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install numpy pandas matplotlib scikit-learn tensorflow nltk jupyter
```

6. Download the NLTK stop words once:

```python
import nltk
nltk.download('stopwords')
```

> The notebook uses `keras.utils.pad_sequences`, so use a recent TensorFlow 2.x version.

## 9. Running the Project

Start Jupyter and run the notebook from top to bottom:

```bash
jupyter notebook Notebook_analyse_sentiment.ipynb
```

Check that the `path`, `glove_path` and model-saving paths in the notebook match your folder layout. Training the model on a CPU takes about one minute per epoch.

## 10. Model Evaluation

The model is evaluated on 20,000 held-out test tweets.

| Metric | Value |
|---|---|
| Training accuracy (final epoch) | 80.66% |
| Test accuracy | 73% (72.96% at the final epoch, best 73.74% at epoch 6) |
| Training run | Stopped by early stopping after 10 of 20 epochs |

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Negative | 0.73 | 0.74 | 0.73 | 10,024 |
| Positive | 0.73 | 0.72 | 0.73 | 9,976 |
| **Accuracy** | | | **0.73** | 20,000 |

Model size: 25,918,729 parameters, of which 491,329 are trainable (the GloVe embedding layer is frozen).

## 11. Expected Output

For each tweet, the model outputs a score between 0 and 1; a score above 0.5 is labelled **Positive**, otherwise **Negative**. The notebook also plots the accuracy and loss curves per epoch and a normalised confusion matrix.

## 12. Limitations

- Only a 100,000-tweet sample of the 1.6 million tweets is used.
- Validation and test data are the same 20,000 tweets, so a separate validation set is recommended.
- Validation loss rises after epoch 5 while training accuracy keeps increasing, which indicates overfitting.
- Tweets are short, informal and noisy, which limits accuracy at this model size.
- The dataset has only positive and negative classes; neutral sentiment is not covered.
- Tweet content can contain personal information; handle data responsibly.

## 13. Future Enhancements

- Tune hyperparameters with Keras Tuner after training on the full dataset.
- Train on all 1.6 million tweets.
- Restore the best epoch's weights and use a lower learning rate.
- Use a separate validation set from the test set.
- Try pre-trained transformer models such as BERT.
- Build a simple web interface for entering a tweet and viewing the predicted sentiment.

## 14. Conclusion

This project shows how deep learning and natural language processing can be applied to sentiment analysis. By cleaning tweet text, representing words with GloVe embeddings and training a CNN + Bidirectional LSTM, the model reaches about 73% accuracy with balanced precision and recall on both classes, and provides a base for further tuning.

## 15. References

- Go, A., Bhayani, R. and Huang, L. (2009). "Twitter Sentiment Classification using Distant Supervision." Stanford University.
- Pennington, J., Socher, R. and Manning, C. (2014). "GloVe: Global Vectors for Word Representation." EMNLP 2014.
- Sentiment140 dataset: <http://help.sentiment140.com/for-students>
- GloVe word vectors: <https://nlp.stanford.edu/projects/glove/>
- TensorFlow / Keras documentation: <https://www.tensorflow.org/>

> **Dataset note:** Follow the dataset providers' documentation and attribution requirements when redistributing or publishing results.
