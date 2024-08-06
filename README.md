# TF-IDF From Scratch

This project implements Term Frequency-Inverse Document Frequency (TF-IDF) from scratch in Python. TF-IDF is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in information retrieval and text mining.

## Objective

The main goals of this implementation are:
1. To identify words that often appear in a document (high term frequency).
2. To ensure these words are relatively unique across the corpus (low document frequency, hence high inverse document frequency).

## Theory

### Term Frequency (TF)
Term Frequency (TF) measures the frequency of a word in a document. It is calculated as follows:
\[ \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d} \]

### Inverse Document Frequency (IDF)
Inverse Document Frequency (IDF) measures how important a term is. While computing TF, all terms are considered equally important. However, certain terms like "is", "of", and "that" may appear frequently but have little importance. Thus, we need to weigh down the frequent terms while scaling up the rare ones, by computing the following:
\[ \text{IDF}(t, D) = \log \left(\frac{\text{Total number of documents } |D|}{\text{Number of documents with term } t}\right) \]

### TF-IDF
TF-IDF is the product of TF and IDF:
\[ \text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D) \]

## Installation

To run this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- nltk

You can install the required libraries using pip:
```bash
pip install pandas numpy nltk
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/tfidf-from-scratch.git
    cd tfidf-from-scratch
    ```

2. Run the Jupyter notebook to see the implementation and results:
    ```bash
    jupyter notebook TFIDF_From_Scratch.ipynb
    ```

## Code Overview

The implementation involves the following steps:
1. Tokenizing the text data.
2. Calculating Term Frequency (TF).
3. Calculating Inverse Document Frequency (IDF).
4. Computing the TF-IDF score for each term in each document.

Here is a brief overview of the main sections of the code:

```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
nltk.download('punkt')  # Download data for tokenizer

# Example function to compute TF
def compute_tf(word_dict, doc):
    tf_dict = {}
    corpus_count = len(doc)
    for word, count in word_dict.items():
        tf_dict[word] = count / float(corpus_count)
    return tf_dict

# Example function to compute IDF
def compute_idf(doc_list):
    import math
    N = len(doc_list)
    idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
    for doc in doc_list:
        for word, val in doc.items():
            if val > 0:
                idf_dict[word] += 1
    for word, val in idf_dict.items():
        idf_dict[word] = math.log(N / float(val))
    return idf_dict

# Example function to compute TF-IDF
def compute_tfidf(tf_bow, idfs):
    tfidf = {}
    for word, val in tf_bow.items():
        tfidf[word] = val * idfs[word]
    return tfidf
```

## Contributing

We welcome contributions to enhance this project! Please fork the repository and create a pull request with your changes. Ensure your code follows best practices and includes appropriate tests.
