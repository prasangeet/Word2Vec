import os
import re

'''
This file will preprocess the data 
1. Remove the boilerplate text 
2. Tokenization
3. Lowercasting
4. Remove punctuation / non-text
5. Compute statistics
6. Word Count 
'''

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

"""
This Class will be used for preprocessing the text
"""
class PreprocessPipeline:

    def __init__(self, datadir, save_path):
        self.datadir = datadir
        self.save_path = save_path
        self.documents = []
        self.all_tokens = []

        self.stop_words = set(stopwords.words('english'))
        self.vocab = []

        self.word2idx = {}
        self.idx2word = {}
        self.encoded_docs = []

    def load_documents(self):
        texts = []

        for root, dirs, files in os.walk(self.datadir):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)

                    with open(path, "r", encoding = "utf-8") as f:
                        texts.append(f.read())

        return texts
    
    def clean_text(self, text: str):
        """
        Here we are cleaning the data:
        1. lowercasing the words
        2. Removing numbers from the data (Since numbers are just a noise)
        3. Match any character that is NOT a word character and NOT whitespace
        """
        text = text.lower()
        
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)

        return text

    def tokenize(self, text):
        """
        This function tokenizes the text 
        Basic tokenization
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Removes stopwords like is, and, are, was 
        """
        return [
            t for t in tokens
            if t not in self.stop_words
            and len(t) > 2
            and t.isalpha()
        ]
    
    def process_document(self, text):
        """
        In this function we will process a single document
        and output its tokens
        """
        text = self.clean_text(text)

        tokens = self.tokenize(text)

        tokens = self.remove_stopwords(tokens)

        return tokens

    def encode(self):
        """
        Builds the vocabulary and encodes all the tokens 
        into 2 variables:
            word2idx
            idx2word
        then encodes each documents
        """
        self.vocab = sorted(set(self.all_tokens))

        self.word2idx = {w : i for i, w in enumerate(self.vocab)}
        self.idx2word = {i : w for w, i in self.word2idx.items()}

        for doc in self.documents:
            encoded = []
            for word in doc:
                encoded.append(self.word2idx[word])
            self.encoded_docs.append(encoded)


    def run(self):
        """
        Main loop for running the PreprocessPipeline
        """
        texts = self.load_documents()

        for text in texts:
            tokens = self.process_document(text)
            if tokens:
                self.documents.append(tokens)
                self.all_tokens.extend(tokens)

        self.encode()

        return self.documents, self.vocab, self.encoded_docs

    def save(self):
        """
        We will save the CORPUS
        """

        with open(self.save_path, "w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(" ".join(doc) + "\n")

    def stats(self):
        """
        Returns the statistics of the current data 
        """
        total_docs = len(self.documents)

        total_tokens = len(self.all_tokens)

        vocab_size = len(self.vocab)

        return total_docs, total_tokens, vocab_size
