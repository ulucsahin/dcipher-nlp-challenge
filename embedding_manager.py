import sister
import numpy as np

class Embedder(object):
    """
     Uses Sister (https://github.com/tofunlp/sister) which uses FastText.
     Generates 300 dimension sentence - word embeddings.
    """
    def __init__(self, embedding_size):
        self.setup_model()
        self.embedding_size = embedding_size

    def setup_model(self):
        # FastText
        self.model = sister.MeanEmbedding(lang="en")

    def get_embeddings(self, sentences):
        # FastText
        embeddings = np.zeros((len(sentences), self.embedding_size))
        for i, sentence in enumerate(sentences):
            embeddings[i] = self.model(sentence)

        return embeddings

    def get_word_embeddings(self, tokenized_sentences):
        # FastText
        amount_of_sentences = len(tokenized_sentences)
        amount_of_words = len(tokenized_sentences[0])
        embedding_size = self.embedding_size

        embeddings = np.zeros((amount_of_sentences, amount_of_words, embedding_size))
        for i, tokens in enumerate(tokenized_sentences):
            embeddings[i] = self.model.word_embedder.get_word_vectors(tokens)

        return embeddings

