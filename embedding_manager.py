import sister
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.parallel


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


# Different version
class Encoder(torch.nn.Module):
    """ Encodes the given text input into a high dimensional embedding vector
        uses LSTM internally
    """

    def __init__(self, embedding_size, vocab_size, hidden_size, num_layers, device=torch.device("cpu")):
        """
        constructor of the class
        :param embedding_size: size of the input embeddings
        :param vocab_size: size of the vocabulary
        :param hidden_size: hidden size of the LSTM network
        :param num_layers: number of LSTM layers in the network
        :param device: device on which to run the Module
        """
        super(Encoder, self).__init__()

        # create the state:
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # create the LSTM layer:
        from torch.nn import Embedding, Sequential, LSTM
        self.network = Sequential(
            Embedding(self.vocab_size, self.embedding_size, padding_idx=0),
            LSTM(self.embedding_size, self.hidden_size,
                 self.num_layers, batch_first=True)
        ).to(device)

    def forward(self, x):
        """
        performs forward pass on the given data:
        :param x: input numeric sequence
        :return: enc_emb: encoded text embedding
        """
        output, (_, _) = self.network(x)
        return output[:, -1, :]  # return the deepest last (hidden state) embedding

