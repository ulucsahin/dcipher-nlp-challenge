import sister
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.parallel


class Embedder(object):
    """
     Uses Sister (https://github.com/tofunlp/sister) which uses FastText.
     Generates 300 dimension sentence embeddings.
    """
    def __init__(self):
        self.setup_model()

    def setup_model(self):
        # INFERSENT
        # params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        # self.model = InferSent(params_model)
        # self.model.load_state_dict(torch.load(Args.infersent_path))
        # self.model.cuda()
        # self.model.set_w2v_path(Args.glove_path)
        # print("Creating vocabulary for InferSent model.")
        # self.model.build_vocab_k_words(K=100000)


        # Sister (FastText)
        self.model = sister.MeanEmbedding(lang="en")

    def get_embeddings(self, sentences):
        # InferSent
        #embeddings = self.model.encode(sentences, bsize=128, tokenize=False, verbose=True)

        # Sister
        embeddings = np.zeros((len(sentences), 300))
        for i, sentence in enumerate(sentences):
            embeddings[i] = self.model(sentence)

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

