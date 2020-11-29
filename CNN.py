import torch
import torch.nn as nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
        keep_probab : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        --------

        """

        in_channels = config.in_channels
        out_channels = config.out_channels
        kernel_heights = config.kernel_heights
        stride = config.stride
        padding = config.padding
        embedding_length = config.embedding_length

        self.title_conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
        torch.nn.init.xavier_uniform_(self.title_conv1.weight)
        self.title_conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
        torch.nn.init.xavier_uniform_(self.title_conv2.weight)
        self.title_conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
        torch.nn.init.xavier_uniform_(self.title_conv2.weight)

        self.abstract_conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
        torch.nn.init.xavier_uniform_(self.abstract_conv1.weight)
        self.abstract_conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
        torch.nn.init.xavier_uniform_(self.abstract_conv2.weight)
        self.abstract_conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
        torch.nn.init.xavier_uniform_(self.abstract_conv3.weight)

        self.dropout = nn.Dropout(config.keep_probab)
        # *2 since we have title and abstract
        self.fc1 = nn.Linear(len(kernel_heights) * out_channels * 2, 30)
        self.fc2 = nn.Linear(30, config.output_size)

    def conv_block(self, input_, conv_layer):
        # input.shape: batch_size.1.token_size.embed_size
        conv_out = conv_layer(input_)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, title_embeddings, abstract_embeddings, batch_size=None):
        """
        Parameters
        ----------
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        abstract_embeddings: embeddings obtained from abstract
        title_embeddings: embeddings obtained from title

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        """

        title_input = title_embeddings
        abstract_input = abstract_embeddings # 16, 1, token_amount , 300

        title_out1 = self.conv_block(title_input, self.title_conv1)
        title_out2 = self.conv_block(title_input, self.title_conv2)
        title_out3 = self.conv_block(title_input, self.title_conv3)

        abstract_out1 = self.conv_block(abstract_input, self.abstract_conv1) # 16, 100
        abstract_out2 = self.conv_block(abstract_input, self.abstract_conv2) # 16, 100
        abstract_out3 = self.conv_block(abstract_input, self.abstract_conv3) # 16, 100

        all_out = torch.cat((title_out1, title_out2, title_out3, abstract_out1, abstract_out2, abstract_out3), 1)  # 16, 300
        fc_in = self.dropout(all_out) # 16, 2
        logits = self.fc1(fc_in)
        logits = self.fc2(logits) # 16, 2

        return logits
