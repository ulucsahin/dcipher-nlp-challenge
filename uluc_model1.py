# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import math

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

    def conv_block(self, input, conv_layer):
        # input.shape: batch_size.1.token_size.embed_size
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
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



######## MODEL 2 #########
class TextClassifier(nn.ModuleList):

    def __init__(self, seq_len, num_words, out_size, stride):
        super(TextClassifier, self).__init__()

        # Parameters regarding text preprocessing
        self.seq_len = seq_len
        self.num_words = num_words
        self.embedding_size = 300

        # Dropout definition
        self.dropout = nn.Dropout(0.25)

        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5

        # Output size for each convolution
        self.out_size = out_size
        # Number of strides for each convolution
        self.stride = stride

        # Embedding layer definition
        #self.embedding = nn.Embedding(self.num_words + 1, self.embedding_size, padding_idx=0)

        # Convolution layers definition
        self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
        # print(self.seq_len) # 300
        # print(self.out_size) # 50
        # print(self.kernel_1) # 2
        # print(self.stride) # 1

        self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
        self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)

        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        # Fully connected layer definition
        self.fc = nn.Linear(self.in_features_fc(), 1)

    def in_features_fc(self):
        '''Calculates the number of output features after Convolution + Max pooling

        Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''
        # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)

        # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
        out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_pool_3 = math.floor(out_pool_3)

        # Calcualte size of convolved/pooled features for convolution_4/max_pooling_4 features
        out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_conv_4 = math.floor(out_conv_4)
        out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_pool_4 = math.floor(out_pool_4)

        # Returns "flattened" vector (input for fully connected layer)
        return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_size

    def forward(self, x):
        # Sequence of tokes is filterd through an embedding layer
        #x = self.embedding(x)


        # Convolution layer 1 is applied
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)

        # Convolution layer 2 is applied
        x2 = self.conv_2(x)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)

        # Convolution layer 3 is applied
        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        # Convolution layer 4 is applied
        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)

        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)

        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(union)
        # Dropout is applied
        out = self.dropout(out)
        # Activation function is applied
        out = torch.sigmoid(out)

        return out.squeeze()