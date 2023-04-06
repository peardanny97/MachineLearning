import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, ):
        # NOTE: you can freely add hyperparameters argument
        super(CustomCNN, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # define cnn model
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs):
        """
        inputs: (Batch x Sequence_length, Channel=1, Height, Width)
        outputs: (Sequence_length, Batch_size, Hidden_dim)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1: code CNN forward path
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()

        # define the properties
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # lstm cell
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem2: Define input, output projection layer to fit dimension
        # output fully connected layer to project to the size of the class
        self.fc_in =
        self.fc_out =
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, feature, h, c):
        """
        feature: (Sequence_length, Batch_size, Input_dim)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem3: Design LSTM model for letter sorting
        # NOTE: sequence length of feature can be various

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return output, h_next, c_next  # (sequence_lenth, batch, num_classes), (num_rnn_layers, batch, hidden_dim), (num_rnn_layers, batch, hidden_dim)
