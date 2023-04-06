import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import LSTM, CustomCNN


class ConvLSTM(nn.Module):
    def __init__(self, sequence_length, num_classes, cnn_layers=None,
                 cnn_input_dim=1, rnn_input_dim=256,
                 cnn_hidden_size=256, rnn_hidden_size=512, rnn_num_layers=1, rnn_dropout=0.0,):
        # NOTE: you can freely add hyperparameters argument
        super(ConvLSTM, self).__init__()

        # define the properties, you can freely modify or add hyperparameters
        self.cnn_hidden_size = cnn_hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_input_dim = cnn_input_dim
        self.rnn_input_dim = rnn_input_dim
        self.rnn_num_layers = rnn_num_layers
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.conv = CustomCNN()
        self.lstm = LSTM()
        # NOTE: you can define additional parameters
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs):
        """
        input is (imgaes, labels) (training phase) or images (test phase)
        images: sequential features of (Batch x Sequence_length, Channel=1, Height, Width)
        labels: (Batch x Sequence_length,)
        outputs should be a size of (Batch x Sequence_length, Num_classes)
        """

        # for teacher-forcing
        have_labels = False
        if len(inputs) == 2:
            have_labels = True
            images, labels = inputs
        else:
            images = inputs

        batch_size = images.size(0) // self.sequence_length
        hidden_state = torch.zeros((self.rnn_num_layers, batch_size, self.rnn_hidden_size)).to(images.device)
        cell_state = torch.zeros((self.rnn_num_layers, batch_size, self.rnn_hidden_size)).to(images.device)

        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem4: input image into CNN and RNN sequentially.
        # NOTE: you can use teacher-forcing using labels or not

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs  # (BxT, N_class)
