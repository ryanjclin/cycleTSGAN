import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, batch_size, var_num, seq_len):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        self.var_num = var_num
        self.seq_len = seq_len

        self.fc1 = nn.Linear(self.var_num, 256)
        self.lstm1 = self.lstm_layer(256, 128, 1)
        self.fc2 = nn.Linear(128, 1)
        
    def lstm_layer(self, input_size, hidden_size, num_layers, bidirectional=False):
        module = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first = True,)
        return module

    def forward(self, x):
        '''x : [batch_size, var_num, seq_len]'''
        
        x = x.reshape((self.batch_size, self.seq_len, self.var_num))

        x = F.relu(self.fc1(x))
        x, (hn, cn) = self.lstm1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = x.reshape((self.batch_size, self.seq_len*1))

        return x




    #     self.fc1 = nn.Linear(self.seq_len, 256)
    #     self.fc2 = nn.Linear(256, 256)
    #     self.fc3 = nn.Linear(256, 512)
    #     self.fc4 = nn.Linear(512, 512)
    #     self.fc5 = nn.Linear(512, 256)
    #     self.fc6 = nn.Linear(256, 128)
    #     self.fc7 = nn.Linear(128, 32)
    #     self.fc8 = nn.Linear(32, 1)

    # def forward(self, x):
    #     '''x : [batch_size, var_num, seq_len]'''

    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = F.relu(self.fc3(x))
    #     x = F.relu(self.fc4(x))
    #     x = F.relu(self.fc5(x))
    #     x = F.relu(self.fc6(x))
    #     x = F.relu(self.fc7(x))
    #     x = self.fc8(x)

    #     return x
    





        self.fc1 = nn.Linear(self.var_num, 256)
        self.lstm1 = self.lstm_layer(256, 128, 1)
        self.fc2 = nn.Linear(128, 1)
        
    def lstm_layer(self, input_size, hidden_size, num_layers, bidirectional=False):
        module = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first = True,)
        return module

    def forward(self, x):
        '''x : [batch_size, var_num, seq_len]'''
        
        x = x.reshape((self.batch_size, self.seq_len, self.var_num))

        x = F.relu(self.fc1(x))
        x, (hn, cn) = self.lstm1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = x.reshape((self.batch_size, self.seq_len*1))

        return x