import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first = True, dropout = 0.1, bidirectional = False, num_layers = 1, bias = True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    def __init__hidden(self,batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),torch.zeros(1, batch_size, self.hidden_size))
    def forward(self, features, captions):
        captions = captions[:,:-1]
        self.hidden = self.__init__hidden(features.shape[0])
        embeddings = self.embeddings(captions)
        embeddings = torch.cat((features.unsqueeze(1),embeddings), dim = 1)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        output = self.linear(lstm_out)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        hidden = self.__init__hidden(inputs.shape[0])
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(lstm_out)
            outputs = outputs.squeeze(1)
            _, max_index = torch.max(outputs, dim = 1)
            output.append(outputs.numpy()[0].items())
            if max_index == 1:
                break
            inputs = self.word_embeddings(max_indice) # inputs shape : (1, embed_size)
            inputs = inputs.unsqueeze(1)
        return output