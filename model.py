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

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_size)

        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=vocab_size)
    
    def forward(self, features, captions):

        captions = captions[:, :-1]
        captions_embed = self.embedding(captions)
        captions_embed = torch.cat((features.unsqueeze(1), captions_embed), dim=1)

        lstm_out, _ = self.lstm(captions_embed)
        scores = self.linear(lstm_out)

        return scores

    def sample(self, inputs, states=None, max_len=20):
        """ Accepts pre-processed image tensor (inputs) and returns
            predicted sentence (list of tensor ids of length max_len)
        """
        pass