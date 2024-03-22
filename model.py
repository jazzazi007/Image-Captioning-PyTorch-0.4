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
    def __init__(self, embed_size, hidden_size, vocab_size, batch_size=64, num_layers=2):
                

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.word_embedding_layer = nn.Embedding(vocab_size, embed_size) # embedding the words
        

        
        self.lstm = nn.LSTM( input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, 
                             dropout = 0.3, 
                             batch_first=True
                           ) #lstm rnn layer
        
        self.linear_fc = nn.Linear(hidden_size, vocab_size) #linear layer
        self.drop = nn.Dropout(p=0.3) # drop out layer
        self.init_weights() # initilizing the weights

    
    def forward(self, features, captions):

  
        captions = captions[:, :-1] #delet the last tokens 
        

        captions = self.word_embedding_layer(captions) #embedding
        

        inputs = torch.cat((features.unsqueeze(1), captions), dim=1) #merging the featues with the captions

        outputs, _ = self.lstm(inputs) #lstm
        
        #outputs = self.drop(outputs)
        #outputs = outputs.view(outputs.size()[0]*outputs.size()[1], self.hidden_size)

        outputs = self.linear_fc(outputs)#linear nn layer
        
        
        return outputs
    
    
    def init_weights(self):
       
        initrange = 0.1
        
        
        self.linear_fc.bias.data.fill_(0)
        
        self.linear_fc.weight.data.uniform_(-1, 1)
        
    def init_hidden(self, batch_size):
        
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_(),
                weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_())
    
    def sample(self, inputs, states=None, max_len=20):
 

        outputs = []   
        output_length = 0
        
        while (output_length != max_len+1):
        # Generate the next output and update the states using the LSTM.

            output, states = self.lstm(inputs,states)
           
        # Apply a linear fully connected layer to the output to map it to the vocabulary size.

            output = self.linear_fc(output.squeeze(dim = 1))
            _, predicted_index = torch.max(output, 1)
            
            outputs.append(predicted_index.cpu().numpy()[0].item())
            
        # Check if the predicted word is the end token (often index 1). If so, break the loop.

            if (predicted_index == 1):
                break

            inputs = self.word_embedding_layer(predicted_index)   
            inputs = inputs.unsqueeze(1)
            

            output_length += 1

        return outputs
   