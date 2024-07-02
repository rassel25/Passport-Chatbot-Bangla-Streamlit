import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nlp_functions import tokenise, stemming, bag_of_words

# import our chat-bot intents file
import json
with open('intents.json','r', encoding="utf8") as json_data:
    intents = json.load(json_data)

print(intents)
# Initializes empty lists to store all words, tags, and pattern-tag pairs.
all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns. This loop processes each intent, collecting tags, tokenizing patterns, and creating pattern-tag pairs.
for intent in intents['intents']:
    # tag collection
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenise(pattern)
        # add to our words list. we dont use append bcz w and all_words both are array so we dont want array inside another array
        all_words.extend(w)
        # add pattern and tag as tuple in our corpus
        xy.append((w, tag))

print(all_words)

ignore_words = ['?']
all_words = [stemming(word) for word in all_words if word not in ignore_words]  # exclude ignore_words in stemming

print(all_words)

print(sorted(pd.unique(all_words)))

print(sorted(pd.unique(tags)))

print(len(xy), "patterns")

print(xy)

print(len(tags), "tags:", tags)

print(len(all_words), "unique stemmed words:", all_words)

x_train = []
y_train = []
for (pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) # we will be doing Crossentropyloss so no need to do one hot encoding

x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

# Hyper-parameters
num_epochs = 1000
batch_size = 16
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 64
output_size = len(tags)
print(input_size, output_size)
print(input_size, len(all_words))

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)).to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "chatbot.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
