import random
import json
import torch
from train import model
from nlp_functions import tokenise, bag_of_words

with open('intents.json', 'r', encoding="utf-8") as json_data:
    intents = json.load(json_data)

FILE = "chatbot.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(model_state)
model.eval()

def get_response(msg):
   sentence = tokenise(msg)
   X = bag_of_words(sentence, all_words)   # [many rows x 1 columns]
   X = X.reshape(1, X.shape[0])  # [1 rows x many columns]
   X = torch.from_numpy(X).to(device)

   output = model(X)
   _, predicted = torch.max(output, dim=1)   # _ is used to omit 1st part of output
   tag = tags[predicted.item()]   # predicted = tensor([2]). predicted.item() = 2

   probs = torch.softmax(output, dim=1)
   prob = probs[0][predicted.item()]
   if prob.item() > 0.80:
      for intent in intents['intents']:
          if tag == intent["tag"]:
               return random.choice(intent['responses'])

   return "দুঃখিত, দয়া করে আবার চেষ্টা করুন"

