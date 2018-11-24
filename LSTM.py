import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

word_to_ix = {}

instructions = []

with open('instructions.txt','r') as f:

	f = f.readlines()
	
	for line in f:

		line = list(map(lambda x : x.lower(),line.strip().split(' ')))
		instructions.append(line)

#print (instructions)

for sent in instructions:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print(word_to_ix)

def prepare_sentence(sent):
	idxs = [word_to_ix[w] for w in sent]
	return torch.tensor(idxs, dtype=torch.long)

EMBEDDING_DIM = 10
HIDDEN_DIM = 5
LABEL_SIZE = 7

layer = nn.Embedding(len(word_to_ix), EMBEDDING_DIM)
lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)
hidden2label = nn.Linear(HIDDEN_DIM, label_size)


hidden = (torch.zeros(1, 1, 5), torch.zeros(1, 1, 5))

for sent in instructions[:1]:

	enc = prepare_sentence(sent)

	#print (enc,layer(enc))

	embeds = layer(enc)

	lstm_out, hidden = lstm(embeds.view(len(sent), 1, -1), hidden)

	print (enc) 
	print (embeds)
	print (lstm_out)









