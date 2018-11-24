import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

torch.manual_seed(1)

word_to_ix = {}
label_to_ix = {}

instructions = []
labels = []

with open('instructions.txt','r') as f:

	f = f.readlines()
	
	for line in f:

		line = line.strip().split(',')
		label = [line[1]]
		sentence = list(map(lambda x : x.lower(),line[0].strip().split(' ')))
		instructions.append((sentence,label))

#print (instructions)

for sent,label in instructions:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)
	for lab in label:
		if lab not in label_to_ix:
			label_to_ix[lab] = len(label_to_ix)

print(word_to_ix)
print(label_to_ix)

def prepare_sentence(sent, to_ix):
	idxs = [to_ix[w] for w in sent]
	return torch.tensor(idxs, dtype=torch.long)

EMBEDDING_DIM = 10
HIDDEN_DIM = 5
VOCAB_SIZE = len(word_to_ix)
LABEL_SIZE = len(label_to_ix)

class LSTMClassifier(nn.Module):

	def __init__(self):
		
		super(LSTMClassifier, self).__init__()

		self.embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
		self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)
		self.fullyconnected = nn.Linear(HIDDEN_DIM, LABEL_SIZE)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(1, 1, HIDDEN_DIM)),
                autograd.Variable(torch.zeros(1, 1, HIDDEN_DIM)))

	def forward(self, sentence):

		embeds = self.embeddings(sentence)
		x = embeds.view(len(sentence), 1, -1)
		lstm_out, self.hidden = self.lstm(x, self.hidden)
		y  = self.fullyconnected(lstm_out[-1])
		log_probs = F.log_softmax(y)
		return log_probs

model = LSTMClassifier()
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

def train():

	for epoch in range(300):

		total_loss = 0.0

		for sentence,label in instructions:

			model.hidden = model.init_hidden()
			model.zero_grad()

			enc_sentence = prepare_sentence(sentence, word_to_ix)
			enc_label = prepare_sentence(label, label_to_ix)
			
			tag_scores = model(enc_sentence)

			loss = loss_function(tag_scores, enc_label)

			total_loss += loss.item()

			loss.backward()
			optimizer.step()

		print("epoch %d loss %f"%(epoch,total_loss))

train()