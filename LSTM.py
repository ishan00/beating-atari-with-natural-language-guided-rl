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
		#log_probs = F.log_softmax(y)
		return y

class ConvNetClassifier(nn.Module):

	def __init__(self):
		
		super(ConvNet, self).__init__()

		self.layer1 = nn.Sequential(
				nn.Conv2d(6, 32, kernel_size = 5, stride = 1, padding = 2),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = 2, stride = 2)
			)

		self.layer2 = nn.Sequential(
				nn.Conv2d(32, 32, kernel_size = 5, stride = 1, padding = 2),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = 2, stride = 2)
			)

		self.layer3 = nn.Sequential(
				nn.Conv2d(32, 64, kernel_size = 4, stride = 1, padding = 2),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = 2, stride = 2)
			)

		self.layer4 = nn.Sequential(
				nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
			)

		self.layer5 = nn.Linear(2*26*20*64 , 10)

		self.layer6 = nn.PReLU()

		self.layer7 = nn.Linear(10, 10)

	def forward(self, x):

		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(out.size(0), -1)
		out = self.layer5(out)
		out = self.layer6(out)
		out = self.layer7(out)
		
		return out

'''
class CustomLoss():

	def forward(self, output_of_lstm, output_of_conv_net):

		print (output_of_lstm)
		print (output_of_conv_net)

		loss = 1.0 - torch.dot(output_of_lstm, output_of_conv_net) / max(torch.norm(output_of_lstm) * torch.norm(output_of_conv_net))

	def backward():
'''

EMBEDDING_DIM = 20
HIDDEN_DIM = 10
VOCAB_SIZE = len(word_to_ix)
LABEL_SIZE = len(label_to_ix)


text_model = LSTMClassifier()
image_model = ConvNetClassifier()
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

def train():

	for epoch in range(300):

		total_loss = 0.0

		for sentence,frame in instructions:

			text_model.hidden = model.init_hidden()
			
			text_model.zero_grad()
			image_model.zero_grad()

			enc_sentence = prepare_sentence(sentence, word_to_ix)
			#enc_label = prepare_sentence(label, label_to_ix)
			
			text_embed = text_model(enc_sentence)

			frame_embed = image_model(frame)

			loss = loss_function(text_embed, frame_embed)

			total_loss += loss.item()

			loss.backward()
			optimizer.step()

		print("epoch %d loss %f"%(epoch,total_loss))

train()