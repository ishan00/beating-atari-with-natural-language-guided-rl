import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from readchar import readchar
from reward import RewardHistory
import pickle
import cv2
import time

'''
Moves
0 - NOP
1 - JUMP
2 - UP
3 - RIGHT
4 - LEFT
5 - DOWN
6	UPRIGHT
7	UPLEFT
8	DOWNRIGHT
9	DOWNLEFT
10	UPJUMP
11	RIGHTJUMP
12	LEFTJUMP
13	DOWNJUMP
14	UPRIGHTJUMP
15	UPLEFTJUMP
16	DOWNRIGHTJUMP
17	DOWNLEFTJUMP


'''


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
	sent = sent.lower().strip().split(' ')
	idxs = [to_ix[w] for w in sent]
	return torch.tensor(idxs, dtype=torch.long)

class LSTMClassifier(nn.Module):

	def __init__(self):
		
		super(LSTMClassifier, self).__init__()

		self.embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
		self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM_LSTM)
		self.fullyconnected = nn.Linear(HIDDEN_DIM_LSTM, 10)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(1, 1, HIDDEN_DIM_LSTM)),
                autograd.Variable(torch.zeros(1, 1, HIDDEN_DIM_LSTM)))

	def forward(self, sentence):

		embeds = self.embeddings(sentence)
		x = embeds.view(len(sentence), 1, -1)
		lstm_out, self.hidden = self.lstm(x, self.hidden)
		#print (lstm_out)
		y  = self.fullyconnected(lstm_out[-1])
		# log_probs = F.log_softmax(y)
		#print (y)
		return y

class ConvNetClassifier(nn.Module):

	def __init__(self):
		
		super(ConvNetClassifier, self).__init__()

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

		self.layer5 = nn.Linear(26*20*64 , 10)

		self.layer6 = nn.PReLU()

		self.layer7 = nn.Linear(10, 10)

	def forward(self, x):

		x = np.swapaxes(x,0,2)
		x = np.swapaxes(x,1,2)

		x = autograd.Variable(torch.from_numpy(x).unsqueeze(0).float())

		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(out.size(0), -1)
		out = self.layer5(out)
		out = self.layer6(out)
		out = self.layer7(out)
		#print (out)
		
		return out

EMBEDDING_DIM = 20
HIDDEN_DIM_LSTM = 10
VOCAB_SIZE = len(word_to_ix)
LABEL_SIZE = len(label_to_ix)

text_model = LSTMClassifier()
image_model = ConvNetClassifier()
loss_function = nn.MSELoss()
optimizer1 = optim.SGD(text_model.parameters(), lr = 0.001)
optimizer2 = optim.SGD(image_model.parameters(), lr = 0.001)

def char_to_action():

	x = readchar()

	list_of_char = ['f','g','h','t',' ','r','y','q']
	list_of_int = [4,5,3,2,1,12,11,-1]

	for i in range(len(list_of_char)):
		if list_of_char[i] == x:
			return list_of_int[i]

	return 0


def main():
	
	env = gym.make('MontezumaRevenge-v0')

	rewards = RewardHistory()

	prev_state = env.reset()

	prev_info = {'ale.lives':6}

	env = gym.make('MontezumaRevenge-v0')

	text_model = torch.load('models/sentence/text_model_20')
	image_model = torch.load('models/image/image_model_20')

	with open('dataset/dataset_true.pickle','rb') as f:
		dataset = pickle.load(f)

	with open('first_room.txt','r') as f:
		instructions = f.readlines()

	instructions = [i.strip() for i in instructions]

	print (instructions)

	state = env.reset()

	info = {'ale.lives':6}

	for t in range(100):
		
		action = char_to_action()

		if action == -1:
			env.close()
			break

		state_new, reward, done, info_new = env.step(action)
		
		stack = np.dstack((state,state_new))

		frame_embed = image_model(stack)

		enc_sentence = prepare_sentence(instructions[0], word_to_ix)

		text_embed = text_model(enc_sentence)

		reward = torch.dot(text_embed[0], frame_embed[0]) / (torch.norm(text_embed) * torch.norm(frame_embed))
		
		r = reward.item()

		print (r)

		env.render()

		info = info_new
		state = state_new

	print('Episode %d\t Episode Reward: %f\t'%(episode))

	env.close()

