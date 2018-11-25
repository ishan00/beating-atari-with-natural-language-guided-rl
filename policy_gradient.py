#https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import gym
import numpy as np
import time
import reward
import pickle

epi = 1000
action_space = 18
gamma = 0.95

class ConvNet(nn.Module):

	def __init__(self, output_size = 18, gamma = 0.95):

		super(ConvNet, self).__init__()

		self.layer1 = nn.Sequential(
				nn.Conv2d(3, 32, kernel_size = 5, stride = 1, padding = 2),
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

		self.layer5 = nn.Linear(26*20*64 , output_size)

		self.layer6 = nn.PReLU()

		self.layer7 = nn.Softmax(dim = 0) # 1-10 Policy Prediction 11 - Value

		self.saved_log_probs = []
		self.rewards = []
		self.gamma = gamma


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

model = ConvNet(action_space, gamma)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
eps = np.finfo(np.float64).eps.item()

def select_action(state):

	state = np.array([state[:,:,0],state[:,:,1],state[:,:,2]])
	state = torch.from_numpy(state).float().unsqueeze(0)

	probs = model.forward(state)
	m = Categorical(probs)
	action = m.sample()
	model.saved_log_probs.append(m.log_prob(action))
	return action.item()

disc = np.zeros(epi)
inv_disc = np.zeros(epi)
disc[0] = 1.0
inv_disc[0] = 1.0
for i in range(1,epi):
	disc[i] = gamma * disc[i-1]
	inv_disc[i] = inv_disc[i-1] / gamma

def finish_episode():

	l = len(model.rewards) # Length of Episode

	if l == 0:
		return

	model_rewards = np.array(model.rewards)
	_rewards = disc[:l] * model_rewards
	cumulative_rewards = np.cumsum(_rewards[::-1])[::-1]
	rewards = cumulative_rewards * inv_disc[:l]

	print (rewards)

	rewards = torch.from_numpy(rewards).float()
	rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

	policy_loss = []

	for log_prob, reward in zip(model.saved_log_probs, rewards):
		policy_loss.append(- log_prob * reward)

	optimizer.zero_grad()
	policy_loss = torch.cat(policy_loss).sum()
	policy_loss.backward()
	optimizer.step()

	del model.rewards[:]
	del model.saved_log_probs[:]



#RewardState = reward.RewardHistory()

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

def main():
	
	env = gym.make('MontezumaRevenge-v0')

	text_model = torch.load('models/sentence/text_model_40')
	image_model = torch.load('models/image/image_model_40')
	model = torch.load('models/policy/model_100')

	with open('dataset/dataset_true.pickle','rb') as f:
		dataset = pickle.load(f)

	with open('first_room.txt','r') as f:
		instructions = f.readlines()

	instructions = [i.strip() for i in instructions]

	print (instructions)

	for episode in range(1000):

		t1 = time.time()

		state = env.reset()
		episode_reward = 0

		info = {'ale.lives':6}

		#if episode % 10 == 0 and episode != 0:

		#	torch.save(model, 'saved/model_' + str(episode))
		#	torch.save(model.state_dict(), 'saved/parameter_' + str(episode))

		for t in range(epi):
			
			action = select_action(state)

			state_new, reward, done, info_new = env.step(action)
			
			stack = np.dstack((state,state_new))

			frame_embed = image_model(stack)

			enc_sentence = prepare_sentence(instructions[0], word_to_ix)

			text_embed = text_model(enc_sentence)

			reward = torch.dot(text_embed[0], frame_embed[0])
			
			r = reward.item()

			#print (r)

			if r < 0.9: r = 0

			model.rewards.append(r)

			env.render()

			info = info_new
			state = state_new
			#if episode % 20 == 0:
			#	env.render()

			#if (t+1) % 40 == 0:
			#	finish_episode()
		
			if done:
				break

		t2 = time.time()

		print('Episode %d\t Episode Reward: %f\t Time: %f'%(episode, episode_reward, t2 - t1))

	env.close()

main()
