#https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from torch.autograd import Variable
import gym
import numpy as np
import time
import reward

torch.set_default_tensor_type('torch.FloatTensor')

epi = 1000
action_space = 2
gamma = 0.95

saved_log_probs = []
rewards = []

class Basic(nn.Module):

	def __init__(self, gamma = 0.95):

		super(Basic, self).__init__()

		self.layer1 = nn.Linear(4,128)
		self.layer2	= nn.Linear(128,1)

		self.gamma = gamma

	def forward(self, x):

		x = Variable(torch.from_numpy(x).float())

		out = F.relu(self.layer1(x))
		out = F.sigmoid(self.layer2(out))
		return out # Prob of Left 0
	
model = Basic(gamma)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
eps = np.finfo(np.float64).eps.item()

def select_action(state):

	probs = model.forward(state)

	m = Bernoulli(probs)
	action = m.sample()

	saved_log_probs.append(m.log_prob(action))
	return action.item()

disc = np.zeros(epi)
inv_disc = np.zeros(epi)
disc[0] = 1.0
inv_disc[0] = 1.0
for i in range(1,epi):
	disc[i] = gamma * disc[i-1]
	inv_disc[i] = inv_disc[i-1] / gamma

def finish_episode():

	global rewards, saved_log_probs

	l = len(rewards) # Length of Episode

	if l == 0:
		return

	model_rewards = np.array(rewards)
	_rewards = disc[:l] * model_rewards
	cumulative_rewards = np.cumsum(_rewards[::-1])[::-1]
	rewards = cumulative_rewards * inv_disc[:l]

	rewards = torch.tensor(rewards).float()
	rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

	policy_loss = []

	for log_prob, reward in zip(saved_log_probs, rewards):
		policy_loss.append(- log_prob * reward)

	optimizer.zero_grad()
	policy_loss = torch.cat(policy_loss).sum()
	policy_loss.backward()
	optimizer.step()

	rewards = []
	saved_log_probs = []

#RewardState = reward.RewardHistory()

def main():
	
	global rewards, saved_log_probs

	env = gym.make('CartPole-v0')

	episode_length = []

	for episode in range(10000):

		state = env.reset()
		episode_reward = 0

		rewards = []
		saved_log_probs = []

		#info_prev = {'ale.lives':6}

		#if episode % 10 == 0 and episode != 0:

		#	torch.save(model, 'saved/model_' + str(episode))
		#	torch.save(model.state_dict(), 'saved/parameter_' + str(episode))

		for t in range(epi):
			
			action = select_action(state)

			if action < 0.5:
				action = 1
			else:
				action = 0

			state, reward, done, info = env.step(action)
			
			#reward = RewardState.reward(state, info, info_prev)
			
			if done:
				episode_length.append(t + 1)
				reward = 0

			episode_reward += reward

			rewards.append(reward)

			#info_prev = info

			if done:
				break

			#if episode % 20 == 0:
			#	env.render()

			if t % 10 == 0 and t != 0:
				finish_episode()

		if episode % 10 == 0:
			print('Episode %d\t Episode length: %d\t Episode Reward: %f'%(episode, t+1, episode_reward))

		if episode_reward > 195:
			print ('Solved')
			break

	env.close()

main()


























