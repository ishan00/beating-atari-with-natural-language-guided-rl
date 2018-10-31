#https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np
import time

epi = 1000
action_space = 6
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
eps = np.finfo(np.float32).eps.item()

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

	t1 = time.time()

	l = len(model.rewards) # Length of Episode

	model_rewards = np.array(model.rewards)
	_rewards = disc[:l] * model_rewards
	cumulative_rewards = np.cumsum(_rewards[::-1])[::-1]
	rewards = cumulative_rewards * inv_disc[:l]

	rewards = torch.tensor(rewards).float()
	rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

	t2 = time.time()

	policy_loss = []

	for log_prob, reward in zip(model.saved_log_probs, rewards):
		policy_loss.append(- log_prob * reward)
	
	t3 = time.time()

	optimizer.zero_grad()
	policy_loss = torch.cat(policy_loss).sum()
	policy_loss.backward()
	optimizer.step()

	del model.rewards[:]
	del model.saved_log_probs[:]

	t4 = time.time()

	print("Time %d\t%d\t%d"%(t2 - t1, t3 - t2, t4 - t3))


def main():
	
	env = gym.make('Riverraid-v0')

	for episode in range(1000):

		state = env.reset()
		episode_reward = 0

		for t in range(epi):
			if (t % 100 == 0): 
				print(t)
			action = select_action(state)
			state, reward, done, _ = env.step(action)
			episode_reward += reward
			model.rewards.append(reward)

			#env.render()
		
			if done:
				break

		print ("----")

		finish_episode()

		print('Episode %d\t Episode Reward: %f'%(episode, episode_reward))

	env.close()

main()

























