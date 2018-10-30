import numpy as np
import readchar
import gym

class RewardHistory():

	def __init__(self):

		self.n_rewards = 21
		self.rewards = np.ones(self.n_rewards, dtype = float) # Each of these rewards correspond to one natural language instruction

		'''

		Index	NLP Instruction

		0		Climb down the ladder
		1		Jump to the rope
		2		Go to the right side of the room
		3		Climb down the ladder
		4		Go to the bottom of the room
		5		Go to the center of the room
		6		Go to the left side of the room
		7		Climb up the ladder
		8		Get the key
		9		Climb down the ladder
		10		Go to the bottom of the room
		11		Go to the center of the room
		12		Go to the right side of the room
		13		Climb up the ladder
		14		Jump to the rope
		15		Go to the center of the room
		16		Climb up the ladder
		17		Go to the top of the room
		18		Go to the right side of the room
		19		Use the key
		20		Go to the right room
		
		'''

	def reward(self):
















env = gym.make('MontezumaRevenge-v0')

env.reset()

# print (env.unwrapped.ale.getMinimalActionSet())
# print (env.unwrapped.ale.lives())
# print (env.unwrapped.ale.getScreenDims())
# print (env.unwrapped.ale.getScreenRGB())
# print (env.unwrapped.ale.getScreenGrayscale())
# print (env.unwrapped.ale.saveScreenPNG('sample.jpg'))

'''
for _ in range(10):
	
	env.reset()

	for _ in range(1000):

		s,r,done,_ = env.step(env.action_space.sample())

		if done == True:
			break

		env.render()

'''
'''
quit = False

last_action = 0

for games in range(10):
	
	if quit:
		break

	env.reset()

	for _ in range(1000):

		char1 = readchar.readchar()
		char2 = readchar.readchar()

		action = last_action

		if char1 == 'q':
			quit = True
			break
		elif char1 == 'w' and char2 == 'w':
			action = 2
		elif char1 == 'a' and char2 == 'a':
			action = 4
		elif char1 == 's' and char2 == 's':
			action = 5
		elif char1 == 'd':
			action = 3
		elif char1 == ' ':
			action = 1
		elif (char1 == 'w' and char2 == 'd') or (char1 == 'd' and char2 == 'w'):
			action = 11
		elif (char1 == 'w' and char2 == 'a') or (char1 == 'a' and char2 == 'w'):
			action = 12
		

		state,reward,done,info = env.step(action)

		env.render()

		last_action = action

		print (info)
'''
































