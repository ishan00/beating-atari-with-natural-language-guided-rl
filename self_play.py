import gym
import numpy as np
from readchar import readchar
from reward import RewardHistory
import pickle
import cv2

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

	for i in range(1000):

		action = char_to_action()

		if action == -1:
			env.close()
			break

		state, reward, done, info = env.step(action)

		with open('saved/sprite_' + str(i) + '.pickle','wb') as f:
			pickle.dump(state,f)

		current_reward = rewards.reward(state, info, prev_info)

		print ('Reward : ',current_reward)

		env.render()

		prev_state = state
		prev_info = info

		if done:
			break

	env.close()




#main()