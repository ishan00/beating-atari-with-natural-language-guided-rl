import gym
import template
import cv2
import numpy as np

env = gym.make('MontezumaRevenge-v0')

env.reset()

# print (env.unwrapped.ale.getMinimalActionSet())
# print (env.unwrapped.ale.lives())
# print (env.unwrapped.ale.getScreenDims())

x = env.unwrapped.ale.getScreenRGB2()

#c1 = cv2.copyMakeBorder(x,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
#c2 = cv2.copyMakeBorder(y,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])

#both = np.hstack((c1,c2))

#cv2.imshow('Templates',both)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# print (env.unwrapped.ale.getScreenGrayscale())
# env.unwrapped.ale.saveScreenPNG('a.png')

template.find_all_objects(x)

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
