import numpy as np
import template
'''
1. going down the 1st ladder
2. jumping on the rope
3. going down the 2nd ladder
4. going up the 3rd ladder
5. fetching the key
6. going down the 3rd ladder
7. going up the 2nd ladder
8. jumping on the rope
9. going up the 1st ladder
10. go to the right room
'''

def objects(dict):
	return	{
		'ladder1':dict['ladder'][0],
		'ladder2':dict['ladder'][1],
		'ladder3':dict['ladder'][2],
		'gate1':dict['gate'][0],
		'gate2':dict['gate'][1],
		'key':dict['key'],
		'rope':dict['rope'],
	}


class RewardHistory():

	def __init__(self):

		self.room = 1
		self.n_rewards = 10
		self.rewards = np.ones(self.n_rewards, dtype = float)

		self.object_locations = [(0, 0) for _ in range(self.n_rewards)]
		
		object_locations = template.find_all_objects()
		object_locations_named = objects(object_locations)

		self.object_locations[0] = object_locations_named['ladder1']
		self.object_locations[1] = object_locations_named['rope']
		self.object_locations[2] = object_locations_named['ladder2']
		self.object_locations[3] = object_locations_named['ladder3']
		self.object_locations[4] = object_locations_named['key']
		self.object_locations[5] = object_locations_named['ladder3']
		self.object_locations[6] = object_locations_named['ladder2']
		self.object_locations[7] = object_locations_named['rope']
		self.object_locations[8] = object_locations_named['ladder1']
		self.object_locations[9] = object_locations_named['gate2']

		print (self.object_locations)

	def reward(self, frameRGB, info_curr, info_next):

		# frame_number to be used later

		sprite_pos = template.find_sprite(frameRGB)

		print (sprite_pos)

		reward = 0

		if info_next['ale.lives'] < info_curr['ale.lives']:
			reward -= 1

		for i in range(self.n_rewards):

			if self.rewards[i] == 0:
				continue

			center = self.object_locations[i]

			if sprite_pos[0] <= center[0] and sprite_pos[0] + sprite_pos[2] >= center[0] and sprite_pos[1] <= center[1] and sprite_pos[1] + sprite_pos[3] >= center[1]:

				reward = self.rewards[i]
				self.rewards[i] = 0
				break

		return reward
		






















