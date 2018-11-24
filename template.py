import numpy as np
import cv2
import pickle
from numpy import unravel_index

templates = {
	'ladder':cv2.imread('templates/ladder.jpg',1),
	'key':cv2.imread('templates/key.jpg',1),
	'rope':cv2.imread('templates/rope.jpg',1),
	'sprite':cv2.imread('templates/sprite.jpg',1),
	'sprite_side':cv2.imread('templates/sprite.jpg',0),
	'sprite_back':cv2.imread('templates/sprite_back.jpg',0),
	'gate':cv2.imread('templates/gate.jpg',1),
}

thresholds = {
	'ladder':0.9,
	'key':0.9,
	'rope':0.9,
	'sprite':0.9,
	'gate':0.85,
}

def find_all_objects():

	image = cv2.imread('templates/room1.jpg',1)

	boxed_image = image.copy()

	list_of_locations = {
		'sprite':None,
		'gate':None,
		'ladder':None,
		'key':None,
		'rope':None,
	}

	for obj in ['sprite','gate','ladder','key','rope']:

		res = cv2.matchTemplate(image,templates[obj],cv2.TM_CCOEFF_NORMED)
		loc = np.where(res >= thresholds[obj])
		h,w,_ = templates[obj].shape[:]

		if len(loc[0]) == 1:

			pt = (loc[0][0],loc[1][0])
			cv2.rectangle(boxed_image, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

			list_of_locations[obj] = (pt[0] + w//2, pt[1] + h//2)

		elif len(loc[0]) > 1:

			selected = []

			for frame in zip(*loc[::-1]):

				found_nearby = False
				
				for sel in selected:
					if abs(sel[0] - frame[0]) + abs(sel[1] - frame[1]) <= 20:
						found_nearby = True
						break

				if found_nearby:
					continue

				selected.append(frame)

				cv2.rectangle(boxed_image, frame, (frame[0] + w, frame[1] + h), (0,255,255), 2)

			if len(selected) == 1:
				list_of_locations[obj] = (selected[0][0] + w//2, selected[0][1] + h//2)
			else:
				list_of_locations[obj] = [(sel[0] + w//2,sel[1] + h//2) for sel in selected]

	print (list_of_locations)
	'''
	c1 = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
	c2 = cv2.copyMakeBorder(boxed_image,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])

	both = np.hstack((c1,c2))

	cv2.imshow('Templates',both)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
	return list_of_locations
	
def show_image(image):

	image = image[:,:,::-1]

	cv2.imshow('Debug',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def find_sprite(image):

	image = image[:,:,::-1]

	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	res1 = cv2.matchTemplate(img,templates['sprite_side'],cv2.TM_CCOEFF_NORMED)
	res2 = cv2.matchTemplate(img,templates['sprite_back'],cv2.TM_CCOEFF_NORMED)

	#cv2.imshow('res1',res1)
	#cv2.waitKey()

	#cv2.imshow('res2',res2)
	#cv2.waitKey()

	h1,w1 = templates['sprite_side'].shape[:]
	h2,w2 = templates['sprite_back'].shape[:]

	if np.amax(res1) > np.amax(res2):
		pt = unravel_index(res1.argmax(), res1.shape)
		return (pt[0] - h1//2, pt[1] - w1//2, pt[0] + h1//2, pt[1] + w1//2)
	else:		
		pt = unravel_index(res2.argmax(), res2.shape)
		return (pt[0] - h2//2, pt[1] - w2//2, pt[0] + h2//2, pt[1] + w2//2)


def debug():
	with open('sprite_7.pickle','rb') as f:
		image = pickle.load(f)

	image = image[:,:,::-1]

	cv2.imshow('Initial',image)
	cv2.waitKey(0)

	res1 = cv2.matchTemplate(image,templates['sprite_side'],cv2.TM_CCOEFF_NORMED)
	res2 = cv2.matchTemplate(image,templates['sprite_back'],cv2.TM_CCOEFF_NORMED)

	cv2.imshow('res1',res1)
	cv2.waitKey()

	cv2.imshow('res2',res2)
	cv2.waitKey()

	h1,w1,_ = templates['sprite_side'].shape[:]
	h2,w2,_ = templates['sprite_back'].shape[:]

	if np.amax(res1) > np.amax(res2):

		print (unravel_index(res1.argmax(), res1.shape))

	else:
		
		print (unravel_index(res2.argmax(), res2.shape))

	#pt = (loc[0][0],loc[1][0],h,w)
	#print (loc)
	#print ('R : %d, C : %d'%(pt[1],pt[0]))

	#cv2.rectangle(image, (pt[0],pt[1]), (pt[0] + w, pt[1] + h), (0,255,255), 2)
	#cv2.imshow('Templates',image)
	#cv2.waitKey(0)

#debug()