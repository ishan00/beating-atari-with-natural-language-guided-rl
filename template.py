import numpy as np
import cv2

templates = {
	'ladder':cv2.imread('templates/ladder.jpg',1),
	'key':cv2.imread('templates/key.jpg',1),
	'rope':cv2.imread('templates/rope.jpg',1),
	'sprite':cv2.imread('templates/sprite.jpg',1),
	'gate':cv2.imread('templates/gate.jpg',1),
}

thresholds = {
	'ladder':0.9,
	'key':0.9,
	'rope':0.9,
	'sprite':0.8,
	'gate':0.85,
}

def find_locations(image):

	image = image[:,:,::-1]

	cv2.imwrite( "templates/sample.jpg", image);

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

			pt = (loc[1][0],loc[0][0])
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

	c1 = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
	c2 = cv2.copyMakeBorder(boxed_image,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])

	both = np.hstack((c1,c2))

	cv2.imshow('Templates',both)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return list_of_locations
	

def show_image(image):

	image = image[:,:,::-1]

	cv2.imshow('Debug',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
