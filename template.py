import numpy as np
import cv2

templates = {
	'ladder':cv2.imread('templates/ladder.png',1),
	'key':cv2.imread('templates/key.png',1),
	'rope':cv2.imread('templates/rope.png',1),
	'sprite':cv2.imread('templates/sprite.png',1),
	'gate':cv2.imread('templates/gate.png',1),
}

thresholds = {
	'ladder':0.7,
	'key':0.9,
	'rope':0.9,
	'sprite':0.8,
	'gate':0.85,
}

def find_locations(image):

	im1 = im.copy()

	list_of_locations = {
		'sprite':None,
		'gate':None,
		'ladder':None,
		'key':None,
		'rope':None,
	}

	for obj in ['sprite','gate','ladder','key','rope']:

		res = cv2.matchTemplate(im,templates[obj],cv2.TM_CCOEFF_NORMED)
		loc = np.where(res >= thresholds[obj])
		h,w,_ = templates[obj].shape[:]

		if len(loc[0]) == 1:

			pt = (loc[1][0],loc[0][0])
			cv2.rectangle(im1, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

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

				cv2.rectangle(im1, frame, (frame[0] + w, frame[1] + h), (0,255,255), 2)

			if len(selected) == 1:
				list_of_locations[obj] = (selected[0][0] + w//2, selected[0][1] + h//2)
			else:
				list_of_locations[obj] = [(sel[0] + w//2,sel[1] + h//2) for sel in selected]

	#c1 = cv2.copyMakeBorder(im,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
	#c2 = cv2.copyMakeBorder(im1,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])

	#both = np.hstack((c1,c2))

	#cv2.imshow('Templates',im1)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	print (list_of_locations)

	return list_of_locations

im = cv2.imread('templates/sample.jpg',1)
find_locations(im)