import numpy as np
import cv2

im = cv2.imread('templates/sample.jpg',1)
im1 = im.copy()

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
	'gate':0.8,
}

for t in ['sprite','gate','ladder','key','rope']:

	res = cv2.matchTemplate(im,templates[t],cv2.TM_CCOEFF_NORMED)
	
	loc = np.where(res >= thresholds[t])
	print (loc)
	for pt in zip(*loc[::-1]):
		print (pt,end=',')
	h,w,_ = templates[t].shape[:]

	if len(loc[0]) > 1:

		mid = (len(loc[0])+1)//2
		pt = (loc[1][mid],loc[0][mid])
		cv2.rectangle(im1, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

	elif len(loc[0]) == 1:

		pt = (loc[1][0],loc[0][0])
		cv2.rectangle(im1, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

	print ('---',pt)

c1 = cv2.copyMakeBorder(im,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
c2 = cv2.copyMakeBorder(im1,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])

both = np.hstack((c1,c2))

cv2.imshow('Templates',both)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow('image',im)
# 
# cv2.waitKey(0)
# cv2.destroyAllWindows()