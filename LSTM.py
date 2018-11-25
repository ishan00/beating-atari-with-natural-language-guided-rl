import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

torch.manual_seed(1)

word_to_ix = {}
label_to_ix = {}

instructions = []
labels = []

with open('instructions.txt','r') as f:

	f = f.readlines()
	
	for line in f:

		line = line.strip().split(',')
		label = [line[1]]
		sentence = list(map(lambda x : x.lower(),line[0].strip().split(' ')))
		instructions.append((sentence,label))

#print (instructions)

for sent,label in instructions:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)
	for lab in label:
		if lab not in label_to_ix:
			label_to_ix[lab] = len(label_to_ix)

print(word_to_ix)
print(label_to_ix)

def prepare_sentence(sent, to_ix):
	sent = sent.lower().strip().split(' ')
	idxs = [to_ix[w] for w in sent]
	return torch.tensor(idxs, dtype=torch.long)

class LSTMClassifier(nn.Module):

	def __init__(self):
		
		super(LSTMClassifier, self).__init__()

		self.embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
		self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM_LSTM)
		self.fullyconnected = nn.Linear(HIDDEN_DIM_LSTM, 10)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(1, 1, HIDDEN_DIM_LSTM)),
                autograd.Variable(torch.zeros(1, 1, HIDDEN_DIM_LSTM)))

	def forward(self, sentence):

		embeds = self.embeddings(sentence)
		x = embeds.view(len(sentence), 1, -1)
		lstm_out, self.hidden = self.lstm(x, self.hidden)
		#print (lstm_out)
		y  = self.fullyconnected(lstm_out[-1])
		# log_probs = F.log_softmax(y)
		#print (y)
		return y

class ConvNetClassifier(nn.Module):

	def __init__(self):
		
		super(ConvNetClassifier, self).__init__()

		self.layer1 = nn.Sequential(
				nn.Conv2d(6, 32, kernel_size = 5, stride = 1, padding = 2),
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

		self.layer5 = nn.Linear(26*20*64 , 10)

		self.layer6 = nn.PReLU()

		self.layer7 = nn.Linear(10, 10)

	def forward(self, x):

		x = np.swapaxes(x,0,2)
		x = np.swapaxes(x,1,2)

		x = autograd.Variable(torch.from_numpy(x).unsqueeze(0).float())

		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(out.size(0), -1)
		out = self.layer5(out)
		out = self.layer6(out)
		out = self.layer7(out)
		#print (out)
		
		return out

EMBEDDING_DIM = 20
HIDDEN_DIM_LSTM = 10
VOCAB_SIZE = len(word_to_ix)
LABEL_SIZE = len(label_to_ix)

text_model = LSTMClassifier()
image_model = ConvNetClassifier()
loss_function = nn.CosineEmbeddingLoss()
optimizer1 = optim.SGD(text_model.parameters(), lr = 0.001)
optimizer2 = optim.SGD(image_model.parameters(), lr = 0.001)

def train():

	with open('dataset/dataset_true.pickle','rb') as f:

		dataset = pickle.load(f)

	with open('dataset/dataset_false.pickle','rb') as g:
		dataset_false = pickle.load(g)

	for epoch in range(100):

		t1 = time.time()

		total_loss = 0.0

		for (frame1,frame2), sentence in dataset[:300]:

			text_model.hidden = text_model.init_hidden()
			
			text_model.zero_grad()
			image_model.zero_grad()

			enc_sentence = prepare_sentence(sentence, word_to_ix)

			text_embed = text_model(enc_sentence)

			stack = np.dstack((frame1,frame2))

			frame_embed = image_model(stack)

			loss = loss_function(text_embed, frame_embed,torch.tensor([1]).float())

			total_loss += loss.item()

			loss.backward()

			torch.nn.utils.clip_grad_norm(text_model.parameters(),1)
			torch.nn.utils.clip_grad_norm(image_model.parameters(),1)

			optimizer1.step()
			optimizer2.step()

			ind = np.random.randint(0,10000,5)

			for j in ind:

				text_model.hidden = text_model.init_hidden()
			
				text_model.zero_grad()
				image_model.zero_grad()

				enc_sentence = prepare_sentence(dataset_false[j][1], word_to_ix)

				text_embed = text_model(enc_sentence)

				stack = np.dstack(dataset_false[j][0])

				frame_embed = image_model(stack)

				loss = loss_function(text_embed, frame_embed,torch.tensor([-1]).float())

				total_loss += loss.item()

				loss.backward()

				torch.nn.utils.clip_grad_norm(text_model.parameters(),1)
				torch.nn.utils.clip_grad_norm(image_model.parameters(),1)

				optimizer1.step()
				optimizer2.step()

		t2 = time.time()

		print("epoch %d loss %f time %f"%(epoch,total_loss,t2-t1))

		if (epoch+1) % 20 == 0:
			torch.save(text_model, 'models/sentence/text_model_' + str(epoch+1))
			torch.save(image_model, 'models/image/image_model_' + str(epoch+1))

def false_dataset():

	#text_model = torch.load('models/text_model_50')
	#image_model = torch.load('models/image_model_50')

	with open('dataset/dataset_true.pickle','rb') as f:
		dataset = pickle.load(f)

	dataset_false = []

	for i in range(300):
		for j in range(300):

			if dataset[i][1] != dataset[j][1]:

				dataset_false.append((dataset[i][0],dataset[j][1]))
				dataset_false.append((dataset[j][0],dataset[i][1]))

	print (len(dataset_false))

	with open('dataset/dataset_false.pickle','wb') as f:
		pickle.dump(dataset_false,f)

	test_dataset_false = []

	for i in range(301, 347):
		for j in range(301, 347):

			if dataset[i][1] != dataset[j][1]:

				test_dataset_false.append((dataset[i][0],dataset[j][1]))
				test_dataset_false.append((dataset[j][0],dataset[i][1]))

	print (len(test_dataset_false))

	'''
	with open('dataset/dataset_false.pickle','wb') as f:
		pickle.dump(test_dataset_false,f)
	'''

def test():

	text_model = torch.load('models/text_model_60')
	image_model = torch.load('models/image_model_60')

	# True labels

	with open('dataset.pickle','rb') as f:
		true_dataset = pickle.load(f)

	items = np.random.randint(301, 347, 15)

	iter = 1
	for i in items:
		(img1, img2), text = true_dataset[i]
		#img1 = img1[:,:,::-1]
		#img2 = img2[:,:,::-1]
		'''
		enc_sentence = prepare_sentence(text, word_to_ix)
		text_embed = text_model(enc_sentence)
		stack = np.dstack((img1, img2))
		frame_embed = image_model(stack)

		dp = torch.dot(text_embed[0], frame_embed[0]) / (torch.norm(text_embed[0]) * torch.norm(frame_embed[0]))
		print(dp)
		'''
		#both = np.hstack((img1, img2))
		c1 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
		c2 = cv2.copyMakeBorder(img2,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])

		both = np.hstack((c1,c2))
		print (text)
		cv2.imshow('sample', both)
		cv2.waitKey(0)
	'''
	with open('test_dataset_false.pickle','rb') as f:
		false_dataset = pickle.load(f)

	items = np.random.randint(0, 3384, 15)

	iter = 1
	for i in items:
		(img1, img2), text = false_dataset[i]
		# img1 = img1[:,:,::-1]
		# img2 = img2[:,:,::-1]
		enc_sentence = prepare_sentence(text, word_to_ix)
		text_embed = text_model(enc_sentence)
		stack = np.dstack((img1, img2))
		frame_embed = image_model(stack)

		dp = torch.dot(text_embed[0], frame_embed[0]) / (torch.norm(text_embed[0]) * torch.norm(frame_embed[0]))
		print(dp)

		# both = np.hstack((img1, img2))
		c1 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
		c2 = cv2.copyMakeBorder(img2,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])

		both = np.hstack((c1,c2))
		cv2.imwrite('images/'+text+str(iter)+'_false' + str(dp.data)+ '.jpg', both)

		iter += 1

	'''


	# False labels

# train()

# false_dataset()

test()
