import torch as ch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
from tensorflow import keras
import os

import utils


def train_as_they_said(model, trainloader, testloader, loss_fn, acc_fn, epochs=40):
	optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)
	# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

	for _ in range(epochs):
		# Train
		running_loss, running_acc = 0.0, 0.0
		num_samples = 0
		model.train()
		iterator = tqdm(trainloader)
		for (x, y) in iterator:
			x, y = x.cuda(), y.cuda()
			# Smile prediction
			y = y[:, 31:32]

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(x)
			loss = loss_fn(outputs, y.float())
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			running_acc  += acc_fn(outputs, y)
			num_samples += x.shape[0]

			iterator.set_description("[Train] Loss: %.5f Accuacy: %.2f" % (running_loss / num_samples, 100 * running_acc / num_samples))

		# Validation
		model.eval()
		running_loss, running_acc = 0.0, 0.0
		num_samples = 0
		for (x, y) in testloader:
			x, y = x.cuda(), y.cuda()
			# Smile prediction
			y = y[:, 31:32]

			outputs = model(x)
			loss = loss_fn(outputs, y.float())
			running_loss += loss.item()
			running_acc  += acc_fn(outputs, y)
			num_samples += x.shape[0]

		print("[Val] Loss: %.5f Accuacy: %.2f" % (running_loss / num_samples, 100 * running_acc / num_samples))


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='census', help='which dataset to work on (census/mnist/celeba)')
	args = parser.parse_args()
	utils.flash_utils(args)


	if args.dataset == 'census':
		# Census Income dataset
		ci = utils.CensusIncome("./census_data/")

		sex_filter    = lambda df: utils.filter(df, lambda x: x['sex:Female'] == 1, 0.65)
		race_filter   = lambda df: utils.filter(df, lambda x: x['race:White'] == 0,  1.0)
		income_filter = lambda df: utils.filter(df, lambda x: x['income'] == 1, 0.5)

		num_cfs = 10
		for i in range(1, num_cfs + 1):
			(x_tr, y_tr), (x_te, y_te), _ = ci.load_data(race_filter)
			# clf = RandomForestClassifier(max_depth=30, random_state=0, n_jobs=-1)
			clf = MLPClassifier(hidden_layer_sizes=(60, 30, 30), max_iter=200)
			clf.fit(x_tr, y_tr.ravel())
			print("Classifier %d : Train acc %.2f , Test acc %.2f" % (i,
				100 * clf.score(x_tr, y_tr.ravel()),
				100 * clf.score(x_te, y_te.ravel())))

			dump(clf, os.path.join('census_models_mlp/race/', str(i)))

	elif args.dataset == 'celeba':
		# CelebA dataset
		model = utils.FaceModel(512).cuda()
		model = nn.DataParallel(model)
		ds = utils.Celeb().get_dataset()
		trainloader, testloader = ds.make_loaders(batch_size=2048, workers=8)

		loss_fn = nn.BCELoss()
		acc_fn = lambda outputs, y: ch.sum((y == (outputs >= 0.5)))
		train_as_they_said(model, trainloader, testloader, loss_fn, acc_fn)

	elif args.dataset == 'mnist':
		# MNIST
		(x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
		x_tr = x_tr.astype("float32") / 255
		x_te = x_te.astype("float32") / 255

		x_tr = x_tr.reshape(x_tr.shape[0], -1)
		x_te  = x_te.reshape(x_te.shape[0], -1)

		# Brightness Jitter
		brightness = np.random.uniform(0.1, 0.5, size=(x_tr.shape[0],))
		x_tr = x_tr + np.expand_dims(brightness, -1)
		x_tr = np.clip(x_tr, 0, 1)

		clf = MLPClassifier(hidden_layer_sizes=(128, 32, 16), max_iter=40)
		clf.fit(x_tr, y_tr)
		print(clf.score(x_tr, y_tr))
		print(clf.score(x_te, y_te))

		dump(clf, 'mnist_models/brightness_3')
	else:
		raise ValueError("Dataset not supported yet")
