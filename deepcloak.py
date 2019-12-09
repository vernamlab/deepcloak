"""
### TEST VECTORS ###
# test inputs to get L0 and Linfinity distances for adversarial samples
from deepcloak import *
data_meta = DC_meta('data/b15openssl110_c20_s1000_f1000_h5.h5', 20, 1000, 1000, 5)
model_path = 'models/mdl_b15openssl110_c20_s1000_f1000_h5_e10.h5'
shuffle_classes = 0
distilled = 0
pert_count = 100
adv_attack = 'GSA'
pert_results = craft_adversarial(model_path, data_meta, adv_attack, distilled=0, shuffle_classes=shuffle_classes, pert_count=pert_count)

atks = ['GA','GSA','LBFGSA','SLSQPA','SMA','BA','GBA','CRA','AUNA','AGNA','BUNA','SPNA']
for atk in atks:
	craft_adversarial(model_path, data_meta, atk, shuffle_classes=1, pert_count=pc)

craft_adversarial(model_path, data_meta, 'GA', shuffle_classes=1, pert_count=pc)
craft_adversarial(model_path, data_meta, 'GSA', shuffle_classes=1, pert_count=pc)
craft_adversarial(model_path, data_meta, 'LBFGSA', shuffle_classes=1, pert_count=pc)
craft_adversarial(model_path, data_meta, 'SLSQPA', shuffle_classes=1, pert_count=pc)
craft_adversarial(model_path, data_meta, 'SMA', shuffle_classes=1, pert_count=pc)
#craft_adversarial(model_path, data_meta, 'BA', shuffle_classes=1, pert_count=pc)
craft_adversarial(model_path, data_meta, 'GBA', shuffle_classes=1, pert_count=pc)
craft_adversarial(model_path, data_meta, 'CRA', shuffle_classes=1, pert_count=pc)
craft_adversarial(model_path, data_meta, 'AUNA', shuffle_classes=1, pert_count=pc)
craft_adversarial(model_path, data_meta, 'AGNA', shuffle_classes=1, pert_count=pc)
craft_adversarial(model_path, data_meta, 'BUNA', shuffle_classes=1, pert_count=pc)
craft_adversarial(model_path, data_meta, 'SPNA', shuffle_classes=1, pert_count=pc)

# test vectors for feature size experiments
from deepcloak import *
meta = DC_meta('data/batch11-1100_c4_s1100_f1000_h6.h5', 2, 100, 1000, 5)
n_epoch = 3
for f in [50, 30, 20, 10]:
	meta.n_feature = f
	print('******* Using %d features *******' %f)
	train_classifier(meta, n_epoch)

# old test vectors
from deepcloak import *
data_meta = DC_meta('data/b15openssl110_c20_s10000_f1000_h5_e100.h5', 20, 10000, 1000, 5)
n_epoch = 5
sl_path = 'sl_temp30_batch11-1100_c4_s1100_f1000_h6.h5'
temperature = 10
early_stop_patience = 5
perturbation_path = 'adv/perturbed_samples/b15openssl110_c20_s10000_f1000_h5_e100_p100/GSA_p100'
model_path = 'models/mdl_temp1DD_b15openssl110_c20_s10000_f1000_h5_e100.h5'
model_path = 'models/mdl_b15openssl110_c20_s10000_f1000_h5_e100.h5'
"""


class DC_meta:
	def __init__(self, path=None, n_class=None, n_sample=None, n_feature=None, n_hpc=None):
		self.path = path
		self.n_class = n_class
		self.n_sample = n_sample
		self.n_feature = n_feature
		self.n_hpc = n_hpc
	#TODO I can add a slice_data method here as: sliced = DC_meta.slice_data(meta_new)

class Pert_result:
	def __init__(self, sample=None, org_class=None, org_acc=None, pert_class=None,\
				 pert_acc=None, MAD=None, MSD=None, L0D=None, LinfD=None):
		self.sample = []
		self.org_class = []
		self.org_acc = []
		self.pert_class = []
		self.pert_acc = []
		self.MAD = []
		self.MSD = []
		self.L0D = []
		self.LinfD = []

	def append(self, sample, org_class, org_acc, pert_class, pert_acc, MAD, MSD, L0D, LinfD):
		self.sample.append(sample)
		self.org_class.append(org_class)
		self.org_acc.append(org_acc)
		self.pert_class.append(pert_class)
		self.pert_acc.append(pert_acc)
		self.MAD.append(MAD)
		self.MSD.append(MSD)
		self.L0D.append(L0D)
		self.LinfD.append(LinfD)

	def to_dict(self):
		k = len(self.sample)
		v = self.org_class, self.org_acc, self.pert_class, self.pert_acc, self.MAD, self.MSD, self.L0D, self.LinfD
		dict_pr = {}
		for i in range(k):
			#print('index is %d' %(k+idx_start))
			dict_pr[i] = v[0][i], v[1][i], v[2][i], v[3][i], v[4][i], v[5][i], v[6][i], v[7][i]
		return dict_pr


def DD_test():
	from deepcloak import adv_test, create_folder, DC_meta, craft_adversarial
	from pandas import DataFrame
	from tabulate import tabulate

#    model_path = 'models/retrained_mdl_b15openssl110_c20_s10000_f1000_h5_e100.h5'
	data_meta = DC_meta('data/b15openssl110_c20_s10000_f1000_h6.h5', 20, 10000, 1000, 5)
	attacks = ['AGNA', 'AUNA', 'BUNA', 'CRA', 'GA', 'GBA', 'GSA', 'LBFGSA', 'SLSQPA', 'SMA', 'SPNA']
	temps = [1, 2, 5, 10, 20, 30, 40, 50, 100]
	res = []
	for adv in attacks:
		for temperature in temps:
			model_path = 'models/mdl_temp' + str(temperature) + 'DD_b15openssl110_c20_s10000_f1000_h5_e100.h5'
			perturbation_path = 'adv/perturbed_samples/b15openssl110_c20_s10000_f1000_h5_e100_p100000/' + adv +'_p100000'
			score = adv_test(model_path, perturbation_path) # testing previously crafted adv. samples
			res.append([adv, temperature ,score[1] ,score[0]])
#            craft_adversarial(model_path, data_meta, adv, shuffle_classes=1, pert_count=100) # crafting new adv. samples
	headers = ('AdvAttack', 'DistTemp', 'Accuracy', 'Loss')
	results = DataFrame(data=res, columns=headers)
	create_folder('DD')
	with open('DD/retrained_mdl_b15openssl110_c20_s10000_f1000_h5_e100' + '.tsv', "w") as f:
		f.write(tabulate(results, tablefmt="tsv", headers=headers))

def retrain_test():
	import h5py, os, time
	import tensorflow as tf
	from keras.models import load_model
	from keras import backend as K
	from keras.utils import to_categorical
	from deepcloak import create_folder, DC_meta, compile_and_train, print_results
	from pandas import DataFrame
	from tabulate import tabulate

	model_path = 'models/retrained_mdl_b15openssl110_c20_s10000_f1000_h5_e100.h5'
	model = load_model(model_path)
	mname = model_path.split(sep='/')[-1].split(sep='.')[0]
	model.name = mname
	bname = mname.split(sep='_')[0]+ '_' + mname.split(sep='_')[1] + '_' +  mname.split(sep='_')[2]
	res = []
	attacks = ['AGNA', 'AUNA', 'BUNA', 'CRA', 'GA', 'GBA', 'GSA', 'LBFGSA', 'SLSQPA', 'SMA', 'SPNA']
	for adv in attacks:
		perturbation_path = 'adv/perturbed_samples/b15openssl110_c20_s10000_f1000_h5_e100_p100000/' + adv + '_p100000'
		# loading the new data and labels
		# this data will be used to test if the perturbations still work
		try:
			# print('Getting the new data from %s' %perturbation_path)
			f = h5py.File(perturbation_path, 'r')  # open the original file in read-only mode
			x_train = f['perturbed_samples'].value
			x_train = x_train.squeeze(4)
			x_train = x_train.reshape((x_train.shape[0],) + model.input_shape[1:])
			y_train = to_categorical(f['perturbed_classes'].value, model.output_shape[-1])
			f.close()
		except IOError as exc:
			print('IO error, could NOT save the file')
			if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
				raise  # Propagate other kinds of IOError.
		score = model.evaluate(x=x_train, y=y_train, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1], '\n') # accuracy here represents the percentage
												# of perturbations that still work against the given model
		res.append([adv, score[1], score[0]])
	headers = ('AdvAttack', 'Accuracy', 'Loss')
	results = DataFrame(data=res, columns=headers)
	create_folder('DD')
	with open('DD/retrained_mdl_b15openssl110_c20_s10000_f1000_h5_e100' + '.tsv', "w") as f:
		f.write(tabulate(results, tablefmt="tsv", headers=headers))


def adv_test(model_path, perturbation_path):
	import h5py, os, time
	from deepcloak import create_folder, compile_and_train, print_results
	import tensorflow as tf
	from pandas import DataFrame
	from keras.models import load_model
	from keras import backend as K
	from keras.utils import to_categorical
	from keras.utils.generic_utils import get_custom_objects
	temperature = int(model_path.split(sep='temp')[1].split(sep='DD')[0])
	def fn(correct, predicted): # custom loss function with temperature
		return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted / temperature)

	model = load_model(model_path, custom_objects={'fn': fn})
	mname = model_path.split(sep='/')[-1].split(sep='.')[0]
	model.name = mname
	bname = mname.split(sep='_')[2] + '_DD' + str(temperature)

	# loading the new data and labels
	# this data will be used to test if the perturbations still work
	try:
		#print('Getting the new data from %s' %perturbation_path)
		f = h5py.File(perturbation_path, 'r')  # open the original file in read-only mode
		x_train = f['perturbed_samples'].value
		x_train = x_train.squeeze(4)
		x_train = x_train.reshape((x_train.shape[0],) + model.input_shape[1:])
		y_train = to_categorical(f['perturbed_classes'].value, model.output_shape[-1])
		f.close()
	except IOError as exc:
		print('IO error, could NOT save the file')
		if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
			raise  # Propagate other kinds of IOError.

	# Evaluate model on test data
	score = model.evaluate(x=x_train, y=y_train, verbose=0)
	print('Distillation temperature is: %d' %temperature)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1], '\n')
	return score

def train_distilled(data_meta, n_epoch, temperature):
	# Defensive distillation trainin steps:
	# Train a model with temperature T, save output probabilities (soft labels)
	# Train a new model with temperature T again, use soft labels in training
	# Save the new model for adversarial crafting
	from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
	from keras.optimizers import Adam
	from keras.callbacks import ModelCheckpoint, EarlyStopping
	from keras.models import Sequential
	from keras.utils import to_categorical
	from sklearn.preprocessing import normalize, MinMaxScaler
	from pandas import DataFrame
	import tensorflow as tf
	import h5py, os, time, errno
	from deepcloak import create_folder, compile_and_train, print_results
	from keras.models import load_model
	from keras import backend as K
	import numpy as np
	def fn(correct, predicted): # custom loss function with temperature
		return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted / temperature)

	n_class = data_meta.n_class
	n_sample = data_meta.n_sample
	n_feature = data_meta.n_feature
	n_hpc = data_meta.n_hpc
	path = data_meta.path
	input_shape = (n_feature, n_hpc, 1)
	output_shape = n_class

	# The FIRST model with a custom temperature, trained with HARD labels
	model = Sequential()
	model.add(Conv2D(100, (10, input_shape[1]), activation='relu', input_shape=input_shape,
					 data_format='channels_last', name='conv1'))
	model.add(MaxPooling2D(pool_size=(10, 1), data_format='channels_last', name='pool1'))
	model.add(BatchNormalization(name='norm1'))
	model.add(Dropout(0.25, name='drop1'))  # regularization layer
	model.add(Conv2D(200, (10, 1), activation='relu', data_format='channels_last', name='conv2'))
	model.add(MaxPooling2D(pool_size=(10, 1), data_format='channels_last', name='pool2'))
	model.add(BatchNormalization(name='norm2'))
	model.add(Dropout(0.25, name='drop2'))
	model.add(Flatten(name='flatten1'))  # makes the weights from CONV layer 1-dimensional before passing them to the
	model.add(Dense(output_shape*20, activation='relu', name='dense1'))  # output_shape is the number of classes
	model.add(Dropout(0.5, name='drop3'))
	model.add(Dense(output_shape, name='dense2'))  # check alternatives for software exit layer
	#model.add(BatchNormalization(name='norm3'))

	model.compile(loss=fn, optimizer='Adam', metrics=['accuracy'])

	mname = 'mdl_' + 'temp' + str(temperature) + '_' + (path.split(sep='/')[1]).split(sep='_')[0] + \
			'_c' + str(n_class) + '_s' + str(n_sample) + '_f' + \
			str(n_feature) + '_h' + str(n_hpc) + '_e' + str(n_epoch) + '.h5'
	model.name = mname
	bname = mname.split(sep='_')[1] + '_' + mname.split(sep='_')[2]
	create_folder('models')
	model_path = 'models/' + mname

	x_train_sorted, y_train_sorted = extract_train_samples(data_meta)
	x_test_sorted, y_test_sorted = extract_test_samples(data_meta)
	#x_train_shuffled, y_train_shuffled = shuffle_array(x_train_sorted, y_train_sorted)
	#x_test, y_test = shuffle_array(x_test, y_test )

	checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=0, save_weights_only=0,
								 save_best_only=1, mode='auto', period=1)
	#early_stopping = EarlyStopping(monitor='val_acc', patience=early_stop_patience)

	start_time = time.time()
	history = model.fit(x=x_train_sorted, y=to_categorical(y_train_sorted), batch_size=32, epochs=n_epoch, verbose=0,
						callbacks=[checkpoint], validation_split=0.25, shuffle=1)
	train_time = time.time() - start_time

	h = DataFrame(history.history)
	print_results(h, bname, mname)  # draw plots, tables etc.

	print('Normal model at temperature %f' %temperature)
	print(h)

	# Evaluate model on test data
	score = model.evaluate(x=x_test_sorted, y=to_categorical(y_test_sorted), verbose=0)
	print('\nTest loss:', score[0])
	print('Test accuracy:', score[1])
	print('Train&Test time for %i samples:' % (x_train_sorted.shape[0]* n_class), int(train_time / 60), 'minutes.\n')

	# Obtaining the soft labels for the distilled model training
	#print('Obtaining soft labels from the model')
	soft_labels = model.predict(x_train_sorted)
	normalized = np.empty(soft_labels.shape)
	for idx in range(soft_labels.shape[0]):
		x = soft_labels[idx]
		normalized[idx] = (x - min(x))
	soft_labels = normalize(normalized, axis=1, norm='l1')

	# The SECOND model with a custom temperature, trained with SOFT labels
	# This model has the Defensive Distillation
	#print('Training the model with soft labels with temperature: %s' % str(temperature))
	model = Sequential()
	model.add(Conv2D(50, (10, input_shape[1]), activation='relu', input_shape=input_shape,
					 data_format='channels_last', name='conv1'))
	model.add(MaxPooling2D(pool_size=(10, 1), data_format='channels_last', name='pool1'))
	model.add(BatchNormalization(name='norm1'))
	model.add(Dropout(0.25, name='drop1'))  # regularization layer
	model.add(Conv2D(100, (10, 1), activation='relu', data_format='channels_last', name='conv2'))
	model.add(MaxPooling2D(pool_size=(10, 1), data_format='channels_last', name='pool2'))
	model.add(BatchNormalization(name='norm2'))
	model.add(Dropout(0.25, name='drop2'))
	model.add(Flatten(name='flatten1'))  # makes the weights from CONV layer 1-dimensional before passing them to the
	model.add(Dense(output_shape*20, activation='relu', name='dense1'))  # output_shape is the number of classes
	model.add(Dropout(0.5, name='drop3'))
	model.add(Dense(output_shape, name='dense2'))  # check alternatives for software exit layer
	#model.add(BatchNormalization(name='norm3'))
	#def fn(correct, predicted):
	#    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted / temperature)
	model.compile(loss=fn, optimizer='Adam', metrics=['accuracy'])

	mname = 'mdl_' + 'temp' + str(temperature) + 'DD_' +(path.split(sep='/')[1]).split(sep='_')[0] + \
			'_c' + str(n_class) + '_s' + str(n_sample) + '_f' + \
			str(n_feature) + '_h' + str(n_hpc) + '_e' + str(n_epoch) + '.h5'
	model.name = mname
	bname = mname.split(sep='_')[1] + '_' + mname.split(sep='_')[2]
	create_folder('models')
	model_path = 'models/' + mname

	#x_train, y_train = extract_train_samples(data_meta)
	#x_test, y_test = extract_test_samples(data_meta)
	#x_train, y_train = shuffle_array(x_train, y_train)
	#x_test, y_test = shuffle_array(x_test, y_test )

	start_time = time.time()
	history = model.fit(x=x_train_sorted, y=soft_labels, batch_size=32, epochs=n_epoch, verbose=0,
						callbacks=[checkpoint], validation_split=0.25, shuffle=1)
	train_time = time.time() - start_time

	h = DataFrame(history.history)
	print_results(h, bname, mname)  # draw plots, tables etc.

	print('Distilled model at temperature %f' %temperature)
	print(h)

	# Evaluate model on test data
	score = model.evaluate(x_test_sorted, to_categorical(y_test_sorted), verbose=0)
	print('\nDistilled model test loss:', score[0])
	print('Distilled model test accuracy:', score[1])
	print('Distilled model Train&Test time for %i samples:' % (x_train_sorted.shape[0] * n_class), int(train_time / 60), 'minutes.\n')

	results = model.predict(x_test_sorted)
	normalized = np.empty(results.shape)
	for idx in range(y_test_sorted.shape[0]):
		x = results[idx]
		normalized[idx] = (x - min(x))
	results = normalize(normalized, axis=1, norm='l1')


def get_soft_labels(model_path, data_meta):
	import h5py, os, time, errno
	from deepcloak import create_folder, compile_and_train, print_results
	from pandas import DataFrame
	from keras.models import load_model
	from keras import backend as K
	from keras.utils import to_categorical

	# Loading the original model
	print('Loading the model from the path: %s' %model_path)
	model = load_model(model_path)
	print('Loading the data from the path: %s' %data_meta.path)
	#TODO do the reads properly, with exceptions
	dpath = data_meta.path.split(sep='/')[1]

	x_train, y_train = extract_train_samples(data_meta)
	print('Obtaining soft labels from the model')
	soft_labels = model.predict(x_train)
	temp = int(model_path.split(sep='temp')[1].split(sep='_')[0])
	#mname = model_path.split(sep='/')[1].split(sep='mdl_')[1]

	n_class = data_meta.n_class
	n_sample = data_meta.n_sample
	n_feature = data_meta.n_feature
	n_hpc = data_meta.n_hpc
	n_epoch = model_path.split(sep='_e')[1].split(sep='.')[0]

	sl_path = 'data/sl_temp' + str(temp) + '_' + dpath.split(sep='_')[0] + \
			'_c' + str(n_class) + '_s' + str(n_sample) + '_f' + \
			str(n_feature) + '_h' + str(n_hpc) + '_e' + str(n_epoch) + '.h5'

	try:
		print('Storing the soft labels as: '+ sl_path)
		f = h5py.File(sl_path, 'w')  # create a new file in write mode
		f.create_dataset('soft_labels', data=soft_labels, compression='gzip', compression_opts=1, dtype='float32')
		f.close()
	except IOError as exc:
		print('IO error, could NOT save the file')
		if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
			raise  # Propagate other kinds of IOError.
	print('Finished writing the soft labels to the disk')
	#print(soft_labels)


def adv_retrain(model_path, data_path, n_epoch):
	import h5py, os, time
	from deepcloak import create_folder, compile_and_train, print_results
	from pandas import DataFrame
	from keras.models import load_model
	from keras import backend as K
	from keras.utils import to_categorical

	# Loading the original model
	model = load_model(model_path)
	mname = model_path.split(sep='/')[-1] # Renaming the model to its original name since Keras does not save model names for some reason
	mname = 'retrained_' + mname if mname.split(sep='_')[0] != 'retrained' else mname
	model.name = mname
	bname = 'retrained_' + mname.split(sep='_')[2]

	# loading the new data and labels
	# this data will be used to re-train the original model with adversarial samples
	# think of it as a form of vaccination against adversarial attacks
	try:
		print('Getting the new data from %s' %data_path)
		f = h5py.File(data_path, 'r')  # open the original file in read-only mode
		x_train = f['perturbed_samples'].value
		x_train = x_train.squeeze(4)
		y_train = to_categorical(f['perturbed_classes'].value, model.output_shape[-1])
		f.close()
	except IOError as exc:
		print('IO error, could NOT save the file')
		if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
			raise  # Propagate other kinds of IOError.

	x_train = x_train.reshape((x_train.shape[0],) +  model.input_shape[1:])
	print('Retraining data shape: %s' %str(x_train.shape))
	print('Retraining label shape: %s' % str(y_train.shape))
	print('Model input shape: %s' % str(model.input_shape))
	print('Model output shape: %s' % str(model.output_shape))

	# Compile and train the model
	start_time = time.time()
	history = compile_and_train(model, x_train, y_train, n_epoch,
						checkpoint_en=1, early_stop_patience=0, tensorboard_en=0)
	train_time = time.time() - start_time
	print('Re-training took %d minutes' %(train_time/60))
	h = DataFrame(history.history)
	print_results(h, bname, mname)


def print_results(h, bname, mname):
	import time, h5py, os
	import tabulate, graphviz, pydot
	import numpy as np
	from deepcloak import create_folder
	from pandas import DataFrame
	from keras.utils import plot_model
	import matplotlib
	matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
	from matplotlib import pyplot as plt

	rpath = 'results/' + bname + '/'
	create_folder(rpath)
	with open(rpath + mname + '.tsv', "w") as f:
		f.write(tabulate.tabulate(h, tablefmt="tsv", headers=h.keys(),
								  # showindex=range(1,n_epoch+1),
								  numalign=1))

	fpath = 'figures/' + bname + '/'
	create_folder(fpath + 'acc/')
	create_folder(fpath + 'loss/')
	# Plot and save the training results over epochs
	fig = plt.figure()
	plt.plot(h['val_loss'])
	plt.plot(h['val_acc'])
	plt.plot(h['loss'])
	plt.plot(h['acc'])
	plt.xlabel('Epoch')
	plt.minorticks_on()
	plt.legend(
		['Validation Loss (min=%.5g)' % h['val_loss'].min(), 'Validation Accuracy (max=%.5g)' % h['val_acc'].max(),
		 'Training Loss (min=%.5g)' % h['loss'].min(), 'Training Accuracy (max=%.5g)' % h['acc'].max()])
	fig_title = bname
	plt.title(fig_title)
	# fig.show()
	fig.savefig(fpath + 'fig_' + mname + '.pdf')
	fig.clear()
	plt.close()

	create_folder('figures/acc')
	# Plot and save the accuracy results over epochs
	fig = plt.figure()
	plt.plot(h['val_acc'])
	plt.plot(h['acc'])
	plt.minorticks_on()
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(['Validation (max=%.5g)' % h['val_acc'].max(), 'Training (max=%.5g)' % h['acc'].max()])
	fig_title = bname
	plt.title(fig_title)
	fig.savefig(fpath + 'acc/fig_acc_' + mname + '.pdf')
	fig.clear()
	plt.close()

	create_folder('figures/loss')
	# Plot and save the loss results over epochs
	fig = plt.figure()
	plt.plot(h['val_loss'])
	plt.plot(h['loss'])
	plt.minorticks_on()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(['Validation (min=%.5g)' % h['val_loss'].min(), 'Training (min=%.5g)' % h['loss'].min()])
	fig_title = bname
	plt.title(fig_title)
	fig.savefig(fpath + 'loss/fig_loss_' + mname + '.pdf')
	fig.clear()
	plt.close()

def create_folder(path):
	import os
	directory = os.path.dirname(path+'/')
	if not os.path.exists(directory):
		os.makedirs(directory)


def merge_data(meta1, meta2): # simple function to combine 2 h5 files, assume that the file sizes are equal
	from deepcloak import DC_meta
	import h5py
	import numpy as np
	# sample data path; 'data/b15openssl110_c20_f1000_h6.h5'

	f = h5py.File(meta1.path, 'r')  # 'r' means that hdf5 file is open in read-only mode
	data1 = f['data'].value
	f.close()

	f = h5py.File(meta2.path, 'r')  # 'r' means that hdf5 file is open in read-only mode
	data2 = f['data'].value
	f.close()

	n_class1 = meta1.n_class
	n_sample1 = meta1.n_sample
	n_feature1 = meta1.n_feature
	n_hpc1 = meta1.n_hpc
	n_class2 = meta2.n_class
	n_sample2 = meta2.n_sample
	n_feature2 = meta2.n_feature
	n_hpc2 = meta2.n_hpc

	data1 = data1[:n_class1, :n_sample1, :n_feature1, :n_hpc1]
	data2 = data2[:n_class2, :n_sample2, :n_feature2, :n_hpc2]

	merged = np.concatenate((data1, data2), axis=1)
	merged_name = meta1.path.split(sep='_')[0].split(sep='/')[1] + '+' + meta2.path.split(sep='_')[0].split(sep='/')[1]
	print('Combined data shape: %s' %(merged.shape,))
	new_shape = merged.shape

	try:
		print('Storing the dataset as: '+ 'data/' + merged_name + '_c' + str(new_shape[0]) +
			'_s' + str(new_shape[1]) + '_f' + str(new_shape[2]) + '_h' + str(new_shape[3]) + '.h5')
		hf = h5py.File('data/' + merged_name + '_c' + str(new_shape[0]) +
			'_s' + str(new_shape[1]) + '_f' + str(new_shape[2]) + '_h' + str(new_shape[3]) + '.h5', 'w')
		hf.create_dataset('data', data=merged, compression='gzip', compression_opts=1, dtype='float32')
		hf.create_dataset('n_class', data=new_shape[0], dtype='i')
		hf.create_dataset('n_sample', data=new_shape[1], dtype='i')
		hf.create_dataset('n_feature', data=new_shape[2], dtype='i')
		hf.create_dataset('n_hpc', data=new_shape[3], dtype='i')
		hf.close()
	except IOError as exc:
		print('IO error, could NOT save the file')
		if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
			raise  # Propagate other kinds of IOError.    
	print('Finished writing the combined data to the disk.')


def craft_adversarial(model_path, data_meta, adv_attack, distilled=0, shuffle_classes=1, pert_count=-1):
	import os, h5py, time
	import numpy as np
	import foolbox
	import tensorflow as tf
	import cleverhans as ch
	import tabulate
	import matplotlib
	matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
	from matplotlib import pyplot as plt
	from pandas import DataFrame
	from keras.models import load_model
	from keras import backend as K
	from deepcloak import create_folder, Pert_result, DC_meta, extract_test_samples, shuffle_array
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	K.set_image_dim_ordering('tf')  # this is very important!

	org_pert_count = pert_count
	start_time = time.time()
	print('This function uses the test samples of the below dataset for perturbation.\n' + data_meta.path)
	# Load the model and the test data
	x_test, y_test = extract_test_samples(data_meta)
	if shuffle_classes:
		x_test, y_test = shuffle_array(x_test, y_test)
		print('Shuffling the classes.')

	if distilled ==1:
		temperature = int(model_path.split(sep='temp')[1].split(sep='DD')[0])
		def fn(correct, predicted):  # custom loss function with temperature
			return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted / temperature)
		org_model = load_model(model_path, custom_objects={'fn': fn})  # load a model with Defensive Distillation
	else:
		org_model = load_model(model_path)  # load the DL model
	org_model.name = model_path.split(sep='models/')[1]
	bounds = (x_test.min(), x_test.max()*2)
	# bounds = (-100,100) # test bounds
	criterion = foolbox.criteria.TopKMisclassification(1)
	adv_model = foolbox.models.KerasModel(model=org_model, bounds=bounds, channel_axis=3, predicts='probabilities')
	attack_dict = {
		#  Gradient-based Attacks
		'GSA':foolbox.attacks.GradientSignAttack,
		'IGSA':foolbox.attacks.IterativeGradientSignAttack,
		'GA':foolbox.attacks.GradientAttack,
		'IGA':foolbox.attacks.IterativeGradientAttack,
		'LBFGSA':foolbox.attacks.LBFGSAttack,
		'SLSQPA':foolbox.attacks.SLSQPAttack,
		'SMA':foolbox.attacks.SaliencyMapAttack,
		'DFA':foolbox.attacks.DeepFoolAttack,   # needs modification
		'NFA': foolbox.attacks.NewtonFoolAttack,

		#  Score-based attacks
		'SPA':foolbox.attacks.SinglePixelAttack, # thinks HPC dim is image columns dim - fix it
		'LSA':foolbox.attacks.LocalSearchAttack, # thinks HPC dim is image columns dim - fix it
		'ALBFGSA':foolbox.attacks.ApproximateLBFGSAttack,

		#  Decision-based attacks
		'BA':foolbox.attacks.BoundaryAttack,  # works in n-steps to reduce perturbation size, very slow with large # of steps
		'GBA':foolbox.attacks.GaussianBlurAttack,
		'CRA':foolbox.attacks.ContrastReductionAttack,
		'AUNA':foolbox.attacks.AdditiveUniformNoiseAttack,
		'AGNA':foolbox.attacks.AdditiveGaussianNoiseAttack,
		'BUNA':foolbox.attacks.BlendedUniformNoiseAttack,  # very fast.
		'SPNA':foolbox.attacks.SaltAndPepperNoiseAttack,  # very fast.
		#'RA':foolbox.attacks.ResetAttack  # Starts with an adversarial and resets as many values as possible to the values of the original.
	}

	# TODO allow other criterias
	selected_attack = attack_dict[adv_attack](model=adv_model, criterion=criterion)
	print(attack_dict[adv_attack].__name__ + ' has been selected as the adversarial attack.' \
											 '\nCalculating the perturbations now.')
	# TODO add standardization to the test data
	pr = Pert_result()
	if pert_count == -1 or None:
		print('Calculating perturbations for all of the given test samples.')
	else:
		print('Calculating only %d perturbations.' % pert_count)

	# TODO rename to more explanatory names
	perturbed_samples = [] # perturbed samples, saved for re-training the model - vaccination against adv. pert.
	perturbed_classes = [] # perturbed samples, saved for re-training the model - vaccination against adv. pert.
	feature_count = (x_test.shape[1]*x_test.shape[2])

	for i in range(y_test.shape[0]):
		if pert_count == 0:
			print('Finished perturbation crafting.')
			break
		try:
			x_org = x_test[i]
			y_org = y_test[i]  #.argmax()
			# adv_model.predictions(x_org)  # pred using adv wrapped model
			pred = org_model.predict(np.expand_dims(x_org,0))
			#  logits = adv_model.get_layer('dense2').get_weights() # some attacks require logits rather than the full model
			#  images = tf.placeholder(tf.float32, (None, 1000, 6, 1)) # input shape

			#pert_result.append([i, org_class, org_acc, pert_class, pert_acc, MAD, MSD])
			if y_org.argmax()!=pred.argmax():
				print("Sample: %d for pert_count: %d already misclassified." %(i, pert_count))
				#pr.append(i, 'misclassified', None, None, None, None, None, None, None)
			else:
				#            selected_attack = foolbox.attacks.ContrastReductionAttack(model=adv_model, criterion=criterion)
				x_pert = selected_attack(input_or_adv=x_org, label=y_org.argmax())
				#            x_pert = selected_attack(image=x_org, label=y_org, input_or_adv=x_org)
				x_pert = np.expand_dims(x_pert, 0)
				y_pert = org_model.predict(x_pert)
				org_class = pred.argmax()
				org_acc = pred.max() * 100
				pert_class = y_pert.argmax()  # TODO rename
				pert_acc = y_pert.max() * 100
				# print('\nOriginal class:  %d\t predicted as: %d\twith %.2f %% confidence' %(y_org, org_class, org_acc) )
				# print('Perturbed class: %d\twith %.2f %% confidence' % (pert_class, pert_acc))
				MAD = foolbox.distances.MeanAbsoluteDistance(x_org, x_pert, bounds).value
				MSD = foolbox.distances.MeanSquaredDistance(x_org, x_pert, bounds).value
				L0D = foolbox.distances.L0(x_org, x_pert).value / feature_count
				LinfD = foolbox.distances.Linfinity(x_org, x_pert, bounds).value

				pr.append(i, org_class, org_acc, pert_class, pert_acc, MAD, MSD, L0D, LinfD)
				perturbed_samples.append(x_pert)
				perturbed_classes.append(y_org) # consider renaming before deployment
				pert_count -= 1
		except ValueError:
			print('Could not create perturbation for sample: %d' %i)
			#pr.append(i, None , None, None, None, None, None, None, None)

	mname = org_model.name + '_p' + str(org_pert_count)   # the data/model name with class, sample, feature, hpc, epoch info
	bname = mname.split(sep='_')[0]  # the data/model batch name

	adv_sample_path = 'adv/perturbed_samples/' + mname + '/' # store adv. samples in a model specific folder
	create_folder(adv_sample_path)
	f_new = h5py.File(adv_sample_path + adv_attack + '_p' + str(org_pert_count) , 'w') # use the model name
	f_new.create_dataset('perturbed_samples', data=perturbed_samples, compression='gzip', compression_opts=1, dtype='float32')
	f_new.create_dataset('perturbed_classes', data=perturbed_classes, dtype='float32')
	#f_new.create_dataset('pert_results', data=pr, dtype='float32') #TODO test if working correctly
	f_new.close()

	prd = pr.to_dict()
	prdf = DataFrame(prd).transpose()
	pr_headers = 'sample', 'org_class', 'org_acc', 'pert_class', 'pert_acc', 'MAD', 'MSD', 'L0D', 'LinfD' #list(pr.__dict__.keys())
	rpath = 'adv/presults/' + mname + '/'
	fpath = 'adv/pfigures/' + mname + '/'
	create_folder(rpath)
	create_folder(fpath)

	with open(rpath + 'pert_res_'+ adv_attack+'_' + mname + '.tsv', "w") as f:
		f.write(tabulate.tabulate(prdf, tablefmt="tsv",headers=pr_headers,numalign=1))

	fig = plt.figure()
	MAD_filtered = list(filter(None, pr.MAD))
	MSD_filtered = list(filter(None, pr.MSD))
	L0D_filtered = list(filter(None, pr.L0D))
	LinfD_filtered = list(filter(None, pr.LinfD))
	plt.semilogy(MAD_filtered, linewidth=0.5)
	plt.semilogy(MSD_filtered, linewidth=0.5)
	plt.semilogy(L0D_filtered, linewidth=0.5)
	plt.semilogy(LinfD_filtered, linewidth=0.5)
	plt.minorticks_on()
	plt.xlabel('Sample number')
	plt.ylabel('Percentage of change (Normalized)')
	plt.legend(['Mean Absolute Distance (mean=%.5f)'%(np.mean(MAD_filtered)), \
				'Mean Squared Distance (mean=%.5f)'%(np.mean(MSD_filtered)), \
				'L0 Norm Distance (mean=%.5f)' % (np.mean(L0D_filtered)), \
				'L-infinity Norm Distance (mean=%.5f)' % (np.mean(LinfD_filtered)) ])
	#    plt.title(org_model.name[4:-3] + '\n' + adv_attack + ' perturbed sample distances')
	plt.title(adv_attack + ' perturbed sample distances')
	fig.savefig(fpath + 'fig_adv_dist_'+ adv_attack+'_' + mname + '.pdf')
	#fig.show()
	fig.clear()
	plt.close('all')

	fig = plt.figure()
	org_acc_filtered = list(filter(None, pr.org_acc))
	pert_acc_filtered = list(filter(None, pr.pert_acc))
	plt.plot(org_acc_filtered, linewidth=0.5)
	plt.plot(pert_acc_filtered, linewidth=0.5)
	plt.minorticks_off()
	plt.xlabel('Sample number')
	plt.ylabel('Classification confidence')
	plt.legend(['Original sample (mean=%.5f)'%(np.mean(org_acc_filtered)),'Perturbed sample (mean=%.5f)'%(np.mean(pert_acc_filtered))])
	#    plt.title(org_model.name[4:-3] + '\n' + adv_attack + ' classification confidence')
	plt.title(adv_attack + ' classification confidence')
	fig.savefig(fpath + 'fig_adv_conf_'+ adv_attack+'_'+ mname + '.pdf')
	#fig.show()
	fig.clear()
	plt.close('all')

	"""
	xo = np.swapaxes(np.squeeze(x_org), 0, 1)
	xp = np.swapaxes(np.squeeze(x_pert), 0, 1)
	fig = plt.figure()
	plt.plot(xp[0] + 1, linewidth=0.2)
	plt.plot((xp[1] - xo[1]) + 2, linewidth=0.2)
	plt.plot(xo[2] + 3, linewidth=0.2)
	plt.plot(xo[3] + 4, linewidth=0.2)
	plt.plot(xo[4] + 5, linewidth=0.2)
	plt.ylabel('HPC')
	plt.legend(['Total instructions', 'Branch instructions', 'Total cache references',
				'L1 instruction cache miss',  'L1 data dache miss' ])
	fig.savefig('pert.pdf')
	fig.clear()
	plt.close('all')
   
		plt.plot(xo[0] - xp[0] + 1, linewidth=0.2)
		plt.plot(xo[1] - xp[1] + 2, linewidth=0.2)
		plt.plot(xo[2] - xp[2] + 3, linewidth=0.2)
		plt.plot(xo[3] - xp[3] + 4, linewidth=0.2)
		plt.plot(xo[4] - xp[4] + 5, linewidth=0.2)
	"""

	run_time = time.time()-start_time
	print('Adversarial sample creation took %i minutes.\n' %(int(run_time/60)) )
	return pr


def extract_test_samples(meta):
	import h5py, os
	import numpy as np

	path = meta.path
	n_class = meta.n_class
	n_sample = meta.n_sample
	n_feature = meta.n_feature
	n_hpc = meta.n_hpc

	# data preparation
	f = h5py.File(path,'r')  # 'r' means that hdf5 file is open in read-only mode
	data = f['data'].value
	f.close()
	data = data[:n_class, :n_sample, :n_feature, :n_hpc]

	for hpc in range(data.shape[-1]):
		data[:, :, :, hpc] = (data[:, :, :, hpc] - data[:, :, :, hpc].mean()) \
						 / data[:, :, :, hpc].std()  # Standardizing

	n_test = int(np.floor(0.2 * n_sample))  # we'll split our data 60-20-20 for train, validation, test

	# Create Class Labels
	labels = np.zeros((data.shape[0], data.shape[1]))  # taking the shape of the data
	for c in range(n_class):
		labels[c] = c
	#   Create training and test sets
	x_test = data[:, -n_test:, :, :]
	y_test = labels[:, -n_test:]
	#   Reshape training and test sets to get rid of class dimension
	x_test = x_test.reshape(x_test.shape[0] * x_test.shape[1], n_feature, n_hpc, 1)
	y_test = y_test.reshape(y_test.shape[0] * y_test.shape[1])
	return x_test, y_test

def extract_train_samples(meta):
	#TODO combine extract test and train samples functions into an -extract_samples function with test/train arg
	import h5py, os
	import numpy as np

	path = meta.path
	n_class = meta.n_class
	n_sample = meta.n_sample
	n_feature = meta.n_feature
	n_hpc = meta.n_hpc

	# data preparation
	f = h5py.File(path,'r')  # 'r' means that hdf5 file is open in read-only mode
	data = f['data'].value
	f.close()
	data = data[:n_class, :n_sample, :n_feature, :n_hpc]

	for hpc in range(data.shape[-1]):
		data[:, :, :, hpc] = (data[:, :, :, hpc] - data[:, :, :, hpc].mean()) \
							 / data[:, :, :, hpc].std()  # Standardizing


	n_train = int(np.floor(0.8 * n_sample))  # we'll split our data 60-20-20 for train, validation, test
	# Create Class Labels
	labels = np.zeros((data.shape[0], data.shape[1]))  # taking the shape of the data
	for c in range(n_class):
		labels[c] = c
	#   Create training and test sets
	x_train = data[:, :n_train, :, :]
	y_train = labels[:, :n_train]
	#   Reshape training and test sets to get rid of class dimension
	x_train = x_train.reshape(x_train.shape[0] * x_train.shape[1], n_feature, n_hpc, 1)
	y_train = y_train.reshape(y_train.shape[0] * y_train.shape[1])
	return x_train, y_train


def shuffle_array(x_dim, y_dim):
	import numpy.random
	assert len(x_dim) == len(y_dim)
	p = numpy.random.permutation(len(x_dim))
	return x_dim[p], y_dim[p]

from keras import callbacks
class My_Callback(callbacks.Callback):
	def on_train_begin(self, logs={}):
		return
	def on_train_end(self, logs={}):
		return
	def on_epoch_begin(self, logs={}):
		return
	def on_epoch_end(self, epoch, logs={}):
		x, y = self.test_data
		loss, acc = self.model.evaluate(x_test, y_test, verbose=1)
		print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
		return
	def on_batch_begin(self, batch, logs={}):
		return
	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		return

def filter_log():
	import os, sys, glob
	import errno

	path = sys.argv[1]
	print('Filtering the log files at path:\n%s' % path)
	files = glob.glob(path + '*')
	for name in files:  # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
		try:
			print('Filename: ' + name)
			shell_command1 = 'head -n 47 ' + name + ' > ../temp.txt'
			shell_command2 = 'egrep "step - loss:|test|Test|Post|params" ' + name + ' >> ../temp.txt'
			shell_command3 = 'cat ../temp.txt > ' + name
			os.system(shell_command1)
			os.system(shell_command2)
			os.system(shell_command3)
		except IOError as exc:
			print('readfile failed')
			if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
				raise  # Propagate other kinds of IOError.

def compile_and_train(model, x_train, y_train, n_epoch, checkpoint_en=1, early_stop_patience=0, tensorboard_en=0):
	from deepcloak import create_folder
	from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	create_folder('models')
	filepath = 'models/' + model.name #+ '.{epoch:02d}-{val_acc:.2f}.h5'
	if filepath[-3:] != '.h5': filepath = filepath + '.h5'

	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_weights_only=0,
								 save_best_only=1, mode='auto', period=1)
	tensorboard = TensorBoard(log_dir='./logs/', histogram_freq=0, batch_size=32)
#                               ,write_graph=0, write_grads=0,write_images=0)
	early_stopping = EarlyStopping(monitor='val_acc', patience=early_stop_patience)

	selected_callbacks = []
	selected_callbacks.append(checkpoint) if checkpoint_en else None
	selected_callbacks.append(early_stopping) if early_stop_patience else None
	selected_callbacks.append(tensorboard) if tensorboard_en else None

	history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=n_epoch, verbose=0,
						callbacks=selected_callbacks, validation_split=0.25, shuffle=1)

	return history

def slice_data(DC_meta):
	import h5py
	from deepcloak import create_folder

	path = DC_meta.path
	n_class = DC_meta.n_class
	n_sample = DC_meta.n_sample
	n_feature = DC_meta.n_feature
	n_hpc = DC_meta.n_hpc

	f = h5py.File(path, 'r')  # open the original file in read-only mode
	data_orig = f['data'].value
	n_class_orig = f['n_class'].value
	n_sample_orig = f['n_sample'].value
	n_feature_orig = f['n_feature'].value
	n_hpc_orig = f['n_hpc'].value
	f.close()

	data_new = data_orig[:n_class, :n_sample, :n_feature, :n_hpc]
	new_path = path.split(sep='_')[0]
	new_path = new_path+'_c'+str(n_class)+'_s'+str(n_sample) +'_f'+str(n_feature)+'_h'+str(n_hpc)+'.h5'

	f_new = h5py.File(new_path, 'w-')
	f_new.create_dataset('data', data=data_new, compression='gzip', compression_opts=1, dtype='float32')
	f_new.create_dataset('n_class', data=n_class, dtype='i')
	f_new.create_dataset('n_sample', data=n_sample, dtype='i')
	f_new.create_dataset('n_feature', data=n_feature, dtype='i')
	f_new.create_dataset('n_hpc', data=n_hpc, dtype='i')
	f_new.close()

	print('Saved the sliced data as:\n%s ' %new_path)
	return new_path

def read_data(DC_meta):  # this function reads txt files and writes the data into a h5 file
	import errno
	import numpy as np
	from pandas import read_csv
	import h5py
	import glob

	path = DC_meta.path
	n_class = DC_meta.n_class
	n_sample = DC_meta.n_sample
	n_feature = DC_meta.n_feature
	n_hpc = DC_meta.n_hpc

	# path = path + '*_@(?|??|100).txt'
	# path = path + '*_@(?).txt'
	files = glob.glob(path + '*.txt')
	class_list = read_csv(path + 'class_list', header=None).values[:, 0].tolist()[:n_class]
	# print('path is: ' + path + '\nfiles are:\n' + str(files))
	i = 0
	hpc = (0,1,2,3,4,5)[:n_hpc]
	c = [-1] * len(class_list)
	data = np.empty((n_class, n_sample, n_feature, n_hpc), dtype='int64')
	for name in files:  # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
		try:
			for cls_name in class_list: # find the file class from the class_list file
				if cls_name in name:
					cls_index = class_list.index(cls_name)
					#print('class is: %s at %s' % (cls_name, cls_index))
					c[cls_index] += 1
					if c[cls_index] < n_sample:
						data[cls_index, c[cls_index], :, :] = read_csv(name, delimiter=',', header=None, nrows=n_feature,usecols=hpc).values
					break
#                else:
#                    print('\nFile class not found! Check filenames and the class_list file.')
#                    return
			#print('file no: %s \tclass: %s\nfilename: %s' % (i, cls_index, name))
			i = i + 1
		except IOError as exc:
			print('readfile failed')
			if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
				raise  # Propagate other kinds of IOError.
	try:
		print('Storing the dataset as: '+path[:-1]+'_c'+str(n_class)+'_s'+str(n_sample)+'_f'+str(n_feature)+'_h'+str(n_hpc)+'.h5')
		hf = h5py.File(path[:-1]+'_c'+str(n_class)+'_s'+str(n_sample)+'_f'+str(n_feature)+'_h'+str(n_hpc)+ '.h5', 'w')
		hf.create_dataset('data', data=data, compression='gzip', compression_opts=1, dtype='float32')
		hf.create_dataset('n_class', data=n_class, dtype='i')
		hf.create_dataset('n_sample', data=n_sample, dtype='i')
		hf.create_dataset('n_feature', data=n_feature, dtype='i')
		hf.create_dataset('n_hpc', data=n_hpc, dtype='i')
		hf.close()
	except IOError as exc:
		print('IO error, could NOT save the file')
		if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
			raise  # Propagate other kinds of IOError.


def dc_cnn_model(input_shape, output_shape):
	from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
	from keras.models import Sequential
	# Define model architecture, declare Sequential model
	model = Sequential()
	kernel1 = (int(input_shape[0]/100), input_shape[1])
	model.add(Conv2D(50, kernel1, activation='relu', input_shape=input_shape, # TODO fix static nr of kernels #50 (10, input_shape[1])
					 data_format='channels_last', name='conv1'))
	# first 3 parameters correspond to the number of convolution filters to use, and their dimensions
	# the number of rows in each convolution kernel,
	# and the number of columns in each convolution kernel, respectively.
	model.add(MaxPooling2D(pool_size=(10,1), data_format='channels_last', name='pool1'))
	model.add(BatchNormalization(name='norm1'))
	model.add(Dropout(0.25, name='drop1'))  # regularization layer
	kernel2 = (int(input_shape[0] / 100), 1) # TODO fix static nr of kernels
	model.add(Conv2D(100, kernel2, activation='relu', data_format='channels_last', name='conv2')) # 100, (100,1)
	model.add(MaxPooling2D(pool_size=(10,1), data_format='channels_last', name='pool2')) # TODO fix static
	model.add(BatchNormalization(name='norm2'))
	model.add(Dropout(0.25, name='drop2'))
	model.add(Flatten(name='flatten1'))  # makes the weights from CONV layer 1-dimensional before passing them to the
	model.add(Dense(output_shape*20, activation='relu', name='dense1'))  # output_shape is the number of classes
	model.add(Dropout(0.5, name='drop3'))
	model.add(Dense(output_shape, activation='softmax', name='dense2'))  # check alternatives for software exit layer

	return model

def train_classifier(meta, n_epoch): # TODO change all metas to DC_meta
	import time, h5py, os
	import tabulate, graphviz, pydot
	import numpy as np
	from deepcloak import compile_and_train, shuffle_array, dc_cnn_model, create_folder
	from pandas import DataFrame
	from cgitb import reset
	from keras import backend as K
	from keras.utils import np_utils, plot_model
	import matplotlib
	matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
	from matplotlib import pyplot as plt
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	K.set_image_dim_ordering('tf')  # this is very important!
	np.random.seed(123)  # for reproducibility

	path = meta.path
	n_class = meta.n_class
	n_sample = meta.n_sample
	n_feature = meta.n_feature
	n_hpc = meta.n_hpc

	f = h5py.File(path, 'r')  # 'r' means that hdf5 file is open in read-only mode
	data = f['data'].value  # out train+test data
	n_class_orig = f['n_class'].value
	n_sample_orig = f['n_sample'].value
	n_feature_orig = f['n_feature'].value
	n_hpc_orig = f['n_hpc'].value
	f.close()

	if path.split(sep='.')[1] == 'mat': # Check if the provided data is in 'mat' format. If so, dimensions have to be rearranged.
		print('You are using a matlab data file. Matlab data array needs to be reshaped.')
		print('Old data shape is: ' + str(data.shape))
		data = np.swapaxes(data, 0, 3)
		data = np.swapaxes(data, 2, 1)
		print('New data shape is: ' + str(data.shape))

	# Print the data information if the user specified not to use the whole dataset.
	# TODO check for larger size as well
	if (n_class != n_class_orig or n_sample != n_sample_orig or n_feature != n_feature_orig or n_hpc != n_hpc_orig):
		print('The imported dataset has;\n' +
			str(n_class_orig) + ' classes, ' +
			str(n_sample_orig) + ' samples, ' +
			str(n_feature_orig) + ' features, ' +
			str(n_hpc_orig) + ' HPCs')

		print('Using only the following;\n' +
			str(n_class) + ' classes, ' +
			str(n_sample) + ' samples, ' +
			str(n_feature) + ' features, ' +
			str(n_hpc) + ' HPCs')

	data = data[:n_class, :n_sample, :n_feature, :n_hpc]  # reduce number of samples and features
	input_shape = (n_feature, n_hpc, 1)
	model = dc_cnn_model(input_shape, n_class)
	mname = 'mdl_' + (path.split(sep='/')[1]).split(sep='_')[0] + \
				 '_c' + str(n_class) + '_s' + str(n_sample) + '_f' + \
				 str(n_feature) + '_h' + str(n_hpc) + '_e' + str(n_epoch) + '.h5'
	model.name = mname
	bname = mname.split(sep='_')[1]
	model.summary()
	# plot_model(model, to_file=model.name+'.png') # pydot has a Keras bug that is not fixed yet
	n_sample_orig = n_sample # we use this when we want to get test results for partial datasets

	### Pre-Processing
	#        print('\nPre-processing Data shape:\t' + str(data.shape))
	#   Normalization of the raw data by either Standardizing or Normalizing
	for hpc in range(data.shape[-1]): # TODO let the user choose between normalization and standardization
		data[:, :, :, hpc] = (data[:, :, :, hpc] - data[:, :, :, hpc].mean()) \
							 / data[:, :, :, hpc].std()  # Standardizing
		# data[:, :, :, hpc] = data[:, :, :, hpc] / data[:, :, :, hpc].max() # Normalizing
	#        print('Post-processing Data shape:\t' + str(data.shape))

# while n_sample > 0:       # loop for partial training, disabled atm
	n_train = int(np.floor(0.8*n_sample))
	n_test  = n_sample - n_train

	#   Create Class Labels
	labels = np.zeros((n_class, n_sample))  # taking the shape of the data
	for cls in range(n_class):
		labels[cls] = cls

	#   Create training and test sets
	x_train = data[:, :n_train, :, :]
	y_train = labels[:, :n_train]
	x_test = data[:, n_train:, :, :]
	y_test = labels[:, n_train:]
	#   Reshape training and test sets to get rid of class dimension
	x_train = x_train.reshape(x_train.shape[0] * x_train.shape[1], n_feature, n_hpc,1)
	y_train = y_train.reshape(y_train.shape[0] * y_train.shape[1])
	x_test = x_test.reshape(x_test.shape[0] * x_test.shape[1], n_feature, n_hpc,1)
	y_test = y_test.reshape(y_test.shape[0] * y_test.shape[1])

	assert len(x_train) == len(y_train), 'x_train and y_train length is not equal'
	x_train, y_train = shuffle_array(x_train, y_train)  # Shuffle the training hence val sets

	# Preprocess class labels, convert 1-dimensional class arrays to 10-dimensional class matrices
	y_train = np_utils.to_categorical(y_train, n_class)
	y_test = np_utils.to_categorical(y_test, n_class)
	print('Number of samples for;\nTraining:\t%i\tValidation:\t%i\tTesting:\t%i\n' %(n_train*0.75, n_train*0.25 , n_test))

	# Create the CNN model
	input_shape = (x_train.shape[1], x_train.shape[2], 1) # TODO redundant - remove
	#model = dc_cnn_model(input_shape, n_class)
	model.name = mname

	# Compile and train the model
	start_time = time.time()
	history = compile_and_train(model, x_train, y_train, n_epoch,
								checkpoint_en=1, early_stop_patience=0, tensorboard_en=0)
	train_time = time.time() - start_time

	h = DataFrame(history.history)
	print_results(h, bname, mname)
	# Evaluate model on test data
	score = model.evaluate(x_test, y_test, verbose=0)
	print('\nTest loss:', score[0])
	print('Test accuracy:', score[1])
	print('Train&Test time for %i samples:' % (n_train * n_class), int(train_time / 60), 'minutes.\n')
	# n_sample -= int(n_sample_orig/10)
	# del model, x_train, x_test, y_train, y_test
	# %reset_selective -f model x_train x_test

	# Save model and weights
	# model.save('models/' + model.name+'.h5')
	# print('Saved the trained model to: models/%s' %model.name+'.h5')

