
import os
import time
import numpy as np
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from utils import SeqConstructor, plot_history, evaluate_model
from models import resnet_fmd
   
# Settings
skip_training = False
model_dir = 'resnet_fmd'
data_path = './FMD/'
img_size = (128,128)
batch_size = 8
epochs = 100
split = [0.6,0.1,0.3]
workers = 4
normalize = True
augmentations = [('fliplr', 0.5), ('noise', 0.5), ('crop', (0.7, 1.0))]
params = [6, 15, 12, 0.55]

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(model_dir+'/Checkpoints/'):
    os.makedirs(model_dir+'/Checkpoints/')
if model_dir[-1] is not '/':
    model_dir += '/'

# Divide dataset into train/val/test sets
constructor = SeqConstructor(data_path, dataset='fmd', img_size=img_size, batch_size=batch_size, split=split, augmentations=augmentations, normalize=normalize)
train_seq = constructor.seq_in_mode('training')
val_seq = constructor.seq_in_mode('validation', shuffle=False)
test_seq = constructor.seq_in_mode('test', shuffle=False)
print(constructor.class_dict)

# Make model
model = resnet_fmd(input_shape=img_size+(3,), params=params)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
plot_model(model, to_file=model_dir+'model.png')

# Fit/load model
if not skip_training:
    
    checkpointer = ModelCheckpoint(model_dir+'Checkpoints/weights_{epoch:d}.h5', save_weights_only=True)
    history = model.fit_generator(train_seq, epochs=epochs, use_multiprocessing=True, validation_data=val_seq, callbacks=[checkpointer], workers=workers)
    best_epoch = np.argmin(history.history['val_loss'])
    model.load_weights(model_dir+'Checkpoints/weights_%d.h5' % best_epoch)
    model.save_weights(model_dir+'model.h5')
    plot_history(history, outdir=model_dir)

else:
    model.load_weights(model_dir+'model.h5')

# Evaluate model
evaluate_model(model, test_seq, model_dir, constructor.classes)








