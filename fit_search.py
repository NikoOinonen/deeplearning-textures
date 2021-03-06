
import os
import time
import numpy as np
import keras.backend as K
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

from utils import SeqConstructor, plot_history, evaluate_model
from models import resnet_minc

def make_random_parameters(param_ranges):
    params = []
    for r in param_ranges:
        param = np.random.uniform(low=r[0], high=r[1])
        if param > 1.0:
            param = int(param)
        params += [param]
    return params
   
# Settings
model_dir = 'resnet_search'
dataset = 'minc'
data_path = './minc-2500/'
img_size = (128,128)
batch_size = 32
epochs = 100
split = [0.6,0.1,0.3]
workers = 4
normalize = True
augmentations = [('fliplr', 0.5), ('crop', (0.70, 1.0))]
N_fits = 15
param_ranges = [(10, 20), (20, 30), (30, 40), (40, 60), (60, 100), (0.1,0.4), (0.0,0.2)]

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if model_dir[-1] is not '/':
    model_dir += '/'

# Divide dataset into train/val/test sets
constructor = SeqConstructor(data_path, dataset=dataset, img_size=img_size, batch_size=batch_size, split=split, augmentations=augmentations, normalize=normalize)
train_seq = constructor.seq_in_mode('training')
val_seq = constructor.seq_in_mode('validation', shuffle=False)
test_seq = constructor.seq_in_mode('test', shuffle=False)
print(constructor.class_dict)

if not os.path.exists(model_dir+'results.csv'):
    with open(model_dir+'results.csv','w') as f:
        f.write('Params\n')
        f.write('filter_1;filter_2;filter_3;filter_4;dense;dropout1;dropout2;best_val_loss\n')

for i in range(N_fits):
    
    print('\nStarting test %d\n' % i)
    start_time = time.time()

    # Generate random hyperparameters
    params = make_random_parameters(param_ranges)
    print(params)

    test_dir = model_dir+'_'.join([str(p)[:5] for p in params])+'/'
    if not os.path.exists(test_dir+'/Checkpoints/'):
        os.makedirs(test_dir+'/Checkpoints/')
    
    # Make model
    model = resnet_minc(input_shape=img_size+(3,), params=params)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file=test_dir+'model.png', show_shapes=True)
    
    # Fit model
    checkpointer = ModelCheckpoint(test_dir+'Checkpoints/weights_{epoch:d}.h5', save_weights_only=True)
    earlystopper = EarlyStopping(patience=10)
    history = model.fit_generator(train_seq, epochs=epochs, verbose=1, validation_data=val_seq,
        callbacks=[checkpointer, earlystopper], use_multiprocessing=True, workers=workers)
    model.save_weights(test_dir+'model.h5')
    plot_history(history, outdir=test_dir)

    with open(model_dir+'results.csv','a') as f:
        f.write(';'.join([str(p) for p in params])+';'+str(min(history.history['val_loss']))+'\n')

    # Evaluate model
    evaluate_model(model, test_seq, test_dir, constructor.classes)

    # Destroy old model graph to avoid clutter
    K.clear_session()

    print('Total time taken: '+str(time.time() - start_time))






