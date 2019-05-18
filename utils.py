import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from random import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from pandas import DataFrame

class ImgSequence(Sequence):

    '''
    A class for loading and preprocessing images. For use with keras' fit_generator, evaluate_generator
    or predict_generator.

    Arguments
        img_list: List of paths to image samples to load
        img_size: (optional) All images are finally scaled to this size
        batch_size: (optional) Sample batch size
        shuffle: (optional) Whether to shuffle the sample set on epoch end
        augmentations: (optional) list of augmentation steps to apply to images. Possible options:
            ('flipud', p): Vertical flip of image with probability p
            ('fliplr', p): Horizontal flip of image with probability p
            ('crop', (p1,p2)): Crop of image with random size between p1 and p2 proportion of original image
            ('noise', a)): Random uniform noise with zero mean and maximum amplitude a, applied after normalization
    '''

    def __init__(self, img_list, img_size=(128,128), batch_size=32, shuffle=True, augmentations=[], norm_mean=0.0, norm_std=1.0):
        self.img_list = img_list
        self.img_size = img_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.augmentations = augmentations
        self._check_augs()

    def _check_augs(self):
        for aug in self.augmentations:
            if aug[0] == 'flipud' or aug[0] == 'fliplr':
                if aug[1] < 0.0 or aug[1] > 1.0:
                    raise ValueError('flipud and fliplr augmentation probabilities must be between 0.0 and 1.0, but found %f' % aug[1])
            elif aug[0] == 'crop':
                if aug[1][0] < 0.0 or aug[1][1] < 0.0 or aug[1][0] > 1.0 or aug[1][1] > 1.0:
                    raise ValueError('crop augmentation proportions must be between 0.0 and 1.0, but found ' + str(aug[1]))
            elif aug[0] == 'noise':
                continue
            else:
                raise ValueError('Invalid augmentation '+str(aug))
    
    def on_epoch_end(self):
        if self.shuffle:
            shuffle(self.img_list)

    def preprocess(self, img_batch):
        a = 0.0
        for aug in self.augmentations:
            if aug[0] == 'flipud':
                img_batch = self.flip(img_batch, 0, aug[1])
            elif aug[0] == 'fliplr':
                img_batch = self.flip(img_batch, 1, aug[1])
            elif aug[0] == 'crop':
                img_batch = self.crop(img_batch, min(aug[1]), max(aug[1]))
            elif aug[0] == 'noise':
                a = aug[1]
        img_batch = self.rescale(img_batch)
        if a > 0.0:
            img_batch = self.add_noise(img_batch, a)
        return img_batch

    def rescale(self, img_batch):
        imgs_rescaled = np.zeros((len(img_batch),)+self.img_size+(3,))
        for i, img in enumerate(img_batch):
            new_img = (imresize(img, size=self.img_size, interp='nearest') - self.norm_mean) / self.norm_std
            imgs_rescaled[i] = new_img
        return imgs_rescaled

    def crop(self, img_batch, p1, p2):
        assert p1 < p2
        new_imgs = []
        for img in img_batch:
            p = np.random.uniform(p1, p2)
            new_size = [int(np.round(s*p)) for s in img.shape[:2]]
            loc = []
            for s1, s2 in zip(img.shape[:2], new_size):
                if s1-s2 > 0:
                    loc.append(np.random.randint(low=0, high=s1-s2))
                else:
                    loc.append(0)
            new_img = img[loc[0]:loc[0]+new_size[0], loc[1]:loc[1]+new_size[1]]
            new_imgs.append(new_img)
        return new_imgs

    def flip(self, img_batch, axis, p):
        new_imgs = np.zeros_like(img_batch)
        for i, img in enumerate(img_batch):
            if np.random.rand() < p:
                new_imgs[i] = np.flip(img, axis)
            else:
                new_imgs[i] = img
        return new_imgs

    def add_noise(self, img_batch, a):
        return img_batch + 2 * a *(np.random.rand(*img_batch.shape) - 0.5)

    def __len__(self):
        return int( np.ceil( len(self.img_list) / float(self.batch_size) ) )

    def __getitem__(self, idx):

        if idx < 0:
            idx = self.__len__() + idx

        batch_data = self.img_list[idx*self.batch_size:(idx+1)*self.batch_size]
        imgs = []
        classes = []
        for sample in batch_data:
            imgs.append(imread(sample[0], mode='RGB'))
            classes.append(sample[1])
        imgs = self.preprocess(imgs)
        classes = np.array(classes)

        return imgs, classes

def load_data2(data_path, dataset, img_size=(128,128), split=[0.5,0.2,0.3]):
    
    if dataset == 'minc':
        with open(data_path+'categories.txt', 'r') as f:
            classes = f.read().split()
        n_split = [int(s*2500) for s in split]
        x_train = np.zeros((n_split[0]*23,)+img_size+(3,))
        x_val = np.zeros((n_split[1]*23,)+img_size+(3,))
        x_test = np.zeros((n_split[2]*23,)+img_size+(3,))
        y_train = np.zeros((n_split[0], 23))
        y_val = np.zeros((n_split[1], 23))
        y_test = np.zeros((n_split[2], 23))
        x = [x_train, x_val, x_train]
        y = [y_train, y_val, y_train]
        class_dict = dict()
        ind_pos = np.cumsum([0]+n_split)
        for ci, c in enumerate(classes):
            class_dict[ci] = c
            base = data_path+'images/'+c+'/'+c+'_00'
            one_hot_vec = np.zeros(len(classes))
            one_hot_vec[ci] = 1
            for i in range(3):
                y[i][n_split[i]*ci:n_split[i]*(ci+1)]
                for img_ind in range(ind_pos[i], ind_pos[i+1]):
                    img = imread(base+'%04d.jpg'%img_ind, mode='RGB')
                    img = imresize(img, size=img_size, interp='nearest')
                    x[i][n_split[i]*ci+img_ind] = img

def load_data(data_path, dataset, split=[0.5,0.2,0.3], img_size=(128,128), batch_size=32, gen_kwargs={}):
    
    if dataset == 'minc':
        with open(data_path+'categories.txt', 'r') as f:
            classes = f.read().split()
        train_list = []
        val_list = []
        test_list = []
        train_classes = []
        val_classes = []
        test_classes = []
        class_dict = dict()
        n_split = [int(s*2500) for s in split]
        ind_pos = np.cumsum(n_split)
        for ci, c in enumerate(classes):
            class_dict[ci] = c
            base = data_path+'images/'+c+'/'+c+'_00'
            train_list += [base+'%04d.jpg'%i for i in range(0, ind_pos[0])]
            val_list += [base+'%04d.jpg'%i for i in range(ind_pos[0], ind_pos[1])]
            test_list += [base+'%04d.jpg'%i for i in range(ind_pos[1], ind_pos[2])]
            train_classes += [c]*n_split[0]
            val_classes += [c]*n_split[1]
            test_classes += [c]*n_split[2]

    gen = ImageDataGenerator(**gen_kwargs)
    train_frame = DataFrame(data={'images': train_list, 'classes': train_classes})
    val_frame = DataFrame(data={'images': val_list, 'classes': val_classes})
    test_frame = DataFrame(data={'images': test_list, 'classes': test_classes})
    train_generator = gen.flow_from_dataframe(train_frame, x_col='images', y_col='classes', target_size=img_size, batch_size=batch_size, shuffle=True)
    val_generator = gen.flow_from_dataframe(val_frame, x_col='images', y_col='classes', target_size=img_size, batch_size=batch_size, shuffle=False)
    test_generator = gen.flow_from_dataframe(test_frame, x_col='images', y_col='classes', target_size=img_size, batch_size=batch_size, shuffle=False)
    
    return train_generator, val_generator, test_generator, class_dict

class ImgSequence2(Sequence):

    '''
    A class for loading and preprocessing images. For use with keras' fit_generator, evaluate_generator
    or predict_generator.

    Arguments
        img_list: List of paths to image samples to load
        img_size: (optional) All images are finally scaled to this size
        batch_size: (optional) Sample batch size
        shuffle: (optional) Whether to shuffle the sample set on epoch end
        augmentations: (optional) list of augmentation steps to apply to images. Possible options:
            ('flipud', p): Vertical flip of image with probability p
            ('fliplr', p): Horizontal flip of image with probability p
            ('crop', (p1,p2)): Crop of image with random size between p1 and p2 proportion of original image
    '''

    def __init__(self, img_list, img_size=(128,128), batch_size=32, shuffle=True, augmentations=[], normalization=True):
        self.img_list = img_list
        self.img_size = img_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.normalization = normalization
        self.augmentations = augmentations
        self._init_normalization()
    
    def on_epoch_end(self):
        if self.shuffle:
            shuffle(self.img_list)

    def preprocess(self, img_batch):
        for aug in self.augmentations:
            if aug[0] == 'flipud':
                img_batch = self.flip(img_batch, 0, aug[1])
            elif aug[0] == 'fliplr':
                img_batch = self.flip(img_batch, 1, aug[1])
            elif aug[0] == 'crop':
                img_batch = self.crop(img_batch, min(aug[1]), max(aug[1]))
        img_batch = self.rescale(img_batch)
        return img_batch

    def rescale(self, img_batch):
        imgs_rescaled = np.zeros((len(img_batch),)+self.img_size+(3,))
        for i, img in enumerate(img_batch):
            new_img = imresize(img, size=self.img_size, interp='nearest')
            imgs_rescaled[i] = new_img
        return imgs_rescaled

    def crop(self, img_batch, p1, p2):
        assert p1 < p2
        new_imgs = []
        for img in img_batch:
            p = np.random.uniform(p1, p2)
            new_size = [int(np.round(s*p)) for s in img.shape[:2]]
            loc = []
            for s1, s2 in zip(img.shape[:2], new_size):
                loc.append(np.random.randint(low=0, high=s1-s2))
            new_img = img[loc[0]:loc[0]+new_size[0], loc[1]:loc[1]+new_size[1]]
            new_imgs.append(new_img)
        return new_imgs

    def flip(self, img_batch, axis, p):
        new_imgs = np.zeros_like(img_batch)
        for i, img in enumerate(img_batch):
            if np.random.rand() < p:
                new_imgs[i] = np.flip(img, axis)
            else:
                new_imgs[i] = img
        return new_imgs

    def __len__(self):
        return int( np.ceil( len(self.img_list) / float(self.batch_size) ) )

    def __getitem__(self, idx):

        if idx < 0:
            idx = self.__len__() + idx

        batch_data = self.img_list[idx*self.batch_size:(idx+1)*self.batch_size]
        imgs = []
        classes = []
        for sample in batch_data:
            imgs.append(imread(sample[0], mode='RGB'))
            classes.append(sample[1])
        imgs = self.preprocess(imgs)
        classes = np.array(classes)

        return imgs, classes


class SeqConstructor():

    '''
    A class used for constructing a Sequence object for passing into keras' fit_generator, evaluate_generator
    or predict_generator.

    Arguments
        data_path: Path to directory with dataset
        dataset: Dataset to use. One of 'minc', 'fmd', or 'kth-tips2'
        img_size: (optional) All images are finally scaled to this size
        batch_size: (optional) Sample batch size
        split: (optional) Portion of samples to assign to [train, val, test] sets. Must sum to at most 1.0.
        shuffle: (optional) Whether to shuffle the sample set on epoch end
        augmentations: (optional) list of augmentation steps to apply to images. Possible options:
            ('flipud', p): Vertical flip of image with probability p
            ('fliplr', p): Horizontal flip of image with probability p
            ('crop', (p1,p2)): Random crop of image with random size between p1 and p2 proportion of original image
        normalize: (optional) Whether the data is normalized featurewise by subtracting the mean and dividing by standard deviation
    '''

    def __init__(self, data_path, dataset, img_size=(128,128), batch_size=32, split=[0.5,0.2,0.3], shuffle=True, augmentations=[], normalize=True):
    
        assert sum(split) <= (1.0+1e-7)

        if data_path[-1] != '/':
            data_path += '/'

        self.data_path = data_path
        self.dataset = dataset
        self.split = split
        self.shuffle = shuffle
        self.img_size = img_size
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.normalize = normalize

        self.train_list = []
        self.val_list = []
        self.test_list = []
        
        if dataset == 'minc':
            with open(self.data_path+'categories.txt', 'r') as f:
                self.classes = f.read().split()
            self.class_dict = dict()
            ind_pos = [int(i) for i in np.round(np.cumsum(split) * 2500.0)]
            for ci, c in enumerate(self.classes):
                self.class_dict[ci] = c
                base = self.data_path+'images/'+c+'/'+c+'_00'
                one_hot_vec = np.zeros(len(self.classes))
                one_hot_vec[ci] = 1
                self.train_list += [[base+'%04d.jpg'%i, one_hot_vec] for i in range(0, ind_pos[0])]
                self.val_list += [[base+'%04d.jpg'%i, one_hot_vec] for i in range(ind_pos[0], ind_pos[1])]
                self.test_list += [[base+'%04d.jpg'%i, one_hot_vec] for i in range(ind_pos[1], ind_pos[2])]
        elif dataset == 'fmd':
            self.classes = ['fabric', 'foliage', 'glass', 'leather', 'metal', 'paper', 'plastic', 'stone', 'water', 'wood']
            self.class_dict = dict()
            for ci, c in enumerate(self.classes):
                self.class_dict[ci] = c
                files = glob.glob(self.data_path+'image/'+c+'/*.jpg')
                ind_pos = [int(i) for i in np.round(np.cumsum(split) * 100.0)]
                for i, f in enumerate(files):
                    one_hot_vec = np.zeros(len(self.classes))
                    one_hot_vec[ci] = 1
                    if i < ind_pos[0]:
                        self.train_list += [[f, one_hot_vec]]
                    elif i < ind_pos[1]:
                        self.val_list += [[f, one_hot_vec]]
                    else:
                        self.test_list += [[f, one_hot_vec]]
        elif dataset == 'kth-tips2':    #TODO Current set division is biased
            self.classes = ['aluminium_foil', 'brown_break', 'corduroy', 'cork', 'cotton', 'cracker', 'lettuce_leaf', 'linen', 'white_bread', 'wood', 'wool']
            self.class_dict = dict()
            for ci, c in enumerate(self.classes):
                self.class_dict[ci] = c
                for sample in ['sample_a', 'sample_b', 'sample_c', 'sample_d']:
                    files = glob.glob(self.data_path+'image/'+c+'/*.jpg')
                    ind_pos = [int(i) for i in np.round(np.cumsum(split) * 100.0)]
                    for i, f in enumerate(files):
                        one_hot_vec = np.zeros(len(self.classes))
                        one_hot_vec[ci] = 1
                        if i < ind_pos[0]:
                            self.train_list += [[f, one_hot_vec]]
                        elif i < ind_pos[1]:
                            self.val_list += [[f, one_hot_vec]]
                        else:
                            self.test_list += [[f, one_hot_vec]]
        else:
            raise ValueError('dataset has to be one of "minc", "fmd", or "kth-tips2", but found '+str(dataset))

        self._fit_normalization()

    def _fit_normalization(self, fit_samples=1000):
        if self.normalize:
            imgs = []
            batch_data = self.train_list[:min(fit_samples, len(self.train_list))]
            for sample in batch_data:
                imgs.append(imresize(imread(sample[0], mode='RGB'), size=self.img_size, interp='nearest'))
            imgs = np.array(imgs)
            self.norm_mean = imgs.mean(axis=0)
            self.norm_std = imgs.std(axis=0)
        else:
            self.norm_mean = 0.0
            self.norm_std = 1.0
    
    def seq_in_mode(self, mode, shuffle=None):
        '''
        Arguments
            mode: One of 'training', 'validation' or 'test'. Decides from which set the samples are returned.
            shuffle: (optional) Whether to shuffle the sample set on epoch end

        Returns: ImgSequence instance
        '''

        if mode == 'training':
            img_list = self.train_list
            augmentations = self.augmentations
        elif mode == 'validation':
            img_list = self.val_list
            augmentations = []
        elif mode == 'test':
            img_list = self.test_list
            augmentations = []
        else:
            raise ValueError('Invalid mode '+str(mode))

        if shuffle is None:
            shuffle = self.shuffle
        
        return ImgSequence(img_list, self.img_size, self.batch_size, shuffle, augmentations, self.norm_mean, self.norm_std)

class LossCalculator:
    '''
    Calculates the losses for each prediction of a model
    '''

    def __init__(self, model):
        self.model = model
        self._make_loss_functions()

    def _make_loss_functions(self):
        import keras.backend as K
        self.loss_funs = []
        for i, lf in enumerate(self.model.loss_functions):
            t = K.placeholder(shape=self.model.outputs[i].shape)
            p = K.placeholder(shape=self.model.outputs[i].shape)
            self.loss_funs.append(K.function([t,p], [lf(t, p)]))

    def __call__(self, true, preds=None, X=None):
        '''
        Arguments: 
            true: Reference outputs
            preds: (optional) Predicted outputs
            X: (optional) Inputs that will be used for making predictions if preds == None
        Note: At least one of preds or X has to be provided
        '''

        if preds is None:
            if X is None:
                raise ValueError('preds and X cannot both be None')
            else:
                preds = self.model.predict_on_batch(X)

        if not isinstance(true, list):
            true = [true]
        if not isinstance(preds, list):
            preds = [preds]
        
        losses = np.zeros((true[0].shape[0], len(true)))
        for i, (t, p) in enumerate(zip(true, preds)):
            loss = self.loss_funs[i]([t,p])[0]
            sh = loss.shape
            if len(sh) > 1:
                loss = np.mean(loss.reshape((sh[0],-1)), axis=1)
            losses[:,i] = loss
        
        if losses.shape[1] == 1:
            losses = losses[:,0]
        if losses.shape[0] == 1 and losses.ndim == 1:
            losses = losses[0]

        return losses

def evaluate_model(model, test_seq, model_dir, classes):

    metrics = Metrics(classes)
    eval_loss = 0
    start_time = time.time()
    loss_calculator = LossCalculator(model)
    for i, (X, true) in enumerate(test_seq):
        
        # Make predictions on batch and calculate loss
        preds = model.predict_on_batch(X)
        loss = np.mean(loss_calculator(true, preds)) 
        eval_loss = eval_loss + (loss - eval_loss) / (i+1)

        # Keep track of metrics
        metrics.add_preds(preds, true)
        
        # Print progress
        ETA = (time.time() - start_time) / (i+1) * (len(test_seq) - (i+1))
        print('Evaluating test set %d/%d - ETA: %ds' % (i+1, len(test_seq), ETA), end='\r' if (i+1) < len(test_seq) else '\n')

    # Output information
    metrics.plot(outdir=model_dir)
    print('Loss on test set: '+str(np.mean(eval_loss)))
    print('Accuracy: '+str(metrics.acc))
    print('Mean Precision: '+str(np.mean(metrics.prec)))
    print('Mean Recall: '+str(np.mean(metrics.rec)))

class Metrics:
    
    def __init__(self, classes):
        self.classes = classes
        self.n_classes = len(classes)
        self.conf_mat = np.zeros((self.n_classes, self.n_classes))

    def add_preds(self, preds, true):
        
        # Add to confusion matrix
        pred_classes = np.argmax(preds[:,:self.n_classes], axis=-1)
        true_classes = np.argmax(true[:,:self.n_classes], axis=-1)
        for t, p in zip(true_classes, pred_classes):
            self.conf_mat[t, p] += 1

    def plot(self, outdir='./', verbose=1):
        
        # Confusion matrix

        conf_mat = self.conf_mat
        conf_mat_norm = np.zeros_like(conf_mat)
        for i, r in enumerate(conf_mat):
            conf_mat_norm[i] = r / np.sum(r)
        
        fig = plt.figure()
        fig.set_size_inches(36, 32)

        ax1 = fig.add_axes([0.1,0.1,0.75,0.85])
        cbar_ax = fig.add_axes([0.88,0.1,0.03,0.85])
        
        im1 = ax1.imshow(conf_mat_norm, cmap=cm.Blues)
        plt.colorbar(im1, cax=cbar_ax)
        ax1.set_xticks(np.arange(self.n_classes))
        ax1.set_yticks(np.arange(self.n_classes))
        ax1.set_xticklabels(self.classes, rotation='vertical')
        ax1.set_yticklabels(self.classes)
        ax1.tick_params(labelsize=24)
        cbar_ax.tick_params(labelsize=32)
        ax1.set_xlabel('Predicted class', fontsize=40)
        ax1.set_ylabel('True class', fontsize=40)
        ax1.set_title('Class confusion matrix', fontsize=56, va='bottom')
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                color = 'white' if conf_mat_norm[i,j] > 0.5 else 'black'
                label = '{:.3f}'.format(conf_mat_norm[i,j])+'\n%d' % conf_mat[i,j]
                ax1.text(j, i, label, ha='center', va='center', color=color, fontsize=20)
        
        # Statistics

        self.acc = np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat)
        self.prec = np.zeros(self.n_classes)
        self.rec = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            self.prec[i] = conf_mat[i,i] / np.sum(conf_mat[:,i])
            self.rec[i] = conf_mat[i,i] / np.sum(conf_mat[i,:])

        outfile = outdir+'metrics.csv'
        with open(outfile, 'w') as f:
            f.write('Accuracy:;%f\n\n' % self.acc)
            f.write('Class;Precision;Recall;Confusion matrix\n')
            for i in range(self.n_classes):
                f.write('%s;%f;%f;' % (self.classes[i], self.prec[i], self.rec[i]))
                for j in range(self.n_classes):
                    f.write(str(self.conf_mat[i,j]))
                    if j < self.n_classes - 1:
                        f.write(';')
                f.write('\n')
            f.write('%s;%f;%f\n\n' % ('Mean', np.mean(self.prec), np.mean(self.rec)))
                
        if verbose > 0: print('Statistical information saved to '+outfile)

        # Save figure
        outfile = outdir+'metrics.png'
        plt.savefig(outfile)
        if verbose > 0: print('Metrics plot saved to '+outfile)
        plt.close()

def plot_history(history, outdir='./'):

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    np.save(outdir+'loss_history.npy', loss)
    np.save(outdir+'val_loss_history.npy', val_loss)
    
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Loss', 'Validation loss'])

    ax = plt.gca()
    ylim = ax.get_ylim()
    ax.vlines(np.argmin(val_loss), ymin=ylim[0], ymax=ylim[1], colors='r', linestyles='dashed')

    outfile = outdir+'loss_history.png'
    plt.savefig(outfile)
    print('Loss history plot saved to '+outfile)
    plt.close()
