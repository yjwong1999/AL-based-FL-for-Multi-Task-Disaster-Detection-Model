import logging
import os
from typing import List

import numpy as np
import pandas as pd 
from math import cos, pi
import math

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

"""Turn of worning for TF"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #https://stackoverflow.com/a/42121886
import tensorflow as tf
# from tensorflow.keras.callbacks import ReduceLROnPlateau
    
from models import get_disaster_head

logger = logging.getLogger(__name__)


"""Datetime"""
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)


"""limit gpu growth"""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


"""model defintition"""
# get augmentor
def get_augmentor():
    augmentor = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1, 0.1),
    ])
    return augmentor
  
# get the backbone
def get_backbone(size, class_num, backbone_h5):
    # define input
    in_shape = (size, size, 3)
    input_images = tf.keras.layers.Input(shape=in_shape)
    x = input_images
    # load the backbone
    backbone = tf.keras.models.load_model(backbone_h5)
    backbone.trainable = False
    _, x, _ = backbone(x)
    # define the model
    model = tf.keras.models.Model(input_images, x, name='backbone')
    return model

  
# get head model
def create_model():
    x = np.ones((1, 26, 26, 512))
    x = tf.convert_to_tensor(x)
    disaster_head = get_disaster_head(x, class_num=7)
    return disaster_head

  
backbone = get_backbone(size=416, class_num=7, backbone_h5='backbone.h5')
disaster_predictor = create_model()
augmentor = get_augmentor()


"""path for all tsv files (ori/seed/pool)"""
# ori tsv path
data_root = '/home/tham/Documents/fyp_yijie/crisis_vision_benchmarks/'
annot_train_path = os.path.join(data_root, 'tasks/disaster_types/consolidated/consolidated_disaster_types_train_final.tsv')
annot_dev_path = os.path.join(data_root, 'tasks/disaster_types/consolidated/consolidated_disaster_types_dev_final.tsv')
annot_test_path = os.path.join(data_root, 'tasks/disaster_types/consolidated/consolidated_disaster_types_test_final.tsv')

# get all unique labels
df = pd.read_csv(annot_train_path, sep='\t')
labels = np.array(df['class_label'].tolist())
unique_label = np.unique(labels)


"""Learning Rate Schedule"""
# define a cosine decay
class MyCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
    
    def __call__(self, step):
        print('--------------------------------------------')
        print(step)
        step = tf.math.minimum(step, self.decay_steps)
        print(step.shape)
        cosine_decay = 0.5 * (1 + cos(pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.initial_learning_rate * decayed
      
      
"""Training Utilities"""
class TrainingUtils:
    def __init__(self, data_root, unique_label):
        self.data_root = tf.convert_to_tensor(data_root, tf.string)
        self.unique_label = unique_label
    
    def load_data(self, data):
        # split the data into x and y (image and label)
        image_path = data[0]
        label = data[1]
        # read the image from disk, decode it, resize it, and scale the
        # pixels intensities to the range [0, 1]
        image_path = tf.strings.join([self.data_root, image_path])
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (416, 416)) / 255.0
        # encode the label
        label = tf.argmax(label == self.unique_label)
        # return the image and the integer encoded label
        return (image, label)
    
      
    @tf.function
    def feature_extraction_augment(self, image, label):
        # perform random horizontal and vertical flips
        image = tf.image.random_flip_left_right(image)
        # brightness
        image = tf.image.random_brightness(image, 0.2)
        # contrast
        image = tf.image.random_contrast(image, 0.5, 2.0)
        # saturation
        image = tf.image.random_saturation(image, 0.80, 1.20) #ori is 0.75-1.25
        # rotation/translation/zoom
        image = tf.expand_dims(image, axis=0)
        image = augmentor(image)
        # feature extraction with augmentation
        tensor = backbone(image, training=False)[0]
        # return the tensor and the label
        return (tensor, label)
     
      
    @tf.function
    def feature_extraction(self, image, label):
        # feature extraction
        image = tf.expand_dims(image, axis=0)
        tensor = backbone(image, training=False)[0]
        # return the tensor and the label
        return (tensor, label)
    
      
    def onehot(self, tensor, label):
        label = tf.one_hot(label, 7)
        return (tensor, label)


# active learning
class ActiveLearning(TrainingUtils):
    def __init__(self, data_root, unique_label, rank, worldsize,
                 annot_train_path, annot_dev_path, annot_test_path,
                 seed_size=700, number_query=64, pool_size=None, 
                 al_strategy='margin'):
        """
        initialise the paths, and active learning setting
        """
        super().__init__(data_root, unique_label)
        # ori tsv path
        # data_root = '/home/tham/Documents/fyp_yijie/crisis_vision_benchmarks/'
        # annot_train_path = os.path.join(data_root, 'tasks/disaster_types/consolidated/consolidated_disaster_types_train_final.tsv')
        # annot_dev_path = os.path.join(data_root, 'tasks/disaster_types/consolidated/consolidated_disaster_types_dev_final.tsv')
        # annot_test_path = os.path.join(data_root, 'tasks/disaster_types/consolidated/consolidated_disaster_types_test_final.tsv')
        
        # SEED tsv path
        idx = annot_train_path.rindex('/')
        train_seed_path = os.path.join(annot_train_path[:idx], '{}_train_seed.tsv'.format(rank))
        
        # pool tsv path
        idx = annot_train_path.rindex('/')
        train_pool_path = os.path.join(annot_train_path[:idx], '{}_train_pool.tsv'.format(rank))
        
        # # get all unique labels
        # df = pd.read_csv(annot_train_path, sep='\t')
        # labels = np.array(df['class_label'].tolist())
        # unique_label = np.unique(labels)
        
        # assign the values to the class
        self.rank = rank
        self.worldsize = worldsize
        
        self.annot_train_path = annot_train_path
        self.annot_dev_path = annot_dev_path
        self.annot_test_path = annot_test_path

        self.train_seed_path = train_seed_path        
        self.train_pool_path = train_pool_path
        
        self.unique_label = unique_label
        
        self.SEED_SIZE = seed_size
        assert self.SEED_SIZE % 7 ==0, "'SEED_SIZE' must be divisible by 7"
        self.SEED_SIZE_PER_CLASS = self.SEED_SIZE // 7
        
        self.NUMBER_QUERY = number_query
        self.POOL_SIZE = pool_size
        self.AL_STRATEGY = al_strategy


    def query_selection(self, df, pool_df):
        print("quering...")
        """get the pool dataset"""
        # get the pool image paths and labels
        img_paths = np.array(pool_df['image_path'].tolist())
        labels = np.array(pool_df['class_label'].tolist())
    
        # tf dataset from the pooling data 
        data = list(zip(img_paths, labels))
        DS = tf.data.Dataset.from_tensor_slices(data)
        DS = (DS
        	.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
            .map(self.feature_extraction, num_parallel_calls=tf.data.AUTOTUNE)
        	.batch(32)
        	.prefetch(tf.data.AUTOTUNE)
        )
        
        """get the softmax tensors"""
        softmax_tensors = None
        for (tensors, labels) in DS:
            softmax_tensor = disaster_predictor(tensors, training=False)
            if softmax_tensors is None:
                softmax_tensors = softmax_tensor
            else:
                softmax_tensors = tf.concat([softmax_tensors, softmax_tensor], axis=0)
        softmax_tensors = np.array(softmax_tensors)
      
        """sampling based on strategy"""
        if self.AL_STRATEGY=="margin":
            # sort each of the softmax tensor
            softmax_tensors.sort(axis=1)
            
            # arg margin = P(x) of most probable - 2nd most probable
            margin = softmax_tensors[:, -1]-softmax_tensors[:, -2]
            
            # get the index to sort the softmax tensors
            indices = np.argsort(margin)
            
            # select the top N (number of query) of softmax tensors with smallest margin
            indices = list(indices[:self.NUMBER_QUERY])
            """
            # select top N/3 harderst, and N/3 moderate hard, and N/3 easiet
            indices_1 = list(indices[-1 * self.NUMBER_QUERY//3:])
            indices_2 = list(indices[len(indices)//2-self.NUMBER_QUERY//3:len(indices)//2])  
            indices_3 = list(indices[:self.NUMBER_QUERY//3])
            indices = indices_1 + indices_2 + indices_3
            """
            np.random.shuffle(indices)
            
            # add the relevant dataset from pool to train df
            sub_df = pool_df.iloc[indices]
            df = df.append(sub_df)
            pool_df = pool_df.drop(indices)
        
        print(df.shape, pool_df.shape)
        # return
        return (df, pool_df)
  
    
    def get_dataset(self, pool=True):
        """Image Paths and Labels"""  
        # decide which tsv to use for training
        if os.path.isfile(self.train_pool_path):
            # SEED data
            train_df = pd.read_csv(self.train_seed_path, sep='\t')
            
            # pool data
            train_pool_df = pd.read_csv(self.train_pool_path, sep='\t')
            
            # call active learning query_selection/sampling
            if pool:
                train_df, train_pool_df = self.query_selection(train_df, train_pool_df)
            
            # training image paths and labels
            train_df = train_df.sample(frac=1)
            train_img_paths = np.array(train_df['image_path'].tolist())#[SEED_SIZE:]
            train_labels = np.array(train_df['class_label'].tolist())#[SEED_SIZE:]
            
        else:
            # training data
            print('Load original seed')
            train_df = None
            df = pd.read_csv(self.annot_train_path, sep='\t')
            df = df[self.rank-1::self.worldsize] #select fractions of data for this client
            for label in self.unique_label:
                sub_df = df[df['class_label'] == label][:self.SEED_SIZE_PER_CLASS] #select top N for seed
                if train_df is None:
                    train_df = sub_df[:]
                else:
                    train_df = train_df.append(sub_df)
            train_df = train_df.sample(frac=1) # shuffle
            train_img_paths = np.array(train_df['image_path'].tolist())
            train_labels = np.array(train_df['class_label'].tolist())
            train_pool_df = df.drop(train_df.index)
            train_pool_df = train_pool_df[:self.POOL_SIZE] #select top N for pool only
    
            # no need active learning, since we just started sampling out the ori SEED
            pass
    
        # reset index for training and pool
        train_df.reset_index(drop=True, inplace=True)
        train_pool_df.reset_index(drop=True, inplace=True)
    
        # Replace pool/seed tsv
        train_df.to_csv(self.train_seed_path, sep="\t", index=False)
        train_pool_df.to_csv(self.train_pool_path, sep="\t", index=False)
            
        # dev data
        dev_df = pd.read_csv(self.annot_dev_path, sep='\t')
        dev_img_paths = np.array(dev_df['image_path'].tolist())
        dev_labels = np.array(dev_df['class_label'].tolist())
        
        # test data
        test_df = pd.read_csv(self.annot_test_path, sep='\t')
        test_img_paths = np.array(test_df['image_path'].tolist())
        test_labels = np.array(test_df['class_label'].tolist())
        
        """return image paths and labels"""
        return (train_img_paths, train_labels), (dev_img_paths, dev_labels) \
                  , (test_img_paths, test_labels)
                  

    def main(self):
        for epoch in range(math.ceil(2000/self.NUMBER_QUERY)):
            print(f"Active learning round {epoch+1}")
        
            """Load images and labels from dataset"""
            # get image paths and labels
            print("get dataset")
            train_dataset, dev_dataset, test_dataset = self.get_dataset()
            (train_img_paths, train_labels) = train_dataset
            (dev_img_paths, dev_labels) = dev_dataset
            (test_img_paths, test_labels) = test_dataset
            
            # training 
            training_data = list(zip(train_img_paths, train_labels))
            trainDS = tf.data.Dataset.from_tensor_slices(training_data)
            
            # dev (validation)
            data = list(zip(dev_img_paths, dev_labels))
            valDS = tf.data.Dataset.from_tensor_slices(data)
            
            # testing
            data = list(zip(test_img_paths, test_labels))
            testDS = tf.data.Dataset.from_tensor_slices(data)
            
            
            """Create the tensorflow dataset"""
            print("get tf dataset")
            # dont use cache to prevent ram overloaded
            trainDS = (trainDS
            	.shuffle(512) #.shuffle(len(training_data))
            	.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
                .map(self.feature_extraction_augment, num_parallel_calls=tf.data.AUTOTUNE)
            	.batch(32)
            	.prefetch(tf.data.AUTOTUNE)
            )
            
            valDS = (valDS
            	.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
                .map(self.feature_extraction, num_parallel_calls=tf.data.AUTOTUNE)
            	.batch(32)
            	.prefetch(tf.data.AUTOTUNE)
            )
            '''
            testDS = (testDS
            	.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
                .map(self.feature_extraction, num_parallel_calls=tf.data.AUTOTUNE)
            	.batch(32)
                .map(onehot, num_parallel_calls=tf.data.AUTOTUNE)
            	.prefetch(tf.data.AUTOTUNE)
            )
            '''
            
            """Training"""
            print("training now")
            if epoch == 0:
                # hyperparameter
                EPOCH = 5
                LR = 0.001 #0.0005
                
                def cosine_decay(epoch, lr):
                    initial_learning_rate = LR
                    decay_steps = EPOCH - 1
                    alpha = 1/10 #1/100
                    # lr will not be used
                    
                    step = min(epoch, decay_steps)
                    cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
                    decayed = (1 - alpha) * cosine_decay + alpha
                    return initial_learning_rate * decayed
                CALLBACKS = tf.keras.callbacks.LearningRateScheduler(cosine_decay)  
                # CALLBACKS = ReduceLROnPlateau(patience=3, verbose=1)
            else:
                # hyperparameter
                EPOCH = 1
                LR = 0.0001
                CALLBACKS = None     
                
            # optimizer
            opt = tf.keras.optimizers.Adam(
                learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                name='Adam'
            )
            '''
            if os.path.isfile('temporary.h5'):
                disaster_predictor = tf.keras.models.load_model('temporary.h5')
                print('loaded the model trained on original seed')
            '''
            disaster_predictor.compile(optimizer=opt,
                                       loss='sparse_categorical_crossentropy',
                                       metrics=['accuracy'])
            
            _ = disaster_predictor.fit(
                x = trainDS,
                validation_data=valDS,
                epochs = EPOCH,
                callbacks = CALLBACKS
            )
            

"""Mnist Shard Descriptor."""
class MnistShardDataset(ShardDataset):
    """Mnist Shard dataset class."""

    def __init__(self, x, y, data_type, rank=1, worldsize=1):
        """Initialise the dataset"""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.x = x[self.rank - 1::self.worldsize]
        self.y = y[self.rank - 1::self.worldsize]
        self.augment = data_type=="train" #if train then we augment

    def __getitem__(self, index: int):
        """Return an item by the index."""
        # get data
        image_paths, ori_labels = self.x[index], self.y[index]      
        # load image
        tensors = None
        for i, image_path in enumerate(image_paths):
            image_path = tf.strings.join([data_root, image_path])
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, (416, 416)) / 255.0
            image = tf.expand_dims(image, axis=0)
            # augment image
            if self.augment:
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, 0.2)
                image = tf.image.random_contrast(image, 0.5, 2.0)
                image = tf.image.random_saturation(image, 0.80, 1.20) #ori is 0.75-1.25
                image = augmentor(image)
            # feature extraction
            tensor = backbone(image, training=False)
            if tensors is None:
                tensors = tensor
            else:
                tensors = tf.concat([tensors, tensor], axis=0)

        # encode the label
        labels = None
        for i, ori_label in enumerate(ori_labels):
            label = tf.argmax(ori_label == unique_label)
            label = tf.expand_dims(label, axis=0)
            if labels is None:
                labels = label
            else:
                labels = tf.concat([labels, label], axis=0)

        # return the image and the integer encoded label
        #print(images.shape, labels.shape)
        #print(index)
        return tensors.numpy(), labels.numpy()

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)



class MnistShardDescriptor(ShardDescriptor):
    """Mnist Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize MnistShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        (train_img_paths, train_labels), (test_img_paths, test_labels) = self.active_learning(self.rank, self.worldsize)
        self.data_by_type = {
            'train': (train_img_paths, train_labels),
            'val': (test_img_paths, test_labels)
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return MnistShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['416', '416', '3']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Mnist dataset, shard number {self.rank}'
                f' out of {self.worldsize}')
      
    def active_learning(self, rank, worldsize):
        al = ActiveLearning(data_root, unique_label, rank, worldsize, \
                            annot_train_path, annot_dev_path, annot_test_path)
        
        if not os.path.isfile(al.train_pool_path):
            al.main()
        train_dataset, dev_dataset, test_dataset = al.get_dataset(pool=False)
        
        (train_img_paths, train_labels) = train_dataset
        (dev_img_paths, dev_labels) = dev_dataset
        (test_img_paths, test_labels) = test_dataset
        
        return (train_img_paths, train_labels), (test_img_paths, test_labels)
