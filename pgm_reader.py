import os
import cv2
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base


def read_image(file_path):
    image = cv2.imread(file_path, 0)  # grayscale
    return image


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = len(labels_dense)
    labels_one_hot = np.zeros((num_labels, num_classes))
    label_list = list(set(labels_dense))
    for i, label in enumerate(labels_dense):
        labels_one_hot[i][label_list.index(label)] = 1
    return labels_one_hot

"""
def extract_data(path):
    images, labels = traverse_dir(path)
    labels_one_hot = dense_to_one_hot(labels, len(set(labels)))
    return images, labels_one_hot
"""

def extract_data(path):
    import scipy.io
    mat = scipy.io.loadmat("olivettifaces.mat")
    images = mat['faces']
    labels = [i//20 for i in range(images.shape[1])]
    labels_one_hot = dense_to_one_hot(labels, len(set(labels)))
    return images.T, labels_one_hot

images = []
labels = []
def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.pgm'):
                image = read_image(abs_path)
                h, w = image.shape[:2]
                image = cv2.resize(image, (h//4, w//4))
                images.append(image)
                labels.append(os.path.basename(path))
    return np.array(images, dtype=np.uint8), labels


class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                #assert images.shape[3] == 1
                images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(dtype=dtypes.float32, reshape=False):
    images, labels = extract_data('./data')
    num_images = images.shape[0]
    TRAIN_SIZE = int(num_images * 0.8)
    VALIDATION_SIZE = int(num_images * 0.1)
    train_images = images[:TRAIN_SIZE]
    train_labels = labels[:TRAIN_SIZE]
    validation_images = images[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]
    validation_labels = labels[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]
    test_images = images[TRAIN_SIZE+VALIDATION_SIZE:]
    test_labels = labels[TRAIN_SIZE+VALIDATION_SIZE:]
    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape)
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

    return base.Datasets(train=train, validation=validation, test=test)

if __name__ == '__main__':
    #images, labels = traverse_dir("./data")
    #images, labels = extract_data('./data')
    read_data_sets()
    #print(images.shape)
    #print(labels.shape)