# 参考tensorflow

import gzip
import numpy

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def dense_to_one_hot(labels_dense, num_classes):
  num_labels = labels_dense.shape[0]
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot[range(num_labels), labels_dense] = 1
  return labels_one_hot

def extract_labels(f, one_hot=False, num_classes=10):
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

class DataSet(object):

  def __init__(self,
               images,
               labels,
               one_hot=False,
               dtype=numpy.float32,
               reshape=True):
    self._num_examples = images.shape[0]
    if reshape:
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
    if dtype == numpy.float32:
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self.images = images
    self.labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0


  def next_batch(self, batch_size, shuffle=True):

    start = self._index_in_epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
      self.images = self._images
      self.labels = self._labels

    if start + batch_size > self._num_examples:

      self._epochs_completed += 1

      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]

      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
        self.images = self._images
        self.labels = self._labels

      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

class read_data_sets():
    def __init__(self, train_dir,
                   one_hot=False,
                   dtype=numpy.float32,
                   reshape=True,
                   validation_size=5000):

      TRAIN_IMAGES = train_dir + 'train-images-idx3-ubyte.gz'
      TRAIN_LABELS = train_dir + 'train-labels-idx1-ubyte.gz'
      TEST_IMAGES = train_dir + 't10k-images-idx3-ubyte.gz'
      TEST_LABELS = train_dir + 't10k-labels-idx1-ubyte.gz'

      with open(TRAIN_IMAGES, 'rb') as f:
        train_images = extract_images(f)

      with open(TRAIN_LABELS, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

      with open(TEST_IMAGES, 'rb') as f:
        test_images = extract_images(f)

      with open(TEST_LABELS, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

      validation_images = train_images[:validation_size]
      validation_labels = train_labels[:validation_size]
      train_images = train_images[validation_size:]
      train_labels = train_labels[validation_size:]

      self.train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
      self.val = DataSet(validation_images,validation_labels,dtype=dtype,reshape=reshape)
      self.test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)



