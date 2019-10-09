from .libraries import *


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))





@timer
# @checkTFversion
def convert_to_tfrecords_float(filename, data_set, label):
  """Converts a dataset to tfrecords."""
  n=len(data_set)
  if data_set.shape[0] != n:
    raise ValueError('Images size %d does not match label size %d.' %(data_set.shape[0], n))
  rows = data_set.shape[1]
  cols = data_set.shape[2]
  depth = data_set.shape[3]

  # print('Writing', 'temp')
  with tf.io.TFRecordWriter('temp.tfrecords') as writer:
    for index in range(n):
      image_raw = data_set[index].tostring()
      l=int(label[index])
      example = tf.train.Example(
          features=tf.train.Features(
              feature={'height': _float_feature(rows),'width': _float_feature(cols),'depth': _float_feature(depth),'label': _int64_feature(l),'image': _bytes_feature(image_raw)}
          )
      )
      writer.write(example.SerializeToString())
  print(filename)
  copyfile('temp.tfrecords', filename)
  os.remove('temp.tfrecords')

@timer
# @checkTFversion
def parser_float(record):
    keys_to_features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(record, keys_to_features)
    parsed_image = tf.io.decode_raw(parsed["image"], tf.float64)
    parsed_image=  tf.reshape(parsed_image, [ parsed["height"], parsed["width"], parsed["depth"]])    
    label = tf.cast(parsed["label"], tf.int32)
    return {'image': parsed_image}, label
  
@timer
# @checkTFversion
def create_tfrecords_float(path,data):
  if len(data)!=2 or len(data[0].shape)!=4 or len(data[1].shape)!=2:
    raise AssertionError("Expecting tuple(np.ndarray(number of images, rows, cols, depth),np.ndarray(number of images,1))")
  print("converting to tfrecords")  
  print("number of images:", data[0].shape[0])
  print("dimension of images: (", data[0].shape[1], data[0].shape[2], data[0].shape[3], ")")
  convert_to_tfrecords_float(path, data[0], data[1])
  return 

@timer
# @checkTFversion
def convert_to_tfrecords(filename, data_set, label):
  """Converts a dataset to tfrecords."""
  n=len(data_set)
  if data_set.shape[0] != n:
    raise ValueError('Images size %d does not match label size %d.' %(data_set.shape[0], n))
  rows = data_set.shape[1]
  cols = data_set.shape[2]
  depth = data_set.shape[3]

  # print('Writing', 'temp')
  with tf.io.TFRecordWriter('temp.tfrecords') as writer:
    for index in range(n):
      image_raw = data_set[index].tostring()
      l=int(label[index])
      example = tf.train.Example(
          features=tf.train.Features(
              feature={'height': _int64_feature(rows),'width': _int64_feature(cols),'depth': _int64_feature(depth),'label': _int64_feature(l),'image': _bytes_feature(image_raw)}
          )
      )
      writer.write(example.SerializeToString())
  print(filename)
  copyfile('temp.tfrecords', filename)
  os.remove('temp.tfrecords')


  # !cp  temp.tfrecords $filename
  # !rm temp.tfrecords
@timer
# @checkTFversion
def parser(record):
    keys_to_features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(record, keys_to_features)
    parsed_image = tf.io.decode_raw(parsed["image"], tf.uint8)
    parsed_image = tf.cast(parsed_image, tf.float32)
    parsed_image=  tf.reshape(parsed_image, [ parsed["height"], parsed["width"], parsed["depth"]])    
    label = tf.cast(parsed["label"], tf.int32)
    return {'image': parsed_image}, label

@timer
# @checkTFversion
def create_tfrecords(path,data):
  # if type(data) is str:
  #   builder=tfds.builder(data)
  #   print(builder.info)
  #   builder.download_and_prepare()
  #   source=os.path.join('/root/tensorflow_datasets/'+data)
  #   # !cp -r  $source $path
  #   copyfile($source, $path)
  #   # datasets = mnist_builder.as_dataset()
  #   # train_dataset, test_dataset = datasets["train"], datasets["test"]
  # if type(data) is tuple:  
  if len(data)!=2 or len(data[0].shape)!=4 or len(data[1].shape)!=2:
    raise AssertionError("Expecting tuple(np.ndarray(number of images, rows, cols, depth),np.ndarray(number of images,1))")
  print("converting to tfrecords")  
  print("number of images:", data[0].shape[0])
  print("dimension of images: (", data[0].shape[1], data[0].shape[2], data[0].shape[3], ")")
  convert_to_tfrecords(path, data[0], data[1])
  return 

@timer
# @checkTFversion
def create_classfile(path,class_names):
  with open(path, 'w') as f:
    f.writelines([i+'\n' for i in class_names])
