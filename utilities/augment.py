from .libraries import *
from .tfrecords import *

def standardize_image(x,y,mean,std):
  normalize = lambda x: ((x - mean) / std).astype('float16')
  return (normalize(x['image']), y)

def normalize_image(x,y):
  return (x['image']/255, y)


def data_aug(ds,class_names,class_weight,augment_weight,out_file):
  def random_crop(x,y,params): 
    return (tf.image.random_crop(x, params['output_dim']),y)
  def flip_LR(x,y,params):
    return (tf.image.flip_left_right(x),y)


  fn_map={'flip_LR':flip_LR,'random_crop':random_crop}
  First=True
  ctr=0
  for i in ds:
    aug_choice=(np.random.randint(1,101,len(class_weight)*len(augment_weight))/100).reshape(len(class_weight),len(augment_weight))
    class_choice=(np.random.randint(1,101,len(class_weight))/100)
    ctr=ctr+1
    x=i[0].numpy()
    y=i[1].numpy()

    a_ch=list(aug_choice[y])
    cw=class_weight[class_names[y]]
    aw=[i['weight'] for i in augment_weight.values()]

    f_list=[list(augment_weight.keys())[i] for i in range(len(a_ch)) if class_choice[i]<=cw and a_ch[i]<=aw[i]]
    # if f_list!=[]:
    #   print('\n')
    for f in f_list:
      f1=fn_map[f]
      k=f1(x,y,augment_weight[f]['params'])
      # print(class_choice[i],a_ch,ctr,f.__name__)
      print('Augmenting Image ',ctr,' with ',f)
      x1=(k[0].numpy()).reshape(1,k[0].shape[0],k[0].shape[1],k[0].shape[2])
      y1=np.array([k[1]])
      if First:
        x_augmented=x1
        y_augmented=y1
        First=False
      else:
        x_augmented=np.vstack((x_augmented,x1))
        y_augmented=np.vstack((y_augmented,y1))
      # print(x_augmented.shape)
      # print(y_augmented.shape)
    # print("***************************************************************************************************")
  create_tfrecords(out_file,(x_augmented, y_augmented))

@timer
# @checkTFversion
def augment_images(path,class_weight,augment_weight,parallelize):
  class_names = [line.rstrip('\n') for line in open(os.path.join(path, "classes.txt"))]
  in_file=os.path.join(path, "train.tfrecords")
  ds=tf.data.TFRecordDataset(tf.data.Dataset.list_files(in_file))
  ds=ds.map(lambda record: parser(record), num_parallel_calls=parallelize)
  ds=ds.map(normalize_image, num_parallel_calls=parallelize)
  out_file=os.path.join(path, "train_aug.tfrecords")
  data_aug(ds,class_names,class_weight,augment_weight,out_file)
