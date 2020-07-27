import numpy as np
import random
import glob
import scipy
import os

def read_data_sets(train_files):
    data_paths = glob.glob(os.path.join(train_files, "**"), recursive=True)
    data_paths = [image_file for image_file in data_paths if "jpg" in image_file]
    samples = []
    for data_path in data_paths:
        img = scipy.misc.imread(data_path, mode='RGB')
        img = scipy.misc.imresize(img, [64, 64])
        img = img/127.5 - 1
        samples.append(img)
    print(samples[0])
    return np.asarray(samples)

def load_mnist(path):
    images_path = os.path.join(path)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape([-1, 28, 28, 1])
    images = images/127.5 - 1
    print(images[0])
    return images

def batch_iter(x, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    num_batches_per_epoch = int((len(x) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        #shuffle_indices = np.random.permutation(np.arange(len(x))) # epoch 마다 shuffling
        #shuffled_x = x[shuffle_indices]
        # data 에서 batch 크기만큼 데이터 선별
        for batch_num in range(num_batches_per_epoch): 
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(x))
            yield x[start_index:end_index]

def data_augmentation (x_batch, padding=None):
    for i in range(len(x_batch)):
        if bool(random.getrandbits(1)):
            x_batch[i] = np.fliplr(x_batch[i]) # matrix 좌우 반전

    oshape = np.shape(x_batch[0]) # 원본 이미지 shape

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding) # padding 했을 때 shape
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0)) # 축 별 padding 크기 (channel 축만 padding 제외)
    for i in range(len(x_batch)):
        new_batch.append(x_batch[i])
        if padding:
            new_batch[i] = np.lib.pad(x_batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - 32)
        nw = random.randint(0, oshape[1] - 32)
        new_batch[i] = new_batch[i][nh:nh + 32, nw:nw + 32] # padding 한 이미지 (40)에서 다시 원본 크기 (32) 만큼 이미지 선택
    return new_batch

