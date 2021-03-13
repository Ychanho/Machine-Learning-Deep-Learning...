import numpy as np
import random
# 데이터의 순서 섞는 것
def shuffle_data(x_train_val, y_train_val, x_test, y_test):
    shuffle_indices = np.random.permutation(np.arange(len(y_train_val)))  # [1,2,3,4...len] ->#[4509,11,1356,4...len]
    shuffled_x = np.asarray(x_train_val[shuffle_indices]) # input x를 shuffle 한 후 list를 numpy 자료형으로
    shuffled_y = y_train_val[shuffle_indices] # output (정답) y를 shuffle
    val_sample_index = -1 * int(0.1 * float(len(y_train_val)))  # training data 50000개 중 validation data 개수 결정, 10%, -5000
    x_train, x_val = shuffled_x[:val_sample_index], shuffled_x[val_sample_index:] # training과 validation data 분배
    y_train, y_val = shuffled_y[:val_sample_index], shuffled_y[val_sample_index:] # training과 validation data 분배
    x_test = np.asarray(x_test) #list를 numpy 자료형으로

    y_train_one_hot = np.eye(10)[y_train]  # [9, 8, 0] -> [[0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0]
    y_train_one_hot = np.squeeze(y_train_one_hot, axis=1)  # (45000, 10)
    y_test_one_hot = np.eye(10)[y_test]
    y_test_one_hot = np.squeeze(y_test_one_hot, axis=1)
    y_val_one_hot = np.eye(10)[y_val]
    y_val_one_hot = np.squeeze(y_val_one_hot, axis=1)
    return x_train, y_train_one_hot, x_test, y_test_one_hot, x_val, y_val_one_hot

def batch_iter(x, y , batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    num_batches_per_epoch = int((len(x) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        shuffle_indices = np.random.permutation(np.arange(len(x))) # epoch 마다 shuffling
        shuffled_x = x[shuffle_indices]
        shuffled_y = y[shuffle_indices]
        # data 에서 batch 크기만큼 데이터 선별
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(y))
            # 만들어 진 batch를 return하기 전에 data augment
            yield list(zip(data_augmentation(shuffled_x[start_index:end_index], 4), shuffled_y[start_index:end_index]))

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
