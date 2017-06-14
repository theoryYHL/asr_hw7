from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import struct
import os
import pickle
import random
from model import MODEL

# pnhtable파일을 읽는 함수
def read_charset(charsize):
    charset_lines = open("char_set.txt").read().splitlines()
    char_dictionary = {}
    id_list = [0] * charsize
    for line in charset_lines:
        a = line.split(" ")
        char_dictionary[a[1]] = int(a[0])
        id_list[int(a[0])] = a[1]
    id_list[-1] = "<blank>"
    char_dictionary["<blank>"]=charsize-1
    id_list[26] = " "
    char_dictionary[" "] = 26
    char_dictionary.pop("")
    return char_dictionary, id_list


# feat파일 하나를 읽는 함수
def featread(file_path):
    feat = open(file_path,"rb")
    framenum = int.from_bytes(feat.read(4),byteorder='big')
    # print(framenum)
    sample_period = int.from_bytes(feat.read(4), byteorder='big')
    # print(sample_period)
    sample_size = int.from_bytes(feat.read(2), byteorder='big')
    # print(sample_size)
    parameter_type = int.from_bytes(feat.read(2), byteorder='big')
    # print(parameter_type)
    data = [[] for i in range(framenum)]
    for frameorder in range(framenum):
        for dim in range(123):
            floatval = struct.unpack(">f",feat.read(4))[0]
            data[frameorder].append(floatval)
    return data

# 모든 feat 파일을 읽는 함수
def allfeatread():
    data_path_list = open("train_wsj0.list").read().splitlines()
    data_num = len(data_path_list)
    train_data = [0] * data_num
    for i,path in enumerate(data_path_list):
        data = featread(path)
        train_data[i] = data
        print(path)
    return train_data

# phn 파일을 읽는 함수
def phnread(file_path):
    phn_lines = open(file_path).read().splitlines()
    start = []
    end = []
    phoneme = []
    for line in phn_lines:
        value = line.split()
        start.append(int(value[0]))
        end.append(int(value[1]))
        phoneme.append(value[2])
    return start, end, phoneme

# phn 파일 내용을 통해 몇 번째 frame이 어떤 phoneme인지를 나타내는 list를 만드느 함수
def make_frame(starts, ends, phonemes,framelength):
    phoneme_frame = []
    frame_middle = 200 #25ms의 중간 = 400/2
    phoneme_order = 0
    phoneme_size = len(phonemes)
    for i in range(framelength):
        while(phoneme_order<phoneme_size-1 and ends[phoneme_order]<frame_middle):
            phoneme_order+=1
        phoneme_frame.append(phonemes[phoneme_order])
        frame_middle+=160 # 10ms씩 미룸
    return phoneme_frame

# 각 frame의 phoneme정보를 이용해 이를 one_hot encoding해주는 함수
def make_int(chartarget, char_dictionary):
    outputlist = [0]*len(chartarget)
    for i in range(len(chartarget)):
        outputlist[i] = char_dictionary[chartarget[i]]
    return outputlist

# 모든 phn file을 이용해서 one_hot 인코딩을 해주는 함수
def alltargetread(frame_lengths, char_dictionary):
    trans_lines = open("train_wsj0.trans").read().splitlines()
    data_path_list = open("train_wsj0.list").read().splitlines()
    target_num = len(data_path_list)
    train_target = [0] * target_num
    target_dictionary = {}
    for line in trans_lines:
        chartarget = line[9:]
        inttarget = make_int(chartarget, char_dictionary)
        id = line[:8]
        target_dictionary[id] = inttarget
        print (id)
    for i,path in enumerate(data_path_list):
        id = path[19:27]
        train_target[i] = target_dictionary[id]
        print(path)
    return train_target

def get_batch():
    pass

def sparse_tensor_materials(x):
  """Create a very simple SparseTensor with dimensions (batch, time).
  Args:
    x: a list of lists of type int
  Returns:
    x_ix and x_val, the indices and values of the SparseTensor<2>.
  """
  x_ix = []
  x_val = []
  for batch_i, batch in enumerate(x):
    for time, val in enumerate(batch):
      x_ix.append([batch_i, time])
      x_val.append(val)
  x_shape = [len(x), np.asarray(x_ix).max(0)[1]+1]

  return x_ix, x_val, x_shape

def ctc_input(train_target, char_dictionary):
    ctc_labels = []
    for single_target in train_target:
        single_labels = []
        for label in single_target:
            single_labels.append(label)
        ctc_labels.append(single_labels)
    return ctc_labels
#######

def read_ctc_output(greedy, id_list):
    ctc_labels = []
    for single_target in train_target:
        single_labels = []
        for label in single_target:
            single_labels.append(29)
            single_labels.append(label)
        single_labels.append(29)
        ctc_labels.append(single_labels)
    return ctc_labels

if __name__ == "__main__":
    charsize = 30 # alpahbet 26 + " "  ","  "." 3 + <blank> 1
    batch_size = 8
    frame_length = 4000

    # char_dictionary: 각 char이 몇 번째인지. char_dictionary["A"] = 0
    # id_list: 몇 번째 char이 무엇인지. id_list[0] = "A"
    char_dictionary, id_list = read_charset(charsize)

    # train_data에 모든 feat 파일을 읽어온다
    # 속도 향상을 위해 traindata_pickle 파일에 저장하고 불러온다
    if os.path.isfile("trainwsj0_pickle"):
        f_data = open("trainwsj0_pickle","rb")
        train_data = pickle.load(f_data)
        f_data.close()
        print("read train pickle file done")
    else:
        train_data= allfeatread()
        f_data = open("trainwsj0_pickle","wb")
        pickle.dump(train_data,f_data)
        f_data.close()
        print("write train pickle file done")

    # 각 train 데이터의 frame수를 저장
    train_frame = [len(x) for x in train_data]

    # train_target에 phn 파일을 가공해서 넣어준다
    # phn파일을 이용해서 one_hot이 되어있다
    # -> one_hot을 사용하지 않고 싶으시면 위에 정의한 함수들을 이용해서 처리를 하시는게 좋을것 같아요
    if os.path.isfile("targetwsj0_pickle"):
        f_target = open("targetwsj0_pickle","rb")
        train_target = pickle.load(f_target)
        f_target.close()
        print("read train target pickle file done")
    else:
        train_target = alltargetread(train_frame, char_dictionary)
        f_target = open("targetwsj0_pickle","wb")
        pickle.dump(train_target,f_target)
        f_target.close()

    ctc_target = ctc_input(train_target,char_dictionary)
    ctc_frame = [len(x) for x in ctc_target]
    zero_finder = []
    for i in range(len(ctc_frame)):
        if(ctc_frame[i]==0):
            zero_finder.append(i)
    zero_finder.reverse()
    for index in zero_finder:
        ctc_target.pop(index)
        train_data.pop(index)
        train_frame.pop(index)
    batch_indices, batch_values, batch_shape = sparse_tensor_materials(ctc_target[0:2])

    data_dim = 123
    # input과 target label
    input_data = tf.placeholder(tf.float32, shape=[None, None, data_dim])  # [batch_size, frame_length, data_dim]
    input_indices = tf.placeholder(tf.int64,shape=[None,2])
    input_values = tf.placeholder(tf.int32, shape=[None])
    input_shape = tf.placeholder(tf.int64, shape=[2])
    input_len = tf.placeholder(tf.int32, shape=[None, ], name='len')

    model = MODEL(charsize, frame_length, batch_size, num_layer=4,keep_prob=0.5)
    # predicted_y => [batch_size, frame_length,phoneme_voca]
    loss, labels, greedy = model(input_data, input_indices, input_values,
                                                    input_shape, seq_len_=input_len, is_train=True)
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_y, logits=predicted_y))
    # cross_entropy를 최소화 하는 방향으로
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.Session() as training_sess:
        print ("model making done")
        # parameter 초기화
        training_sess.run(tf.global_variables_initializer())
        # 트레이닝 데이터의 개수
        data_num = len(train_data)
        # 일단 모든 트레이닝 데이터에 대해 한 번씩만 학습을 해봄
        # batch 는 현재 1
        s = 0
        for i in range(10000):
            if batch_size is not 1:
                s += batch_size
                e = s + batch_size
                if (e >= data_num):
                    s = 0
                    e = batch_size
                batch_len = np.asarray(train_frame[s:e])
                print(batch_len)

                sparse_tensor_materials(train_data[s:e])
                max_len = batch_len.max()
                tmp_X = []
                for x in train_data[s:e]:
                    x.extend([[0] * data_dim] * (max_len - len(x)))
                    tmp_X.append(x)
                batch_X = np.asarray(tmp_X)
                batch_indices, batch_values,batch_shape = sparse_tensor_materials(ctc_target[s:e])

            else:
                batch_X = np.asarray([train_data[i%2]])
                batch_len = np.asarray([train_frame[i%2]])
                batch_indices, batch_values, batch_shape = sparse_tensor_materials([ctc_target[i%2]])
            
            np.asarray(batch_X)
            np.asarray(batch_indices)
            np.asarray(batch_values)
            np.asarray(batch_shape)
            np.asarray(batch_len)
            _, _loss, _y, _labels = training_sess.run([train_step, loss, greedy, labels],
                feed_dict={input_data: batch_X, input_indices: batch_indices, input_values: batch_values,
                input_shape:batch_shape , input_len: batch_len})

            print(i)
            print("loss: ",_loss)
            print("greedy: ",_y)
        # parameter 저장. ./params_MODEL에 저장됨
        model.save_params(training_sess)

