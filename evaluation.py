import numpy as np
import tensorflow as tf
import struct
import os
import pickle
from model import MODEL
from train import *

if __name__ == "__main__":
    phoneme_voca = 61 # phoneme의 개수
    batch_size = 1
    frame_length = 1000

    # phone_dictionary: 각 phoneme이 몇 번째인지. phone_dictionary["aa"] = 0
    # id_list: 몇 번째 phoneme이 무엇인지. id_list[0] = "aa"
    # phonemegroup_dictionary: 각 phoneme이 어느 group에 속하는지. phonemegroup_dictionary["ax"] = "ah"
    phoneme_dictionary, id_list = read_phntable(phoneme_voca)
    phonemegroup_dictionary = read_phngroup()

    # train_data내에 모든 feat 파일을 읽어온다
    # 속도 향상을 위해 traindata_pickle 파일에 저장하고 불러온다
    if os.path.isfile("traindata_pickle"):
        f_data = open("traindata_pickle","rb")
        train_data = pickle.load(f_data)
        f_data.close()
        print("read train pickle file done")
    else:
        train_data= allfeatread()
        f_data = open("traindata_pickle","wb")
        pickle.dump(train_data,f_data)
        f_data.close()

    # 각 train 데이터의 frame수를 저장
    train_frame = [len(x) for x in train_data]

    # train_target에 phn 파일을 가공해서 넣어준다
    # phn파일을 이용해서 one_hot이 되어있다
    # -> one_hot을 사용하지 않고 싶으시면 위에 정의한 함수들을 이용해서 처리를 하시는게 좋을것 같아요
    if os.path.isfile("traintarget_pickle"):
        f_target = open("traintarget_pickle","rb")
        train_target = pickle.load(f_target)
        f_target.close()
        print("read train target pickle file done")
    else:
        train_target = allphnread(train_frame,phoneme_dictionary)
        f_target = open("traintarget_pickle","wb")
        pickle.dump(train_target,f_target)
        f_target.close()


    input_data = tf.placeholder(tf.float32, shape=[None, None,123])  # [batch_size, frame_length, data_dim]
    target_y = tf.placeholder(tf.float32, shape=[None, None, 61]) # [batch_size, frame_length,phoneme_voca]

    model = MODEL(charsize, frame_length, batch_size, num_layer=4, keep_prob=1.0)
    initial_states = model.zero_state()
    predicted_y, final_state = model(input_data, initial_states)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predicted_y, 2), tf.argmax(target_y, 2)), tf.float32))

    with tf.Session() as testing_sess:
        # parameter 초기화
        testing_sess.run(tf.global_variables_initializer())
        # parameter load
        model.load_params(testing_sess)
        # 트레이닝 데이터의 개수
        data_num = len(train_data)
        for i in range(1):
            # 모든 train data에 대해 결과를 살펴봄
            # batchsize = 1
            batch_X = np.array([train_data[i]])
            batch_y = np.array([train_target[i]])
            output, acc = testing_sess.run([predicted_y,accuracy],feed_dict={input_data: batch_X, target_y: batch_y})
            sequence = []
            for batch in output:
                for frame in batch:
                    phoneme_order = np.argmax(frame)
                    phoneme_frame = id_list[phoneme_order]
                    sequence.append(phoneme_frame)
            print (i)
            print (sequence)
            print (acc)