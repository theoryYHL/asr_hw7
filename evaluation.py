import numpy as np
import tensorflow as tf
import struct
import os
import pickle
from model import MODEL
import nltk
import operator
from train import *


def exist_check(string):
    word = string.split(" ")[-1]
    english_vocal = set(w.lower() for w in nltk.corpus.words.words())
    if(word.lower() in english_vocal):
        return 0
    else:
        return -np.inf

def L_end(string):
    if(len(string)>0):
        return string[-1]
    else:
        return "<blank>"


def wer(ref, hyp, debug=False):
    r = ref.split()
    h = hyp.split()
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    DEL_PENALTY=1
    INS_PENALTY=1
    SUB_PENALTY=1

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i] + "\t" + "****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    return (numSub + numDel + numIns) / (float)(len(r))

def make_string(mainstring, id_list):
    output = [0]*len(mainstring)
    for i in range(len(mainstring)):
        output[i] = id_list[mainstring[i]]
    outstring = ''.join(output)
    return outstring


def prefix_beam_search(frame_prob,id_list):
    width = 20
    probability = np.reshape(frame_prob,(-1,30))
    probability = np.log(probability)
    T = probability.shape[0]
    prob_prev_blank={}
    prob_prev_nonblank = {}
    prob_next_blank={}
    prob_next_nonblank = {}
    A_prev = [""]
    prob_prev_blank[""] = 0.
    prob_prev_nonblank[""]=-np.inf
    for t in range(T):
        A_next = []
        prob_next_blank={}
        prob_next_nonblank = {}
        for prefix in A_prev:
            prev_log_sum = max(np.exp(prob_prev_blank[prefix]),np.exp(prob_prev_nonblank[prefix]))
            for c in range(30):
                # if(c==29):
                #     prob_next_blank[prefix] = probability[t,c]+prev_log_sum
                #     A_next.append(prefix)
                # else:
                #     prefix_ex = prefix+id_list[c]
                #     if(id_list[c]==L_end(prefix)):
                #         prob_next_nonblank[prefix_ex] = probability[t,c]+prob_prev_blank[prefix]
                #         prob_next_nonblank[prefix] = probability[t,c]+prob_prev_blank[prefix]
                #     elif(c==26):
                #         prob_next_nonblank[prefix_ex] = exist_check(prefix)+(probability[t,c]+prev_log_sum)
                #     else:
                #         prob_next_nonblank[prefix_ex] = probability[t, c] + prev_log_sum
                #     if(prefix_ex in A_prev):
                #         prob_next_blank[prefix_ex] = probability[t, 29] + prev_log_sum
                #         prob_next_nonblank[prefix_ex] = probability[t, c] +prob_prev_nonblank[prefix_ex]
                #     A_next.append(prefix_ex)
                if(c==29):
                    prob_next_blank[prefix] = probability[t,c]+prev_log_sum
                    A_next.append(prefix)
                else:
                    prefix_ex = prefix+id_list[c]
                    if(id_list[c]==L_end(prefix)):
                        prefix_ex = prefix
                        prob_next_nonblank[prefix] = probability[t,c]+prob_prev_blank[prefix]
                    elif(c==26):
                        prob_next_nonblank[prefix_ex] = exist_check(prefix)+(probability[t,c]+prev_log_sum)
                    else:
                        prob_next_nonblank[prefix_ex] = probability[t, c] + prev_log_sum
                    if(prefix_ex in A_prev):
                        prob_next_blank[prefix_ex] = probability[t, 29] + prev_log_sum
                        prob_next_nonblank[prefix_ex] = probability[t, c] +prob_prev_nonblank[prefix_ex]
                    A_next.append(prefix_ex)
        sum_dictionary ={}
        for new_prefix in A_next:
            if (new_prefix not in prob_next_blank):
                prob_next_blank[new_prefix]=-np.inf
            if (new_prefix not in prob_next_nonblank):
                prob_next_nonblank[new_prefix]=-np.inf
            sum_dictionary[new_prefix] = max(prob_next_blank[new_prefix], prob_next_nonblank[new_prefix])
        sorted_prob = sorted(sum_dictionary.items(), key=operator.itemgetter(1), reverse=True)[:width]
        A_prev = []
        for tuple in sorted_prob:
            A_prev.append(tuple[0])
        prob_prev_blank=prob_next_blank
        prob_prev_nonblank = prob_next_nonblank
        print (t)
    answer = sorted_prob[0][0]





if __name__ == "__main__":
    charsize = 30 # alpahbet 26 + " "  ","  "." 3 + <blank> 1
    batch_size = 1
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
        if (ctc_frame[i] == 0):
            zero_finder.append(i)
    zero_finder.reverse()
    for index in zero_finder:
        ctc_target.pop(index)
        train_data.pop(index)
        train_frame.pop(index)
        train_target.pop(index)

    data_dim = 123
    # input과 target label
    input_data = tf.placeholder(tf.float32, shape=[None, None, data_dim])  # [batch_size, frame_length, data_dim]
    input_indices = tf.placeholder(tf.int64, shape=[None, 2])
    input_values = tf.placeholder(tf.int32, shape=[None])
    input_shape = tf.placeholder(tf.int64, shape=[2])
    input_len = tf.placeholder(tf.int32, shape=[None, ], name='len')

    model = MODEL(charsize, frame_length, batch_size, num_layer=4, keep_prob=1.0)
    _,_,greedy,output = model(input_data, input_indices, input_values,
                                 input_shape, seq_len_=input_len, is_train=True)

    with tf.Session() as testing_sess:
        # parameter 초기화
        testing_sess.run(tf.global_variables_initializer())
        # parameter load
        model.load_params(testing_sess)
        # 트레이닝 데이터의 개수
        data_num = len(train_data)
        for i in range(100):
            # 모든 train data에 대해 결과를 살펴봄
            # batchsize = 1
            batch_X = np.asarray([train_data[i+1]])
            batch_len = np.asarray([train_frame[i+1]])
            frame_prob = testing_sess.run(output,
                feed_dict={input_data: batch_X, input_len: batch_len})
            rescoring = prefix_beam_search(frame_prob,id_list)
            #beam = testing_sess.run(greedy,
            #    feed_dict={input_data: batch_X, input_len: batch_len})

            #y_ = make_string(beam[0][0].values,id_list)
            y = make_string(train_target[i],id_list)
            nowwer = wer(y,rescoring)
            #totalWER.append(nowwer)
            print (i)