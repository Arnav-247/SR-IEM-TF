# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import numpy as np


class Model:
    def __init__(self, ycap,loss, u, input_seq, C_mask, neg, is_training, yoh):
        self.ycap = ycap
        self.u = u
        self.input_seq = input_seq
        self.C_mask = C_mask
        self.neg = neg
        self.is_training = is_training
        self.yoh = yoh
        self.loss = loss

    def predict_sess(self, sess, u, seq, item_idx, yoh):
        return sess.run([self.ycap ,self.loss],
                        {self.u: u, self.input_seq: seq,
                         self.C_mask: np.zeros((len(u), args.maxlen), dtype=np.float32), self.neg: item_idx,
                         self.is_training: False, self.yoh: yoh})


if __name__ == '__main__':
    from collections import defaultdict
    import tensorflow as tf
    import args
    from sampler import WarpSampler
    from util import batch_evaluate
    import time


    # tf.enable_eager_execution()
    # sess = tf.InteractiveSession()

    def data_partition(d_dir, fname):
        max_uid = 0
        max_iid = 0
        User = defaultdict(list)
        user_train = {}
        user_valid = {}
        user_test = {}
        # assume user/item index starting from 1
        f = open(f'data/{d_dir}/{fname}.csv', 'r')
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            temp = line.rstrip().split(',')
            u = int(temp[0])
            i = int(temp[1])
            max_uid = max(u, max_uid)
            max_iid = max(i, max_iid)
            User[u].append(i)

        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
        return [user_train, user_valid, user_test, max_uid, max_iid]


    dataset = data_partition(args.data_dir, args.dataset_test)
    user_train, user_valid, user_test, max_uid, max_iid = dataset
    print('max uid', len(user_train))
    num_batch = len(user_train) // args.batch_size
    print('num_batcg: ', num_batch)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    # f = open(os.path.join(args.train_dir, 'log.txt'), 'w')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    sampler = WarpSampler(dataset)

    sess.__enter__()
    saver = tf.train.import_meta_graph('out_models/ml-25m-3-regularized/my-model')
    saver.restore(sess, tf.train.latest_checkpoint('out_models/ml-25m-3-regularized'))

    graph = tf.get_default_graph()
    u = graph.get_tensor_by_name("u:0")
    input_seq = graph.get_tensor_by_name("input_seq:0")
    C_mask = graph.get_tensor_by_name("C_mask:0")
    neg = graph.get_tensor_by_name("neg:0")
    yoh = graph.get_tensor_by_name("yoh:0")
    ycap = graph.get_tensor_by_name("SR-IEM/ycap_n:0")
    loss = graph.get_tensor_by_name("loss:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    # sess.run(op_to_restore, feed_dict)

    T = 0.0
    t0 = time.time()
    model = Model(ycap,loss, u, input_seq, C_mask, neg, is_training, yoh)
    t_test = batch_evaluate(model, dataset, sess, False)
    print(
        'time: %f(s), test (NDCG@20: %.4f, HIT@20: %.4f, MRR@20: %.4f, loss:%.4f)' % (
            T, t_test[0], t_test[1], t_test[2], t_test[3]))

    # f.close()
    # sampler.close()
    print("Done")


