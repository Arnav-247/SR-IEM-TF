# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import sys

import numpy as np

if __name__ == '__main__':
    from collections import defaultdict
    import tensorflow as tf
    import args
    import os
    from sampler import WarpSampler
    from util import batch_evaluate
    from model import Model
    from tqdm import tqdm
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


    dataset = data_partition(args.data_dir, args.dataset)
    user_train, user_valid, user_test, max_uid, max_iid = dataset
    print('max uid', len(user_train))
    num_batch = len(user_train) // args.batch_size
    print('num_batcg: ', num_batch)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.train_dir, 'log.txt'), 'w')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    sampler = WarpSampler(dataset)

    sess.__enter__()
    model = Model(max_uid, max_iid, args)
    sess.run(tf.initialize_all_variables())

    T = 0.0
    t0 = time.time()
    saver = tf.train.Saver()

    for epoch in tqdm(range(1, args.num_epochs + 1), total=args.num_epochs, ncols=70, unit='e', desc="Epoch",
                      file=sys.stdout, position=0):

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        # initial_learning_rate,
        # decay_steps=3,
        # decay_rate=0.1,
        # staircase=True)
        avg_loss = 0
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b', desc='Batch',
                         file=sys.stdout, position=1):
            u, seq, neg, yoh = sampler.next_batch()
            loss, _ = sess.run([model.loss, model.train_op],
                               {model.u: u, model.C_mask: np.zeros((len(u), args.maxlen), dtype=np.float32),
                                model.input_seq: seq, model.neg: neg,
                                model.yoh: yoh,
                                model.is_training: True})
            avg_loss += loss
        avg_loss /= (num_batch)
        print("avg_loss", avg_loss)
        sampler.reset()
        if epoch % 5 == 0:
            t1 = time.time() - t0
            T += t1
            t_test = batch_evaluate(model, dataset, sess, False)
            # t_test = evaluate(model, dataset, sess)
            # t_valid = evaluate_valid(model, dataset, sess)
            t_valid = batch_evaluate(model, dataset, sess, True)
            print("")
            print(
                'epoch:%d, time: %f(s), valid (NDCG@20: %.4f, HIT@20: %.4f, MRR@20: %.4f, loss: %.4f), test (NDCG@20: %.4f, HIT@20: %.4f, MRR@20: %.4f, loss: %.4f)' % (
                    epoch, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_test[0], t_test[1], t_test[2],
                    t_test[3]))

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()

            t0 = time.time()
        if epoch % 5 == 0:
            saver.export_meta_graph(filename='out_models/my-model', as_text=True)
            saver.save(sess, 'out_models/my-model', global_step=epoch, write_meta_graph=False)

f.close()
# sampler.close()
print("Done")

# # Some tensor we want to print the value of
# t1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
# # seq = tf.reshape(t1, [2,2,4])
# # b = [1,2,3,4]
# # b = tf.reshape(b, [2,2])

# # b_exp = tf.tile(tf.expand_dims(b, -1),[1,1,4])
# # Add print operation
# # tf.print(b.numpy())
# # tf.print(b_exp.numpy())
# # tf.print(seq.numpy())
# # tf.print(seq[:,-1,:].numpy())


# # params = tf.reshape(t1, [-1,2]) # i, D
# # ids = [[1,2,3], [4,5,6], [7,8,9]] # N, I
# # ids = tf.reshape(ids, [3*3]) # N*I
# # p_emb = tf.nn.embedding_lookup(params, ids)
# # p_emb_res = tf.reshape(p_emb, [3,3,2]) # N, I, D
# # p_emb_res_trans = tf.transpose(p_emb_res, [0,2,1])

# # tf.print(ids.numpy())
# # tf.print(params.numpy())
# # tf.print()
# # tf.print()
# # tf.print(p_emb.numpy())
# # tf.print()
# # tf.print()
# # tf.print(p_emb_res.numpy())
# # tf.print(p_emb_res_trans.numpy())

# yoh = np.zeros([4,5], dtype=np.uint8)
# idx = [2,3,4, 1]

# for i in range(len(idx)):
#     yoh[i,idx[i]] = 1

# yoh = tf.convert_to_tensor(yoh)
# tf.print(yoh.numpy())
