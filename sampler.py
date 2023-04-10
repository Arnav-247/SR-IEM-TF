import numpy as np
import args
import copy

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, cand_size, batch_offset_idx, SEED):
    def sample(user):

        # while True:
        #     user = np.random.randint(1, usernum + 1)
        #     if user in user_train:
        #         if len(user_train[user]) <= 1:
        #             continue
        #         else:
        #             break

        # while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([cand_size], dtype=np.int32)
        yoh = np.zeros([cand_size], dtype=np.uint8)  # N, I
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            # if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        for i in range(args.cand_size):
            neg[i] = random_neq(1, itemnum + 1, ts)
        gt_idx = np.random.randint(0, cand_size)
        neg[gt_idx] = user_train[user][-1]
        yoh[gt_idx] = 1
        return (user, seq, neg, yoh)

    np.random.seed(SEED)
    one_batch = []
    mkeys = list(user_train.keys())
    for i in range(batch_offset_idx, batch_offset_idx + args.batch_size):
        u = mkeys[i]
        if len(user_train[u]) < 1:
            continue
        one_batch.append(sample(u))
    return zip(*one_batch)
    #
    # for i in range(batch_size):
    #     one_batch.append(sample())
    # return zip(*one_batch)


def make_eval_batch(dataset, batch_offset_idx , SEED, isValidationBatch):
    [user_train, valid, test, usernum, itemnum] = dataset

    def sample(user):
        # while True:
        #     user = np.random.randint(1, usernum + 1)
        #     if user in user_train and user_train[user] > 1:
        #         break

        # while user in user_train or (user_train[user]) <= 1:
        #     user = np.random.randint(1, usernum + 1)
        seq = np.zeros([args.maxlen], dtype=np.int32)
        neg = np.zeros([args.cand_size], dtype=np.int32)
        yoh = np.zeros([args.cand_size], dtype=np.uint8)  # N, I
        idx = args.maxlen - 1
        if not isValidationBatch:
            seq[idx] = valid[user][0]
            idx -= 1

        ts = set(user_train[user])
        for i in reversed(user_train[user]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        for i in range(args.cand_size - 1):
            neg[i] = random_neq(1, itemnum + 1, ts)
        gt_idx = 0 #np.random.randint(0, args.cand_size)

        if isValidationBatch:
            neg[gt_idx] = valid[user][0]
        else:
            neg[gt_idx] = test[user][0]
        yoh[gt_idx] = 1
        return user, seq, neg, yoh, gt_idx

    np.random.seed(SEED)
    one_batch = []
    mkeys = list(user_train.keys())
    for i in range(batch_offset_idx, batch_offset_idx + args.eval_batch_size):
        u = mkeys[i]
        if len(user_train[u]) < 1:
            continue
        if len(test[u]) < 1:
            continue
        one_batch.append(sample(u))
    return zip(*one_batch)


class WarpSampler(object):

    def __init__(self, dataset):
        user_train, user_valid, user_test, max_uid, max_iid = dataset
        self.dataset = dataset
        self.User = user_train
        self.usernum = max_uid
        self.itemnum = max_iid
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.maxlen = args.maxlen
        self.cand_size = args. cand_size
        self.batch_offset_idx = 0

    def next_batch(self):
        idx = self.batch_offset_idx
        self.batch_offset_idx += args.batch_size
        return sample_function(self.User, self.usernum, self.itemnum, self.batch_size, self.maxlen, self.cand_size, idx,
                               np.random.randint(2e9))

    def next_eval_batch(self, isValidationBatch):
        idx = self.batch_offset_idx
        self.batch_offset_idx += args.eval_batch_size
        return make_eval_batch(self.dataset, idx, np.random.randint(2e9), isValidationBatch)

    def reset(self):
        self.batch_offset_idx = 0