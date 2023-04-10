import random
import sys
import numpy as np
import copy
import args
from sampler import WarpSampler
from tqdm import tqdm
import time


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def batch_evaluate(model, dataset, sess, isValidationBatch=False):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    valid_user = 0.0
    LOSS = 0.0
    sampler = WarpSampler(dataset)
    num_batch = len(dataset[0]) // args.eval_batch_size
    tnow = time.time()
    for i in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b', desc="Eval", file=sys.stdout,
                  position=2):
        # tnow = time.time()
        u, seq, neg, yoh, gt_idx = sampler.next_eval_batch(isValidationBatch)
        # print(time.time() - tnow)
        # tnow = time.time()
        predictions, loss = model.predict_sess(sess, u, seq, neg, yoh)
        # print(time.time() - tnow)
        LOSS += loss
        for i, p in enumerate(predictions):
            p = p * -1
            rank = p.argsort().argsort()[gt_idx[i]]
            rank += 1
            valid_user += 1

            if rank < 11:
                NDCG += 1 / np.log2(rank + 1)
                MRR += 1 / rank
                HIT += 1

    return NDCG / valid_user, HIT / valid_user, MRR / valid_user, LOSS / num_batch


def make_eval_batch(dataset, SEED, isValidationBatch):
    [user_train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

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
        if isValidationBatch:
            neg[0] = valid[user][0]
        else:
            neg[0] = test[user][0]
        yoh[0] = 1
        return user, seq, neg, yoh

    np.random.seed(SEED)
    one_batch = []
    for u in user_train:
        if len(user_train[u]) < 1:
            continue
        if len(test[u]) < 1:
            continue
        one_batch.append(sample(u))
    return zip(*one_batch)


def evaluate(model, dataset, sess):
    [train, valid, test, usernum, itemnum] = dataset
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    print("User Length", len(users))

    for u in train:

        if len(train[u]) < 1 or len(test[u]) < 1:
            # print("#")
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        yoh = np.zeros([args.cand_size], dtype=np.uint8)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1

        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(args.cand_size - 1):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        yoh[0] = 1

        predictions = model.predict_sess(sess, [u], [seq.tolist()], [item_idx], [yoh.tolist()])
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0]
        rank += 1
        valid_user += 1

        if rank < 21:
            NDCG += 1 / np.log2(rank + 1)
            MRR += 1 / rank
            HIT += 1
        if valid_user % 1000 == 0:
            print('.')
            sys.stdout.flush()

    return NDCG / valid_user, HIT / valid_user, MRR / valid_user


def evaluate_valid(model, dataset, sess):
    [train, valid, test, usernum, itemnum] = dataset

    NDCG = 0.0
    valid_user = 0.0
    HIT = 0.0
    MRR = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if u not in train or len(train[u]) < 1 or len(valid[u]) < 1:
            # print('#')
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        yoh = np.zeros([args.cand_size], dtype=np.uint8)

        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(args.cand_size - 1):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        yoh[0] = 1
        predictions = model.predict_sess(sess, [u], [seq.tolist()], [item_idx], [yoh.tolist()])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]
        rank += 1
        valid_user += 1

        if rank < 21:
            NDCG += 1 / np.log2(rank + 1)
            MRR += 1 / rank
            HIT += 1
        if valid_user % 1000 == 0:
            print('.')
            sys.stdout.flush()

    return NDCG / valid_user, HIT / valid_user, MRR / valid_user
