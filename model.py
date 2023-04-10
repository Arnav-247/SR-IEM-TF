from modules import *
import numpy as np
import args

class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self.u = tf.placeholder(tf.int32, shape=(None), name='u')
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen), name='input_seq')
        self.C_mask = tf.placeholder(tf.float32,shape=(None, args.maxlen), name='C_mask')
        self.neg = tf.placeholder(tf.int32, shape=(None, args.cand_size), name = 'neg')
        self.yoh = tf.placeholder(tf.float32, shape=(None, args.cand_size), name = 'yoh')
        neg = self.neg

        with tf.variable_scope("SR-IEM", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=0.0,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # # Dropout
            # self.seq = tf.layers.dropout(self.seq,
            #                              rate=args.dropout_rate,
            #                              training=tf.convert_to_tensor(self.is_training))
            # self.seq *= mask

            ## IEM
            queries = normalize(self.seq)  # N,T,D
            print("query size: ",queries.get_shape().as_list())
            keys = normalize(self.seq)  # N,T,D
            print("key size: ", keys.get_shape().as_list())
            Q = tf.layers.dense(queries, args.hidden_units, activation='sigmoid', kernel_regularizer='l2')  # N, T, D
            K = tf.layers.dense(keys, args.hidden_units, activation='sigmoid', kernel_regularizer='l2')  # N,T,D
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T, T)
            print("outputs size: ", outputs.get_shape().as_list())
            C = tf.math.sigmoid(outputs) / (K.get_shape().as_list()[-1] ** 0.5)  # N,T,T
            print("C size: ", C.get_shape().as_list())

            # mask = tf.ones(C.get_shape().as_list(), dtype=tf.float32)  # N,T,T
            # print("mask size: ", mask.get_shape().as_list())
            # mask_diag = tf.zeros(C.get_shape().as_list()[:2], dtype=tf.float32)  # N, T
            # print("mask_diag size: ", mask_diag.get_shape().as_list())
            # mask = tf.linalg.set_diag(mask, mask_diag)  # N,T,T
            # print("mask size: ", mask.get_shape().as_list())
            # # C_masked = tf.tensordot(C, mask) # N,T,T
            C_masked = tf.linalg.set_diag(C, self.C_mask)
            # C_masked = C * mask  # N,T,T
            print("C_masked size: ", C_masked.get_shape().as_list())
            alpha = tf.reduce_sum(C_masked, 2) / (C_masked.get_shape().as_list()[1] - 1)  # N, T

            beta = tf.nn.softmax(alpha)  # N, T
            beta_exp = tf.tile(tf.expand_dims(beta, -1), [1, 1, args.hidden_units])  # N,T,D
            print("beta_exp size: ", beta_exp.get_shape().as_list())
            print("self.seq size: ", self.seq.get_shape().as_list())
            ## Preference Fusion
            zl = tf.reduce_sum(beta_exp* self.seq, 1)
            print("zl size: ", zl.get_shape().as_list())
            z_conc = tf.concat([zl, self.seq[:, -1, :]], 1)  # N, 2D
            zh = tf.layers.dense(z_conc, args.hidden_units, activation=None, kernel_regularizer='l2')  # N, D
            zh_exp = tf.tile(tf.expand_dims(zh, -1), [1, 1, args.cand_size])  # N, D, I

            ## Item Recommendation
            neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.cand_size])  # N * I
            neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)  # N*I , D
            neg_emb = tf.reshape(neg_emb, [tf.shape(self.input_seq)[0], args.cand_size, args.hidden_units])  # N, I, D
            neg_emb = tf.transpose(neg_emb, [0, 2, 1])  # N,D,I
            # zcap = tf.tensordot(zh_exp, neg_emb, 1)  # N, I
            zcap = tf.reduce_sum(zh_exp * neg_emb, 1)
            self.ycap = tf.nn.softmax(zcap, name="ycap_n")  # N, I


            ## Training OHE
            self.loss = -1 * tf.reduce_sum(self.yoh * tf.log(self.ycap) + (1 - self.yoh) * tf.log(1 - self.ycap))
            print("loss ", self.loss)


        #             # Feed forward
        #             self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
        #                                    dropout_rate=args.dropout_rate, is_training=self.is_training)
        #             self.seq *= mas
        #             self.seq = normalize(self.seq)

        #         pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        #         neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        #         pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        #         neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        #         seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        #         self.test_item = tf.placeholder(tf.int32, shape=(101))
        #         test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        #         self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        #         self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
        #         self.test_logits = self.test_logits[:, -1, :]

        #         # prediction layer
        #         self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        #         self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        #         # ignore padding items (0)
        #         istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        #         self.loss = tf.reduce_sum(
        #             - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
        #             tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        #         ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)
        self.loss = tf.identity(self.loss, name='loss')
        tf.summary.scalar('loss', self.loss)
        if reuse is None or self.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            #     initial_learning_rate=args.lr,
            #     decay_steps=steps_per_epoch,
            #     decay_rate=learning_rate_decay_factor,
            #     staircase=True)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        # else:
        #     tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict_sess(self, sess, u, seq, item_idx, yoh):
        return sess.run([self.ycap, self.loss],
                        {self.u: u, self.input_seq: seq, self.C_mask: np.zeros((len(u), args.maxlen), dtype=np.float32), self.neg: item_idx, self.is_training: False, self.yoh: yoh})
