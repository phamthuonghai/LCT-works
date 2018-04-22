# coding=utf-8

source_1 = """#!/usr/bin/env python3

# This source depends on the NASNet A Mobile network, which can be downloaded
# from http://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nasnet_a_mobile.zip.

import numpy as np
import tensorflow as tf

try:
    from nets.inception import inception_v3
    from nets.nasnet import nasnet
except Exception as e:
    from .nets.inception import inception_v3
    from .nets.nasnet import nasnet


class Dataset:
    def __init__(self, filename, shuffle_batches=True):
        data = np.load(filename)
        self._images = data[\"images\"]
        self._labels = data[\"labels\"] if \"labels\" in data else None

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images))\\
            if self._shuffle_batches else np.arange(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images))\\
                if self._shuffle_batches else np.arange(len(self._images))
            return True
        return False


def get_layer(layer_def, features, is_training):
    def_params = layer_def.split('-')
    if def_params[0] == 'F':
        features = tf.layers.flatten(features)
    elif def_params[0] == 'R':
        features = tf.layers.dense(features, units=int(def_params[1]), activation=tf.nn.relu)
    elif def_params[0] == 'RB':
        features = tf.layers.dense(features, units=int(def_params[1]), activation=None, use_bias=False)
        features = tf.layers.batch_normalization(features, training=is_training)
        features = tf.nn.relu(features)
    elif def_params[0] == 'D':
        features = tf.layers.dropout(features, rate=float(def_params[1]), training=is_training)
    return features


def variable_summaries(var, name):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    return [
        tf.contrib.summary.scalar('activation/' + name + '_mean', mean),
        tf.contrib.summary.scalar('activation/' + name + '_stddev', stddev),
        tf.contrib.summary.scalar('activation/' + name + '/max', tf.reduce_max(var)),
        tf.contrib.summary.scalar('activation/' + name + '/min', tf.reduce_min(var)),
    ]


class Network:
    WIDTH, HEIGHT = 224, 224
    LABELS = 250
    CHECKPOINTS = {
        'nasnet': 'nets/nasnet/model.ckpt',
        'inception_v3': 'nets/inception/inception_v3.ckpt',
    }

    def __init__(self, args, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                                     intra_op_parallelism_threads=args.threads))

        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 1], name=\"images\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")
            self.learning_rate = tf.placeholder_with_default(0.01, None)

            images = 2 * (tf.tile(tf.image.convert_image_dtype(self.images, tf.float32), [1, 1, 1, 3]) - 0.5)

            if args.pretrained == 'inception_v3':
                with tf.contrib.slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                    features, _ = inception_v3.inception_v3(images, num_classes=None, is_training=True)
                    features = tf.squeeze(features, [1, 2])
            else:
                with tf.contrib.slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
                    features, _ = nasnet.build_nasnet_mobile(images, num_classes=None, is_training=True)
            self.nasnet_saver = tf.train.Saver()

            nasnet_features = features

            # Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `self.loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`
            with tf.variable_scope('our_beloved_vars'):
                for layer_def in args.model.strip().split(';'):
                    features = get_layer(layer_def, features, self.is_training)
                output = tf.layers.dense(features, self.LABELS, activation=None)

                self.predictions = tf.argmax(output, axis=1)
                tf.losses.sparse_softmax_cross_entropy(self.labels, output)

            self.loss = tf.losses.get_total_loss()
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.training = tf.contrib.slim.learning.create_train_op(self.loss,
                                                                     optimizer,
                                                                     clip_gradient_norm=args.clip_gradient,
                                                                     variables_to_train=
                                                                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                       'our_beloved_vars'))
            self.training_all = tf.contrib.slim.learning.create_train_op(self.loss,
                                                                         optimizer,
                                                                         clip_gradient_norm=args.clip_gradient,
                                                                         variables_to_train=None)

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", self.loss),
                                           tf.contrib.summary.scalar(\"train/lr\", self.learning_rate),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]\\
                                          + variable_summaries(nasnet_features, 'pretrained')\\
                                          + variable_summaries(features, 'near_output')
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name=\"given_loss\")
                self.given_accuracy = tf.placeholder(tf.float32, [], name=\"given_accuracy\")
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.given_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            self.nasnet_saver.restore(self.session, self.CHECKPOINTS[args.pretrained])

    def train_batch(self, images, labels, learning_rate):
        self.session.run([self.training, self.summaries[\"train\"]],
                         {self.images: images, self.labels: labels,
                          self.is_training: True, self.learning_rate: learning_rate})

    def train_batch_all(self, images, labels, learning_rate):
        self.session.run([self.training_all, self.summaries[\"train\"]],
                         {self.images: images, self.labels: labels,
                          self.is_training: True, self.learning_rate: learning_rate})

    def evaluate(self, dataset_name, dataset, batch_size):
        loss, accuracy = 0, 0

        while not dataset.epoch_finished():
            batch_images, batch_labels = dataset.next_batch(batch_size)
            batch_loss, batch_accuracy = self.session.run(
                [self.loss, self.accuracy],
                {self.images: batch_images, self.labels: batch_labels, self.is_training: False})

            loss += batch_loss * len(batch_images) / len(dataset.images)
            accuracy += batch_accuracy * len(batch_images) / len(dataset.images)
        self.session.run(self.summaries[dataset_name], {self.given_loss: loss, self.given_accuracy: accuracy})

        return accuracy, loss

    def predict(self, dataset, batch_size):
        labels = []
        while not dataset.epoch_finished():
            images, _ = dataset.next_batch(batch_size)
            labels.append(self.session.run(self.predictions, {self.images: images, self.is_training: False}))
        return np.concatenate(labels)


if __name__ == \"__main__\":
    import argparse
    import json
    import os
    import re
    import sys
    import random
    from collections import namedtuple

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--params\", default=\"params.json\", type=str, help=\"Param file path.\")
    parser.add_argument(\"--epochs\", default=300, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--threads\", default=16, type=int, help=\"Maximum number of threads to use.\")
    parser.add_argument(\"--learning_rate\", default=0.01, type=float, help=\"Initial learning rate.\")
    parser.add_argument(\"--min_learning_rate\", default=1e-4, type=float, help=\"Minimum learning rate.\")
    parser.add_argument(\"--lr_drop_max\", default=3, type=int, help=\"Number of epochs to drop learning rate.\")
    parser.add_argument(\"--lr_drop_rate\", default=0.7, type=float, help=\"Rate of dropping learning rate.\")
    parser.add_argument(\"--early_stop\", default=10, type=int, help=\"Number of epochs to endure before early stopping.\")
    parser.add_argument(\"--warm_up_epochs\", default=20, type=int, help=\"Number of epochs to warmup.\")
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        param_list = json.load(f)
        num_retry = 0
        n_params = len(param_list)
        while True:
            param = param_list[random.randint(0, n_params - 1)]
            logdir = \"logs/{}\".format(
                \",\".join(
                    \"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(param.items()))
            )
            if not os.path.exists(logdir):
                param['logdir'] = logdir
                param['epochs'] = args.epochs
                param['threads'] = args.threads
                break
            num_retry += 1
            if num_retry > n_params:
                exit(111)

    os.makedirs(param['logdir'])

    print(\"=====================================================\")
    print(param['logdir'])
    print(\"=====================================================\")

    param = namedtuple('Params', param.keys())(*param.values())

    # Load the data
    train = Dataset(\"./data/nsketch-train.npz\")
    dev = Dataset(\"./data/nsketch-dev.npz\", shuffle_batches=False)
    test = Dataset(\"./data/nsketch-test.npz\", shuffle_batches=False)

    # Construct the network
    network = Network(param)

    # Train
    min_loss = 10000
    early_stopping = 0
    recent_losses = []
    lr = args.learning_rate
    for i in range(args.epochs):
        while not train.epoch_finished():
            images, labels = train.next_batch(param.batch_size)
            if i > param.warmup:
                network.train_batch_all(images, labels, lr)
            else:
                network.train_batch(images, labels, lr)

        cur_acc, cur_loss = network.evaluate(\"dev\", dev, param.batch_size)

        print(\"Acc: %f, loss: %f\" % (cur_acc, cur_loss))
        sys.stdout.flush()

        # To avoid spikes
        recent_losses.append(cur_loss)
        if len(recent_losses) > 5:
            recent_losses = recent_losses[1:]
        cur_loss = sum(recent_losses) / len(recent_losses)

        if i > param.warmup:
            if cur_loss < min_loss:
                min_loss = cur_loss
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping % args.lr_drop_max == 0:
                    lr *= args.lr_drop_rate
                    lr = max(args.min_learning_rate, lr)
                if early_stopping > args.early_stop:
                    break

    # Predict test data
    with open(\"{}/nsketch_transfer_test.txt\".format(param.logdir), \"w\") as test_file:
        labels = network.predict(test, param.batch_size)
        for label in labels:
            print(label, file=test_file)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;2Z1<09^nWjy+8Iy!C^=OSot7dtklRv8_?`{V?u_b+i{4AEp=|a|llnHoIx<J~-FLu2J%443MF^i-abU6dJv*TR=!lh8@CZ6lkIsf&ff7{<Sl-*nM$w0U!nbeqj`({x&O`!z+e6z3Ld(DZSlW=2pa?HQN%6lD-cKCF$-$85CWRN}TKF?*v7ruqme7CzNn9_#<s9g`Fnd5Fy0aG|1xY6D`#{CE2=Y*M3B)C!oQG%Nf=bDzXV4yeT(fU=EzJAbTFp;b<4X&merw#`s-ojN+x8ITHG}{qyRe-32gO%q}U1>RxMe>xv-TykKQ|rFnK!R3@H-k)4ktNT-1hj^$r~L~9g-!9nXeslZ8m^8XKP=jg&YZcjaTZ(c{hyuA6cVnFF<r;8da|FT&Xf7a{Zj6fDw<$s+%Fybc{gLcPsBVqA2(dyd61+I78WmCm2mihljdTyTr4zD~UWe<6XjH7>E%Js?kke3W9Y(J-D4z`NRYKZS@IK2bJtfu<7e7gX=)3q7wzGFxRqjTqa%pBJ+$(;3wK{L#VT1URK!JjppMiHK1VHffaLc%InjOM?-K{JWAeg?TIB~JN@K>Uowiz;&h{*|#*q)MP=RF?G-{;iFz4CHwhm{7pTM&zS7cM=!&F=oOzgeO{Tj<a0Bm4Cl*H>-;YE3iRiLCwd&iRNt8^QWCSsO3`;y>qq_8PYRe&ApRN?sNJ8M&P7-A$TWGkSpP|{f)%;k_8_)B0hm1G1rbjz?ytb&~nEes>VF1SX*#v&^rl`OrfpfS(oHK+FSTwdzuPXy48LCLg>wtBrj7&_Y(#unH~o>fQM4c8HTC{PjDu>M=E9ChH;i2+5;z7-M{HZZ*pF-0dz<^QA~N#iX2aUrfAd;=+hemYYpTJl(vDJ9<sHtz2?0QBxjagp&oYQvbRDg$Du2j7g**tD=$IxJzD4mv)x8-2;)!7c>di9p1Pu1BW5ky7#)X(Xx;|9kvz+NgTahs%J=@;lHN7NR3k;qO7a+HxYe(dFf@$y!hErQ7KfgRLSDjXjP4KrJ&rz72lHujk>zbfcJa%{?aG_vmUy0a(y(r8->E<0>O1+n`j1qSKZ^-j<>(jBbk7rmejehO(kaWdPRNAc<yry8LaCLH4w*?B?<TGbhamWKa;=eGAlUkFZ#@|@$tUSjZV%!67-VFS#<sNYGwh^vXxFTpg=!=)TkpdU_}Z3R(&SuXt2k)=Ir9-!<eIXftk@lw-wg+@k&xN2AGr1mJw`e1C}43YuUL$W`qkLH&p)S5lDDYAomF)9(Wj}uV_D-{ltPTRS?_A_aG(G*wY7-{V0`M8P=BRn+~&w{v$l4A=X6IhvD?_kVn1e{{M2Ncc1l16C;!C9%iBu8tL=eDcDPKa0bno)UHT#{RX1hm+mg$URk`nf%A*w%P*Ek^)Vq+;04Tus;j7NHcZOKv@7=-Lj%ycbv6lDQv#8g2dk1qIzPMmp5!RwAGtlk?9@h*yYx9G#SXH(>t|!$lC8}lh?Zd}2vX^Y_xVgku*fQkZhvOh*B-e|wjXU^U)v6UA5?(y$!Wt1Uye7*@f$;j*r_rF!yS1?Is_a4dd&miZ_D6yCk3h-`Xtur*CHym4xQis3Ty?41?@v2N45Ge1mYi@e!4I)WaCQ2RAfKx=S36p&9?h_GicN}|=I3Aj%qs*eG!B50WZ7gCFO9oJQv;~!3eQC5g*Q^5kyfz%#_`L4TC@U;;x)LNxGEjj&zfCk$+$8ZB$Qd(d;TfLKfN72(o{^91~3^`l`jhIi-`q`P>(H;1lSJN_|8cq%G9F?zgs0KB}Lx*+%Y~WIVJp}W$){Eyd9*<sPxw)K0^q0vjLn;Zd&zvi3z8HUry`J05}MSIk(2XaeDRxW2k=f;sJ#!nacrfa|^c&jk&rkNp5I^lCv5EF`ZQ<E*$DmUErN9aj?FzjP|d}Txm>cd3vDRMfPl{bC9hrfSwKwsdh%9@_cPv4}nzm`l8b4on=ADcvZf&W)I364q4@ec7L}TA^99&TV*xApJ!=H#6c;v)>hQ5oD~E!<D7%fYa@(D%EZjmy_`rukEv)tPDaV8AfHA9<PaEf1K7Wlq;O@JO+gKzRboBv<xWFhuJZg(0CsF#sHsuV1}4PA2hJO>NQqyFg8#s&;$s19QA!h>>7eL>f~dI3zD!DLZYBCGv*z+e!)Qg85nFfIH{_4Rm#O3>e<cqBDbs<^W&w-tf+l|=^NRkM6@Twxj!Uf^3-d_%fjky)%doyVuTSz6it|`UlI-Wl89n2*)vV7K-eFcTQ!5-w_|&96Rec(BV>&7BTE9O{V$5)}4GsJTa*i6a8NUAlLI4W2(f(Hx+%N*9bFV+8kDKm4NDxuaQ$zJ#n`#xvw!UlN!Wt(lJ1oQol#!b?>Vya^5$6r-h^d-RU!%^W(f;smXG{K(-OT#L0bg}vtten<(@G@@Ph%-x4;58x-_k9pgP;=Hy8gum{<S4b&t)$jv(Q?Ky6nM3s#1O&$(@-SfiTN$vXP1ai<Uyr!iV+&J$8uvHe#bD`wV;9kd&(SXRr7uVs)$~pOZ)L8IN%*vUYs0UZT69A)R5ySCOP14xG*}T!208{(ja=JAYks&ha^ybS}QI8qT*!9q|hWO_vP3`%cC{5}?VO^{dl_5!ZO7bNfr!;R+vuAHZ)@7B!S}Ib#4TY#I%>gVhW^JsOp~vx9-eDn7N6+!gdA_f5_fQ)#JHc{baofnjsV?*H1c893s4faN9}xCkFMEULSATT>|=&Ns2pB13+F?^ve_Sz8;lwlY6o*M7*Txi=y!h*wywWOO}%7=f!1&QC?mRve;72`3x9oqutKwPw|De++w`8oSx>!Y$x?P#_Wl$`T--{KoW#O}SR{F6*}2?@HT^rTS+ie_Bal9B<NB+OglO$CsdT{_F&NOC3q;dixuAh3~M{yz-o`{o#|ww<F32)PXjICyUmn2^o{jf7U0otO#VM0!w`bqlK~@Plwt>G!-V=Mo>P(Gu1*MxW$z<M~wa7XE%Es?BKdQW@m+Z58R8;hg|CjI)9+#ComXaCA4*lLm2*>V_-ErNcs@DrvBKW`eGp#<M!m|so!Eq*~U+dR`Qg-RLQsnbg;(kWd3~r{6qhDOh}#AGuf1qaAm~?C=H<r+&b#yM9ks=5tJ)qFZmm(_kIRLQwf+HvQ=LLgy3q*)>cg2*V@Ak-SPV|X+=TeJ8=DjnN3Jb9HJ7_nkH0Sp$W71XlPOU3Ww6kRavM->R5HeXbO{Ydd_UR#W3Vuv>P(z#WfP+>3L!|Ws?9S)1Y@_|K>~yw@TV-#dk`ciIv2AstbH9B8R^%szbtseLGR&?=S4Jz5=D|1*lU@$OS<n(<@NLMWHrH_^_k)tgF$}kF=O?ArwGuB*PHX!E(2>>YK^Ptd}q>2Vm9|x(vmcA`q9OUE7ca5eL@W%2RQl=|*qV;v3&U2*iVX;*`*ByHBv1v;3D~Yfxo7Ovaegxt>dOP!>q+@@>_YvW;&U{SdtC;pRFCfm^5U%=UgDs9_C&#lu-P!}cq5RQ0fzQVx7Errr7AmP|{2BHw39*??Jp=BR5UIPwjO>s=DChr=wG8z!DxqW&ojdf$GTCp}G4WZYqei0D&h9w3Hu)JBa27!Cjce~_x=Z8#q{00Ep9?Kc1bHws`wvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
