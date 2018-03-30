# coding=utf-8

source_1 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, args, batches_per_epoch=None, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                                     intra_op_parallelism_threads=args.threads))
        with self.session.graph.as_default():
            # Construct the network and training operation.
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name=\"images\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")
            self.predictions = None
            self.loss = None
            self.build()

            # Training
            global_step = tf.train.create_global_step()
            if args.learning_rate_final is not None and batches_per_epoch is not None:
                lr = tf.train.exponential_decay(args.learning_rate, global_step, batches_per_epoch,
                                                (args.learning_rate_final/args.learning_rate) ** (1.0/(args.epochs-1)),
                                                staircase=True)
            else:
                lr = args.learning_rate
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.training = tf.train.AdamOptimizer(learning_rate=lr).minimize(
                    self.loss, global_step=global_step, name=\"training\")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", self.loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", self.loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            self.saver = tf.train.Saver()

    def build(self):
        features = self.images
        layer_defs = args.cnn.strip().split(',')
        for layer_def in layer_defs:
            def_params = layer_def.split('-')
            if def_params[0] == 'C':
                features = tf.layers.conv2d(features, filters=int(def_params[1]), kernel_size=int(def_params[2]),
                                            strides=int(def_params[3]), padding=def_params[4],
                                            activation=tf.nn.relu)
            elif def_params[0] == 'M':
                features = tf.layers.max_pooling2d(features, pool_size=int(def_params[1]),
                                                   strides=int(def_params[2]))
            elif def_params[0] == 'F':
                features = tf.layers.flatten(features)
            elif def_params[0] == 'R':
                features = tf.layers.dense(features, units=int(def_params[1]), activation=tf.nn.relu)
            elif def_params[0] == 'RB':
                features = tf.layers.dense(features, units=int(def_params[1]), activation=None, use_bias=False)
                features = tf.layers.batch_normalization(features, training=self.is_training)
                features = tf.nn.relu(features)
            elif def_params[0] == 'CB':
                features = tf.layers.conv2d(features, filters=int(def_params[1]), kernel_size=int(def_params[2]),
                                            strides=int(def_params[3]), padding=def_params[4], activation=None,
                                            use_bias=False)
                features = tf.layers.batch_normalization(features, training=self.is_training)
                features = tf.nn.relu(features)
            else:
                continue

        output_layer = tf.layers.dense(features, self.LABELS, activation=None, name=\"output_layer\")
        self.predictions = tf.argmax(output_layer, axis=1)
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope=\"loss\")

    def train(self, images, labels):
        self.session.run([self.training, self.summaries[\"train\"]],
                         {self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, images, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]],
                                       {self.images: images, self.labels: labels, self.is_training: False})
        return accuracy

    def predict(self, dataset, images, labels):
        labels = np.clip(labels, 0, 9)
        preds, _ = self.session.run([self.predictions, self.summaries[dataset]],
                                    {self.images: images, self.labels: labels, self.is_training: False})
        return preds

    def save(self, path):
        self.saver.save(self.session, path)


if __name__ == \"__main__\":
    import argparse
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--params\", default=None, type=str, help=\"Params file.\")
    parser.add_argument(\"--batch_size\", default=1000, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=200, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--threads\", default=8, type=int, help=\"Maximum number of threads to use.\")
    parser.add_argument(\"--learning_rate\", default=0.01, type=float, help=\"Initial learning rate.\")
    parser.add_argument(\"--learning_rate_final\", default=0.001, type=float, help=\"Final learning rate.\")
    parser.add_argument(\"--cnn\", default=\"CB-10-3-2-same,M-3-2,F,R-100\", type=str,
                        help=\"Description of the CNN architecture.\")
    args = parser.parse_args()

    if args.params is not None:
        import json
        import random
        from collections import namedtuple
        with open(args.params, 'r') as f:
            param_list = json.load(f)
            while True:
                param = param_list[random.randint(0, len(param_list)-1)]
                logdir = \"logs/{}-{}\".format(
                    os.path.basename(__file__),
                    \",\".join(
                        \"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(param.items()))
                )
                if not os.path.exists(logdir):
                    param['logdir'] = logdir
                    args = namedtuple('Params', param.keys())(*param.values())
                    break
    else:
        # Create logdir name
        args.logdir = \"logs/{}-{}\".format(
            os.path.basename(__file__),
            \",\".join((\"{}={}\".format(
                re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
        )

    if not os.path.exists(\"logs\"):
        os.mkdir(\"logs\")  # TF 1.6 will do this by itself

    print(\"=====================================================\")
    print(args.logdir)
    print(\"=====================================================\")

    # Load the data
    from tensorflow.examples.tutorials import mnist

    mnist = mnist.input_data.read_data_sets(\"mnist-gan\", reshape=False, seed=42)

    # Construct the network
    batches_per_epoch = mnist.train.num_examples // args.batch_size
    network = Network(args, batches_per_epoch)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        dev_acc = network.evaluate(\"dev\", mnist.validation.images, mnist.validation.labels)
        print(\"Dev: {:.2f}\".format(100 * dev_acc))

    network.save(os.path.join(args.logdir, \"model\"))

    # Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    predicts = network.predict(\"test\", mnist.test.images, mnist.test.labels)
    with open(os.path.join(args.logdir, \"predict.txt\"), \"w\") as f:
        f.write('\\n'.join(predicts.astype(str))+'\\n')
"""

source_2 = """import json
import itertools

params_set = {
    'batch_size': [100, 250, 500],
    'epochs': [100],
    'threads': [4],
    'learning_rate': [0.01, 0.001],
    'learning_rate_final': [None, 0.001, 0.0001],
    'cnn': [
        'CB-10-3-2-same,M-3-2,F,R-100',
        'C-10-3-2-same,M-3-2,F,R-100',
        'CB-32-3-3-valid,CB-32-3-3-valid,M-2-2,F,R-128',
        'CB-32-3-3-valid,CB-32-3-3-valid,M-2-2,F,RB-128',
    ],
}


def is_valid(item):
    return item['learning_rate_final'] is None or item['learning_rate_final'] < item['learning_rate']

full_set = [x for x in (dict(zip(params_set, x)) for x in itertools.product(*params_set.values())) if is_valid(x)]

with open('params.json', 'w') as f:
    json.dump(full_set, f)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;7%VCVqE|k0)oax6*GjyvYTC`s}Z?E4>dgkac{%n=7fv}Z9C2yLvSj;GvI6~#Qz^B=K$n%0+Z_d+C_KFzILwpcdH>kw|Ezv+<XwK;+u~kCIG8b*anJS!qYZ}GW=YDQj@#}md3BWGf%*tbs;|bI=GjWo98HwhL;%(^D+&nFEDSk$iTqaf&~*i%zc8_&D0F4NzHvk?b|wbO~nh!D${&j(P`YZ2Lu2vf-6@)Dy!LJwAHr@lKHZ|rupkUhP$k@FZ7ZLhwyg5Xp(!%I~2+$UZV+4lQbiuSo0K8BOBnO@zc!eoDv^@3}1`;pd0FXMCAGrzR++<|7@94=KN4?Pldh$QK#<w{qXEeXZa$yfMe-m^Id7&pl7TKS*07;uH+$U=70PHkNKLvsj+4201_NK+jhTbh7PY}pU?fui+5CwYtd+>c{JR456hlF*SG1p2p?f?$~g%?MAQC-`G@5?dtoLn<8z5>j+4B;JQXFUMz|=1%ritOKB#Khtk+&Q@Z6YSA1ymRBL+WEGZ-82-E0`eYIzt6)m3Y>J0c7)c{*-#u@<-JO)-A~(-={Ppk|&3e18{oql9!nGD#r+LidX4vbd-Ekw@{saKunP?^c+f?XvuRBu1lNb+`7#f2JUR6S-uY_LCBWTHt>BeT+5t*7-CLMzFgvZ3yP5@r#6PlD%`(NZY8>&WnYxXoJ^ViE4$*tKC+AUxK-i|IJVKQ@cawx1Vk=>_bVOZ9`n@C3Q!dheHcoBwE#TCzp}jKul2p$&<LCH_A_gV1Pi^toq3*rl=Zjn#t7gv2{Y4!Vck;iwag*MY@1_+?2Xh0LX$=tt>?x{JWMT+1&tE<o0aX;?O@;S6zTZA5$sE&lm_0kx6oyk#3wB!Wv{G4Gi5@2&`zYUVqFyM(>z9(47OIcAk5ax?izk1Z7p#2N~m3tA@_<IF6>x5KFQVAXwbPo$Wgh@p&38RNG7BAn?Rk3P+J9RRZfdj@r_PQ?yWvG{qPSC5bE|MFr&4)SK_?ZCc?&i{Nb9W#V#TR})9*E~J>Wq`Burg!j^zHYDzFv7t=UZ6xZ)2Z2W<_TUZm4}KPup!DhQ#fXMj2s<)EMFV|hAfPJ#&B{ZkY1nAyItEewn)8$!m=c+s<7$!;*5$MMI>qMr!CaX+8o-|CI@Q{81VYFLV&*AXGzBHVg%_dv!wiga^NF-1XjA)nG_nE_jVN89(R_aWc)<8#%7&YaYQ0uT<saJH2V_~EQ(me0b*+&{+&3r}I|M7!+Od!4<0c67?&ojtdO|lYUynshH_$knwcb-Irecf2gXcYKVG&1mwQzF@|1Q<WjN>|PJ&`Z)kwmszJY9Q8`e0;kEyp9*SK7SYg++EW{7+1s6S0To%Af}>Kosg}>+SOvK%3JeO<4QC4Lxmoo8Z;dCk3(t-2o=cvHe40tQZw7%s4`iHUblKXJHiG5Zb$8{5#(Ck`PwUX<fZ4CTM~E`35toFLLL!NkLMg!4WzmWZvGL*QUm<H^Odig6Lnc*SF-$dv5S8bYq5nWvlNHRwCQB@-5&ec_1;7yaC<QZ{#{m`!-cW7Yy19E``~<)G-e<-eDvmfc_5!dHa3f@=#B5)$*wvnbH~`BETnjr$NX|dwyEn{ftM~wvtEhm0N2?0Q_~6!v(LgntsAsM|16R&k@Lefqa{Gx_vHngrTY%Nb4R-CpYBrFB6DiCNG*j1uZzpKLS##N&zX$x?c{owN0pm*xcFdfgxAAKzo?&LHt#WQ`<|g{Tn6enQ^TJ3hhERb5E;47K5=plx7QDYY)?`X^L|pyNysX`;Z~!-=9Uv_1X~%G6$9|=N=Id)m%Ovtxp{_tXI*^mcORVTaiH@1=<=JIMA2dyJS+3|E==x<xA(zLRpVSrj&(JJjGa@Y&5vWVHSvY{o_I%xA9KzLe##XR!k$nkxr!fr**z@+?wpQuG6xlQ5<BOA-Jk*jGoE)K=L-$)pv9&u(5-*{DVEjr=}2N<4H|e-ql%)bzR2j@275xZ#(~U_WzmI>lJlX%9KAxwB$6Nx$pJN1hwk{AI8Pq+dIc^_4KDW;enCIxwOt;jHmH<!9GBAiHHNc{6RMnHHl)5yt$enH7&uOXG-8hM6>5z3#%SfD%04*NZMbM`rhTtVOp|)jl;LG6xmlIF<y82Pi@bjuh{BgfewZR0NPE|G;N`Pof+fi_<fNXelm)HfF{CiS^Ggq&12EZ`n*@koJDmXLM|c1z_qOZP#2ebI56|p>y|6ul5Dmu=s!7|{f|4Yr9xi=2DSm2TDE9umy^pvk}WNPJdSR%Hj`q{T9EROKXbj0Y!U;30{<nV7O#{lS=yCKl^+G0>KyiF8BMl28A<J`mkOC^C!hrk9<-Y6WLEILnX;i95X(zT_jB?K2;fE#PieK724!bo;eYpc*0k+RsYvfwA55Xh`-nVwZ>*`-YtMn*Iu&GS1PX2wp*j0VjF!N8me;JJ65Gs5Oenv)I8k~<$ynhy;pT_FHH-n#b1q)mj2nle_Jk;vcllkmAb)VF_W{&}8ap@NaCU&VsI4b*r*xdATcvREU<ZjYX`1AXBjc?dj11(n_KQL@03{n4Tl5tk%*wA&yBqLw@HGcXQQpKAMdteoj)&cFbCy{p1>~sVOj)N(vhaGk>XP-2Ik5LgY!oH;R9dFBH{mGp*+v^izq_c(8p0sg5G;hrw)7;#Uq~8hH<QpPY-0*3gH43iN!~8vB4I<Ay~dh-n(WD62Y&Xt1D{hJF2U=?0@2V7O(B(=BEy;;NKP*jenPO6gHNn|q>GAaH5)y;ShTtppOU6`;6Ohet{|vEg?xD)S3}Liq_NYZ_9X_3T7Mhi2|(stph_)9;*s(XxAUcp7g9ci3)mTxh{Hfb3;AL;iPh8zJ%dXibwV6Ww1Fp@8+E$C_;Op!;&)*o)8YJgx-EvGqNdK61^G=l3>gp8(95ww-}d3cQ=qX(OMo7)`c5sC+4>y9I_WwtNYke=+TVaZ@?<Vc#ILf|5(B6Gd??ynbuT&J5s`Q54Wu9-`gs0TF-xjL!+XnfFWBPJ4z!Z+Uz1LN^UxL{Lk+D5H~Eb`bZ<bYl%Vi5l_}J7R8h9cfQ&c(!Sr|^4)Z!dN$Fyl$UEFZIi?b~L|~s|Vw19=TWUpqX8Rz0EzOx`^+OJYUI){WNjSZ0W`I~+0qw0ThuMkM9a@>|8(C=rO)F`l`|Wo!<6o8#<%A0CvDU@Ni8<@p*3OLHi@Y*kBIT+|xku<V-Qy(fhI_mwCW`OBif_>R9DD7e$#F%z=_G~cJrL<$5~(eeHk9k_M^3?+LHbmU-AucT-&dOPa72ZXFrd}F1J$E$aFa1sJ*n=IS(y3d;8GpPV`{T6(jGqng;ENd5!mOtr?y?FUv@%Qs7JM|IrKx1iOscR=-i)@i(W`Ct{COkpT~YGKjKh(&pG<fv1Yb!i;I|yaj7mErw-O)(fK5D<wuMeKi`ae*k(@ZxLhdc`pS8?U(3bEf&o!BO$TdaJ%RzLLr+j?bWojk=SgQ#P6dF5;Js^SI;tejErSldmr)>)TL&WdSse6PW2Vs63F{8bx})1?wO;PZL24B1rw8{ATA4~(Jkj(e@$v0$c0%7+ZrM4D^NsBvv{w;>0sT^`{<HAy+vHeBm&eE5HPmYAR`l<&p?+UpQAabckS77=)4s@5qOu3V!Iq2W2WiXl!@NLB&z4{P_{1S^Ja@J@t|bSQh%=}xbz*0XnVKp+pk~iSUrp9QB}5FGDsYdn4WSdul5H3gXd0M&*+6=Dx>7Uz`g-#eRM#1<!cr%5dje^vC*Ls!AmCmC4Oh&AcmoR0sJiR(b$gVQv72h@!f63D03^6>IN%P6-%c&3E-4lIww<0nLlyflHq<OAZlG5O4F*7#F6>_|GkZIc9)XXMpxL{l&y0zFN+nU}{Ty3LSiFKtVfp{1u(CHmE<{YV5n}6_KDO+=0kgOl*%78yLcg#d*UzN>fQor0S!$(5P3kgp;CHe92o-T)k@h!6eZGGRMo&D;Gb6~ND44kZu;}wECE~BIcmP8<V<VDH7Oa889Cjfl&2NfS&F2f-v@Tha+$tLr;dmJizyhv4cI9B>Quo-`8p@N+Cvmoyew5V*;kFAP<@WEQKj?GsRp`FO!NV&d^*Udxy`VvD*A%$5oN0p<Rj%Rd>i2*OurEEpSN2?=e3h^A3KoyK{s4ZUX|jz_xoNNk8Pw33zYDYtG}R<rv#o^7YgZ8e*w)!B1n?q{nPcdXk5PWsT#U|*Vo%nsAiWe;P;a9pl;@$jx7V8h_MTpj2nC2e-`WGCA))18*Za2=p@1c0NY%K`0NMm<tvD4rCgobcQ9;ro|FA`RkU^AteF^r`{}WutllAZ;<VsPG)nxk7>Vu3ylRkZ*0*6gWL*x~c$@glz!?Krx60i|P-J>W4cP}be!m4i<QA^aUo?wQd$238vLcO;dXdKlL`+DVnWrsI5OSLh?N3jmJXU1EI*svItBLGn8bAt5ahR?apbzZOn5?Sb&z@d0;(5$&8wNqCC1n8=*K8q0^mvSbmRFUn-evWJLUvSkmpj0aIIA$}Y!Y{tkN5{fj(!ATfwblLG=5Y!i&Bnh-Lat(fuT$1jc0aoQccT=y`go&QUtV4#S9pMy<g|K?V!YVWlW>v;T6b!D%}-DCGw}>gSl+#8VAViFRZhNw*J9y2f+4kw5`l7q7Q2%Lz-sRj%J<o9XNw)n8!ryA;Epp3OE*$;f<isM)<SuX+wu}S0pa8OG9$e(@@f>VTaQX}e4fygFdH3aoY3BU;o+c|tq$#en7H9Ep8g5e5}cR(>1N0eGkR`=Kwnd#)eaDM53|qjz{nNMQU+n}ig||Jhl(2S?q#Fv@ENtx%(IH+;{_zgbY?7Kh*i>;cdc}<aMhU`Xm?z?+s?t1rHK5_Q-3K4eGr(Cmhf~Yt?iQ0K@v(e*C-?3N$Z@-u7d78*7}<%GDAYhcyARI4JA^=-eXl9Nf6|^8WFWo*=*NcE9lfaoV7b;G&Vq5PFZS@%Wm1J)tETmS1x0WJ`sDW#sJiy9kM?yx}T$h?=f%8)Gsq+%P9zbxQ_gJTEx&^f%Ddbqb5cbMd9{f{lpXhr_J$(M{Z6OXHh2Z#x{<7;)3{9=-7U$=+PGMM#b(Dx>&3LK%gbv-ir<)#7M+Jde_>AL(8Q)xxl8&dztA9fH&htN4mL2*+gq(l#cX^{|wXzzMN!sMx{B$l?&T^?3lF^W!0iU0M+4uQa5mk8iGvWe=fhV$91vvUL^`qUKwfSYrI?w;h<72>`gYTgQmnlLj(a+u9{X<e1gFR&s5*URI7QQjbhr+7ooeVcY3gwW@}yorlm59f#3+KGNK^yi~pyV24zUUI2(1V>piIW+eM^Y-zng>qZKy{*ly`?@zWgGu>7c9Yp3;4XY-*VFn!7YyVxZ&kiwp;difAY_H_9%jgfm_p&fC4ZHT1^V<~98+bUO_`xtuMr)bpUul{oz|Kt96S}~KAHsBny3t?kAgLWj;`HY!3%vxBDo>soO1Fmu$o(sI(5ACXgOO%SP1>5D}hC30gF9B~4D)u-6p6plx8F}xtmAoE+Z9;<!CI@v|f^=kki%ocoVT$N23f`c_+8IMaygt$PTj{J2yMD6i8HiH07MiMPvD6jU>%MX<3IKi9)KGm3Ea_hLv$1{ju>;pdS`5Wl^7=K4{A~jOZ`fVszm+ktIX(gl_ON39-4*}rOMKh^9lMT9D>v9RBu4=zo(q$=65K~xM65Vn49#@Vk6&!~vSAVQ&ckBiSpdXb^iEe?{KfGQB8gfv4_%(!)|m@^KYppz_o0?Y^R}UiF<uzQ{nwUZOIy<&UY#>%K8|dLJ>|(Gf5*0h{3nw6y{vOcrZa)pmbGUsU_V<yG43B;-sHld`7JV%1Ooikchj+?<oxs9j0~#;g`Y}~2&LFML?VKxnF=}?3|ckR3}e1yj@(QZfeCWoBy`>Y!MFD&0pb96ea94pT!*wx%up<lOfy4Gw|0>`jy~iNDh|puertoi&w>h%z0kG*KRFE_hB3yKc=j#L&IB}#8MuFA(>J6-O7^2y5T4#k_b6dIVn2sW+OZZ>IJUEPPs}<b^*ODHBIt=`&G!Tu$3!AL4oNh`avr}eL1bpI46a)YO=Vm~X;U^^2fLYag@`I@lJd<2*`Ts~5-AGCpEvYQ+h!jKp=AR6Jd7{X5<+e)E&#@Z>vh{7I7G0s=o>8N9(}fzGkR1&K1Ca~$T9Pl+Eg?~QeWR(DCXkbVQ34BD29MSP*AO$yXz{6N`b|%@*0_l^rv|YcJ#^cxVfH@%u-xTw+s1vr1<8H9t+Svfmr#)Sc`*bCKbmAIQkq@Umeoj?a)YMg2<Orlc&!TY&P)2h1I`mT*((4fW;?K=1XN=#isBxcjt)ps_xmTxh{t300000+y?DceD>bb00I6cpqv2!z#laNvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
