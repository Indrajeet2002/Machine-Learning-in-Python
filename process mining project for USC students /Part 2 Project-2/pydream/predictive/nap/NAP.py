from numpy.random import seed
#from tensorflow import set_random_seed
from tensorflow.random import set_seed 
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#from sklearn.externals import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
# from keras.layers import Dropout, Dense, Input
from tensorflow.keras.models import Sequential, model_from_json, Model
#from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dropout, Dense, Input, BatchNormalization, concatenate
from tensorflow.keras.utils import to_categorical
from pydream.util.TimedStateSamples import TimedStateSample
import itertools
import tensorflow as tf


"""Use legacy optmizier for this project"""
from tensorflow.keras.optimizers.legacy import Adam

"""Uncomment this if you're having problems with numpy() eager execution?"""
tf.compat.v1.enable_eager_execution()

tf.compat.v1.disable_v2_behavior()

def multiclass_roc_auc_score(y_test, y_pred, average="weighted"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

class NAP:
    def __init__(self, tss_train_file=None, tss_test_file=None, severity_scores=None, options=None):
        """ Options """

        self.opts = {"seed" : 7,
                     "n_epochs" : 100,
                     "n_batch_size" : 10,
                     "dropout_rate" : 0.2, # updated from 0.2 to 0.1
                     "eval_size" : 0.1,
                     "activation_function" : "softmax"}
        self.setSeed()

        if options is not None:
            for key in options.keys():
                self.opts[key] = options[key]

        """ Load data and setup """
        if tss_train_file is not None and tss_test_file is not None:
            self.X_train, self.X2_train, self.severity_train, self.Y_train = self.loadData(tss_train_file)
            self.X_test, self.X2_test, self.severity_test, self.Y_test = self.loadData(tss_test_file)

            print('Xtrain shape:', self.X_train.shape)
            print('X2train shape:', self.X2_train.shape)
            print('Severity train shape:', self.severity_train.shape)
            print('X2test shape:', self.X2_test.shape)
            print('Xtest shape:', self.X_test.shape)
            print('Severity test shape:', self.severity_test.shape)

            self.oneHotEncoderSetup()
            self.Y_train = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_train).reshape(-1, 1)))
            self.Y_test = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_test).reshape(-1, 1)))

            # self.Y_train = to_categorical(self.Y_train, num_classes=2)
            # self.Y_test = to_categorical(self.Y_test, num_classes=2)

            print('Ytrain shape:', self.Y_train.shape)
            print('Ytest shape:', self.Y_test.shape)

            self.stdScaler = MinMaxScaler()
            self.stdScaler.fit(self.X_train)
            self.X_train = self.stdScaler.transform(self.X_train)
            self.stdScaler.fit(self.X_test)
            self.X_test = self.stdScaler.transform(self.X_test)
            self.stdScaler.fit(self.X2_train)
            self.X2_train = self.stdScaler.transform(self.X2_train)
            self.stdScaler.fit(self.X2_test)
            self.X2_test = self.stdScaler.transform(self.X2_test)
            self.stdScaler.fit(self.severity_train)
            self.severity_train = self.stdScaler.transform(self.severity_train)
            self.stdScaler.fit(self.severity_test)
            self.severity_test = self.stdScaler.transform(self.severity_test)

            # self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, test_size=self.opts["eval_size"], random_state=self.opts["seed"],
            #                                                   shuffle=True)

            insize = self.X_train.shape[1]
            insize_meta = self.X2_train.shape[1]
            insize_severity = self.severity_train.shape[1]
            outsize = len(self.Y_train[0])

            """ Create Model """
            # define set of inputs
            inputA = Input(shape=(insize,))
            inputB = Input(shape=(insize_meta,))
            inputC = Input(shape=(insize_severity,))

            # the first branch operates on the first input
            x = Dense(800, activation=self.opts["activation_function"])(inputA)
            x = BatchNormalization()(x)
            x = Dropout(self.opts["dropout_rate"])(x)
            x = Dense(400, activation=self.opts["activation_function"])(x)
            x = Dropout(self.opts["dropout_rate"])(x)
            x = Dense(200, activation=self.opts["activation_function"])(x)
            x = Dropout(self.opts["dropout_rate"])(x)
            x = Dense(100, activation=self.opts["activation_function"])(x)
            x = Dropout(self.opts["dropout_rate"])(x)
            x = Model(inputs=inputA, outputs=x)

            # the second branch opreates on the second input
            y = Dense(64, activation="softmax")(inputB) #32
            y = BatchNormalization()(y)
            y = Dropout(self.opts["dropout_rate"])(y)
            y = Dense(32, activation="softmax")(y) #16
            y = Dropout(self.opts["dropout_rate"])(y)
            y = Dense(16, activation="softmax")(y)
            y = Model(inputs=inputB, outputs=y)

            # The third branch handles the severity scores
            s = Dense(200, activation=self.opts["activation_function"])(inputC)
            s = BatchNormalization()(s)
            s = Dropout(self.opts["dropout_rate"])(s)
            s = Dense(100, activation=self.opts["activation_function"])(s)
            s = Dropout(self.opts["dropout_rate"])(s)
            s = Dense(50, activation=self.opts["activation_function"])(s)
            s = Dropout(self.opts["dropout_rate"])(s)
            s = Dense(25, activation=self.opts["activation_function"])(s)
            s = Dropout(self.opts["dropout_rate"])(s)
            s = Model(inputs=inputC, outputs=s)

            # combine the output of the two branches
            combined = concatenate([x.output, y.output, s.output])
            z = Dense(32, activation="softmax")(combined)
            #z = Dropout(0.2)(z)
            z = Dense(16, activation="softmax")(z)
            z = Dense(outsize, activation='softmax')(z)

            self.model = Model(inputs=[x.input, y.input, s.input], outputs=z)
            self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])  # , metrics.Precision, metrics.Recall])
            #self.model.compile(loss=['categorical_crossentropy', auc_loss], loss_weights = [0.5, 0.5], optimizer=optm, metrics=['accuracy']) #, metrics.Precision, metrics.Recall])
            #self.model.compile(loss=auc_loss, optimizer=optm, metrics=['accuracy'])

            # self.model.summary()

            # TODO: [Maryam]: following is the Old model
            # self.model = Sequential()
            # self.model.add(Dense(insize, input_dim=insize, activation=self.opts["activation_function"]))
            # self.model.add(Dropout(self.opts["dropout_rate"]))
            # self.model.add(Dense(int(insize * 1.2), activation=self.opts["activation_function"]))
            # self.model.add(Dropout(self.opts["dropout_rate"]))
            # self.model.add(Dense(int(insize * 0.6), activation=self.opts["activation_function"]))
            # self.model.add(Dropout(self.opts["dropout_rate"]))
            # self.model.add(Dense(int(insize * 0.3), activation=self.opts["activation_function"]))
            # self.model.add(Dropout(self.opts["dropout_rate"]))
            # self.model.add(Dense(outsize, activation='softmax'))
            # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            #
            #
            plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    def train(self, checkpoint_path, name, save_results=False):
        event_dict_file = str(checkpoint_path) + "/" + str(name) + "_split_onehotdict.json"
        with open(str(event_dict_file), 'w') as outfile:
            json.dump(self.one_hot_dict, outfile)

        with open(checkpoint_path + "/" + name + "_split_model.json", 'w') as f:
            f.write(self.model.to_json())

        ckpt_file = str(checkpoint_path) + "/" + str(name) + "_split_weights.hdf5"
        checkpoint = ModelCheckpoint(ckpt_file, monitor='test_rec_mean', verbose=1, save_best_only=True, mode='max') # test_rec_mean

        sample_weight = class_weight.compute_sample_weight('balanced', self.Y_train)

        factor = 1. / np.sqrt(2)
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='auto',
                                      factor=factor, cooldown=0, min_lr=1e-5, verbose=2)

        classes = np.unique(self.Y_train)
        le = LabelEncoder()
        y_ind = le.fit_transform(self.Y_train.ravel())
        recip_freq = len(self.Y_train) / (len(le.classes_) *
                                     np.bincount(y_ind).astype(np.float64))
        #class_weight = recip_freq[le.transform(classes)]
        #class_weight = np.array([0.4, 0.6])

        print("Class weights : ", class_weight)

        #earlystop = EarlyStopping(patience=100)

        hist = self.model.fit([self.X_train, self.X2_train, self.severity_train], [self.Y_train], sample_weight=sample_weight, #class_weight=class_weight,
                            batch_size=self.opts["n_batch_size"], epochs=self.opts["n_epochs"], shuffle=True,
                            validation_data=([self.X_test, self.X2_test, self.severity_test], [self.Y_test]),
                            callbacks=[self.EvaluationCallback(self.X_test, self.X2_test, self.severity_test, self.Y_test), checkpoint, reduce_lr])

        if save_results:
            results_file = str(checkpoint_path) + "/" + str(name) + "_split_results.json"
            with open(str(results_file), 'w') as outfile:
                json.dump(str(hist.history), outfile)

    # def train(self, checkpoint_path, name, save_results=False):
    #     event_dict_file = str(checkpoint_path) + "/" + str(name) + "_nap_onehotdict.json"
    #     with open(str(event_dict_file), 'w') as outfile:
    #         json.dump(self.one_hot_dict, outfile)
    #
    #     with open(checkpoint_path + "/" + name + "_nap_model.json", 'w') as f:
    #         f.write(self.model.to_json())
    #
    #     ckpt_file = str(checkpoint_path) + "/" + str(name) + "_nap_weights.hdf5"
    #     checkpoint = ModelCheckpoint(ckpt_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    #     # hist = self.model.fit([self.X_train], [self.Y_train], batch_size=self.opts["n_batch_size"], epochs=self.opts["n_epochs"], shuffle=True,
    #     #                  validation_data=([self.X_val], [self.Y_val]),
    #     #                  callbacks=[self.EvaluationCallback(self.X_test, self.Y_test), checkpoint])
    #     hist = self.model.fit([self.X_train], [self.Y_train], batch_size=self.opts["n_batch_size"], epochs=self.opts["n_epochs"], shuffle=True,
    #                      validation_data=([self.X_val], [self.Y_val]))
    #     joblib.dump(self.stdScaler, str(checkpoint_path) + "/" + str(name) + "_nap_stdScaler.pkl")
    #     if save_results:
    #         results_file = str(checkpoint_path) + "/" + str(name) + "_nap_results.json"
    #         with open(str(results_file), 'w') as outfile:
    #             json.dump(str(hist.history), outfile)

    def oneHotEncoderSetup(self):
        """ Events to One Hot"""
        events = np.unique(self.Y_train)

        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(events)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.onehot_encoder.fit(integer_encoded)

        self.one_hot_dict = {}
        for event in events:
            self.one_hot_dict[event] = list(self.onehot_encoder.transform([self.label_encoder.transform([event])])[0])

    def  loadData(self, file):
        # TODO: [Maryam]: Need to enhance this function to include demographic
        #  information and sevirity scores. This will be similar
        #  to SPLIT.loadData() of mortality_NAP project
        x, x2, x3, y  = [], [], [], []
        with open(file) as json_file:
            tss = json.load(json_file)
            for sample in tss:
                if sample["nextEvent"] is not None:
                    x.append(list(itertools.chain(sample["TimedStateSample"][0],
                                                  sample["TimedStateSample"][1],
                                                  sample["TimedStateSample"][2])))
                    y.append(sample["nextEvent"])

                    x2s = list()
                    x2s.append((float(sample["age"])))
                    x2s.append(float(sample["gender"]))
                    x2s.append(float(sample["ethnicity"]))
                    x2.append(x2s)

                    # TODO: [Maryam]: Need to populate 'SEVERITY SCORE' here.
                    # x3s = list()
                    # list(itertools.chain(sample['severity']))
                    # x3s = float(sample["charlson"]) # here severity scores are as list
                    # x3s + float(sample["elixhauser"])
                    # x3s = list()
                    # for s in sample['severity']:
                    #     x3s.append(s)
                    x3.append(sample['severity'])

        return np.array(x), np.array(x2), np.array(x3), np.array(y)

    def setSeed(self):
        seed(self.opts["seed"])
        #set_random_seed(self.opts["seed"])
        set_seed(self.opts["seed"])

    def loadModel(self, path, name):
        with open(path + "/" + name + "_split_model.json", 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(path + "/" + name + "_split_weights.hdf5")
        with open(path + "/" + name + "_split_onehotdict.json", 'r') as f:
            self.one_hot_dict = json.load(f)
        # self.stdScaler = joblib.load(path + "/" + name + "_nap_stdScaler.pkl")

    def intToEvent(self, value):
        one_hot = list(np.eye(len(self.one_hot_dict.keys()))[value])
        for k, v in self.one_hot_dict.items():
            if str(v) == str(one_hot):
                return k

    def predict_test(self, path, name, tss):
        self.loadModel(path=path, name=name)
        for sample in tss:
            y_prob = self.model.predict([self.X_test, self.X2_test, self.severity_test])
            y_pred = np.argmax(y_prob, axis=1)

        return y_pred, y_prob

    def predict(self, tss):
        """
        Predict from a list TimedStateSamples

        :param tss: list<TimedStateSamples>
        :return: tuple (DREAM-NAP output, translated next event)
        """
        if not isinstance(tss, list) or not isinstance(tss[0], TimedStateSample) :
            raise ValueError("Input is not a list with TimedStateSample")

        preds = []
        next_events = []
        for sample in tss:
            features = [list(itertools.chain(sample.export()["TimedStateSample"][0], sample.export()["TimedStateSample"][1], sample.export()["TimedStateSample"][2]))]
            features = self.stdScaler.transform(features)
            pred = np.argmax(self.model.predict(features), axis=1)
            preds.append(pred[0])
            for p in pred:
                next_events.append(self.intToEvent(p))
        return preds, next_events

    """ Callback """
    class EvaluationCallback(Callback):
        def __init__(self, X_test, X2_test, severity_test, Y_test):
            self.X_test = X_test
            self.X2_test = X2_test
            self.severity_test = severity_test
            self.Y_test = Y_test
            self.Y_test_int = np.argmax(self.Y_test, axis=1)

            self.test_accs = []
            self.losses = []

        def on_train_begin(self, logs={}):
            self.test_accs = []
            self.losses = []

        def on_epoch_end(self, epoch, logs={}):
            y_pred = self.model.predict([self.X_test, self.X2_test, self.severity_test])
            y_pred = y_pred.argmax(axis=1)

            test_acc = accuracy_score(self.Y_test_int, y_pred, normalize=True)
            test_loss, _ = self.model.evaluate([self.X_test, self.X2_test, self.severity_test], self.Y_test)

            precision, recall, fscore, _ = precision_recall_fscore_support(self.Y_test_int, y_pred, average='weighted',
                                                                           pos_label=None, warn_for=tuple())
            auc = multiclass_roc_auc_score(self.Y_test_int, y_pred, average="weighted")

            logs['test_acc'] = test_acc
            logs['test_prec_weighted'] = precision
            logs['test_rec_weighted'] = recall
            logs['test_loss'] = test_loss
            logs['test_fscore_weighted'] = fscore
            logs['test_auc_weighted'] = auc

            precision, recall, fscore, support = precision_recall_fscore_support(self.Y_test_int, y_pred,
                                                                                 average='macro', pos_label=None, warn_for=tuple())
            auc = multiclass_roc_auc_score(self.Y_test_int, y_pred, average="macro")
            print('test_prec_mean',precision)
            print('test_rec_mean',recall)
            print('test_fscore_mean',fscore)
            print('test_auc_mean',auc)

            logs['test_prec_mean'] = precision
            logs['test_rec_mean'] = recall
            logs['test_fscore_mean'] = fscore
            logs['test_auc_mean'] = auc