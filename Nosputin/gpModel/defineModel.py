import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as tfkl

import numpy as np
import streamlit as st

from abc import *

class NosputinModel(metaclass=ABCMeta):
    @abstractmethod
    def train_model(self, norm=False):
        pass
    
    @abstractmethod
    def build_model_structure(self):
        pass
    
    @abstractmethod
    def get_preds(self):
        pass
    
class BaseNosputinModel(NosputinModel):
    def __init__(self, x, y, noises, x_t, y_t, noises_t):
        self.epoches = 100
        self.input_dim = 248
        self.batch_size = 10
        
        self.np_x = np.array(x, dtype=np.float64)
        self.np_y = np.array(y, dtype=np.float64).reshape(len(y),1)
        self.np_noises = np.array(noises, dtype=np.float64).reshape(len(y),1)
        self.xdim = len(self.np_x[0])
        
        st.write("Feature dimension of company : {}".format(self.xdim))
        st.write("dim(X):{} , dim(Y):{}".format(self.np_x.shape, self.np_y.shape))
        self.np_x_t = np.array(x_t, dtype=np.float64)
        self.np_y_t = np.array(y_t).reshape(len(y_t),1)
        self.np_noises_t = np.array(noises_t, dtype=np.float64).reshape(len(y_t),1)

    def build_model_structure(self):
        pass
    
    def get_preds(self):
        train_preds = self.model.predict(self.np_x)
        test_preds = self.model.predict(self.np_x_t)
        train_preds = train_preds.transpose()[0]
        test_preds = test_preds.transpose()[0]
        return train_preds, test_preds
        
    def train_model(self, norm=False):
        self.model.compile(optimizer=self.opt, loss=self.loss)
        train_y = np.concatenate((self.np_y, self.np_noises),axis=1)
        test_y = np.concatenate((self.np_y_t, self.np_noises_t),axis=1)
        self.model.fit(x=self.np_x, y=train_y,
                      shuffle=True,
                      epochs=self.epoches,
                      batch_size=self.batch_size,
                      validation_data=(self.np_x_t, test_y))
        
    def save_model(self, name="ns_model"):
        pass
    
    def load_model(self, name="ns_model"):
        pass
    
        
class BayesianNNNosputinModel(BaseNosputinModel):
    def __init__(self, x, y, noises, x_t, y_t, noises_t):
        super().__init__(x, y, noises, x_t, y_t, noises_t)
        
        self.model = tf.keras.Sequential()
        self.opt = tf.optimizers.Adam(learning_rate=0.0001)
        self.loss = NosputinSquareLoss()
        self.build_model_structure()
        
    def build_model_structure(self):
        def add_batch_norm(model):
            #model.add(tfkl.BatchNormalization())
            model.add(tfkl.Activation(tf.nn.leaky_relu))
        
        self.model.add(tfkl.InputLayer(input_shape=[self.input_dim]))
        
        self.model.add(tfkl.Dense(1000, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(500, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(100, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(2,activation=None))
        
    def train_model(self, norm=False):
        super().train_model(norm=norm)


class NaiveNNNosputinModel(BaseNosputinModel):
    def __init__(self, x, y, noises, x_t, y_t, noises_t):
        super().__init__(x, y, noises, x_t, y_t, noises_t)
        
        self.model = tf.keras.Sequential()
        self.opt = tf.optimizers.Adam(learning_rate=0.0001)
        self.loss = NosputinNaiveSquareLoss()
        self.build_model_structure()
        
    def build_model_structure(self):
        def add_batch_norm(model):
            #model.add(tfkl.BatchNormalization())
            model.add(tfkl.Activation(tf.nn.leaky_relu))
        
        self.model.add(tfkl.InputLayer(input_shape=[self.input_dim]))
        
        self.model.add(tfkl.Dense(1000, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(500, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(100, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(1,activation=None))
        
    def train_model(self, norm=False):
        super().train_model(norm=norm)

        
class MassiveNNNosputinModel(BaseNosputinModel):
    def __init__(self, x, y, noises, x_t, y_t, noises_t):
        super().__init__(x, y, noises, x_t, y_t, noises_t)
        
        self.model = tf.keras.Sequential()
        self.opt = tf.optimizers.Adam(learning_rate=0.0004)
        self.loss = NosputinNaiveAbsLoss()
        self.build_model_structure()
        
    def build_model_structure(self):
        def add_batch_norm(model):
            model.add(tfkl.BatchNormalization())
            model.add(tfkl.Activation(tf.nn.leaky_relu))
        
        self.model.add(tfkl.InputLayer(input_shape=[self.input_dim]))
        
        self.model.add(tfkl.Dense(5000, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(3000, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(500, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(100, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(1,activation=None))
        
    def train_model(self, norm=False):
        super().train_model(norm=norm)

        
class MassiveBayesianNNNosputinModel(BaseNosputinModel):
    def __init__(self, x, y, noises, x_t, y_t, noises_t):
        super().__init__(x, y, noises, x_t, y_t, noises_t)
        
        self.batch_size = 50
        self.model = tf.keras.Sequential()
        self.opt = tf.optimizers.Adam(learning_rate=0.0001)
        self.loss = NosputinKLDLoss()
        self.build_model_structure()
        
    def build_model_structure(self):
        def add_batch_norm(model):
            #model.add(tfkl.BatchNormalization())
            model.add(tfkl.Activation(tf.nn.leaky_relu))
        
        self.model.add(tfkl.InputLayer(input_shape=[self.input_dim]))
        
        self.model.add(tfkl.Dense(5000, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(3000, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(500, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(100, activation=None))
        add_batch_norm(self.model)
        
        self.model.add(tfkl.Dense(2,activation=None))
        
    def train_model(self, norm=False):
        super().train_model(norm=norm)

        
class NosputinNaiveSquareLoss(tf.keras.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        return (y_true[0] - y_pred)**2

        
class NosputinNaiveSquareLoss(tf.keras.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        return (y_true[0] - y_pred)**2

    
class NosputinKLDLoss(tf.keras.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        """
        y_pred is a tuple consisting with (value, varaince).
        """
        true_normal = tfp.distributions.Normal(y_true[0], tf.math.abs(y_true[1]))
        pred_normal = tfp.distributions.Normal(y_pred[0], tf.math.abs(y_pred[1]))
        return tfp.distributions.kl_divergence(true_normal, pred_normal)
    
    
class NosputinSquareLoss(tf.keras.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        """
        y_pred is a tuple consisting with (value, varaince).
        """
        return tf.math.square(y_true - y_pred)

    
class NosputinNaiveAbsLoss(tf.keras.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        """
        y_pred is a tuple consisting with (value, varaince).
        """
        return tf.math.abs(y_true[0] - y_pred)