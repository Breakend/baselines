import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, hidden_sizes=(64,64), activation=tf.nn.relu, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.activation = activation

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs

            for i, h in enumerate(self.hidden_sizes):
                x = tf.layers.dense(x, h)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = self.activation(x)

            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, hidden_sizes=(64,64), activation=tf.nn.relu, action_merge_layer=1, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.action_merge_layer = action_merge_layer

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs

            for i, h in enumerate(self.hidden_sizes):
                if i == self.action_merge_layer:
                    x = tf.concat([x, action], axis=-1)
                x = tf.layers.dense(x, h)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = self.activation(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
