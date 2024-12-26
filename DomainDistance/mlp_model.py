import os

# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import sionna

# ------------- Define classifier model for domain discriminator ------------- #
class MLPDiscriminator(tf.keras.Model):
    def __init__(self, feature_dim, label_dim=None, c=0.01):
        super(MLPDiscriminator, self).__init__()

        self.use_label = (label_dim is not None)  # By default no label
        self.c = c  # Clipping parameter

        # Flatten and feature tf.keras.layers
        self.flatten = tf.keras.layers.Flatten()
        self.linear1_feat = tf.keras.layers.Dense(64, input_shape=(tf.reduce_prod(feature_dim),))

        if self.use_label:
            self.label_dim = label_dim
            self.linear1_label = tf.keras.layers.Dense(64, input_shape=(label_dim,))

        self.relu1 = tf.keras.layers.ReLU()
        self.linear2 = tf.keras.layers.Dense(1)


    def call(self, x, y=None): 
        
        '''
            x: the input image
            y is the label of the oringinal problem, and it is the feature in the domain discrimination problem
        '''
        # Process features
        x = self.flatten(x)
        x = self.linear1_feat(x)

        # Process labels if applicable
        if self.use_label:
            nu = self.label_dim / ((self.label_dim - 1) ** 0.5)  # Normalizing scale
            y = tf.cast(tf.one_hot(y, self.label_dim), tf.float32)
            y = nu * y
            y = self.linear1_label(y)
            x = x + 2 * y  # Label information has more weight

        x = self.relu1(x)
        x = self.linear2(x)
        
        # return the un-activated logits; the binary cross-entropy loss will be applied in the loss function
        return x


    def clip_weights(self):
        """Clips the weights of the model's tf.keras.layers between -c and c."""
        for layer in self.tf.keras.layers:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(tf.clip_by_value(layer.kernel, -self.c, self.c))
            if hasattr(layer, 'bias'):
                layer.bias.assign(tf.clip_by_value(layer.bias, -self.c, self.c))


    def normalize_weights(self):
        """Normalizes weights for tf.keras.layers, unsupported for label tf.keras.layers."""
        if self.use_label:
            raise NotImplementedError("Normalization for label tf.keras.layers not implemented.")

        else:
            norm1 = tf.norm(self.linear1_feat.kernel, ord=2)
            self.linear1_feat.kernel.assign(self.linear1_feat.kernel / norm1)
            self.linear1_feat.bias.assign(self.linear1_feat.bias / norm1)

            norm2 = tf.norm(self.linear2.kernel, ord=2)
            self.linear2.kernel.assign(self.linear2.kernel / norm2)
            self.linear2.bias.assign(self.linear2.bias / (norm1 * norm2))


# if __name__ == '__main__':
#     # Test the model
#     model = MLPDiscriminator((28, 28, 1))
#     predictions = model(tf.random.normal((1, 28, 28, 1)))
#     print(predictions)
#     print(model.summary())