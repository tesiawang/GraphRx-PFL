import os
# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def create_metric(name='acc'):
    """
    metric function can be any function with scalar output.
    """
    if name == 'acc':
        return lambda logits, target: tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.cast(target,tf.bool)), tf.float32))
    elif name == 'err':
        return lambda logits, target: tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(logits, axis=1), tf.cast(target,tf.bool)), tf.float32))
    elif name == 'bacc': # Binary accuracy
        return lambda logits, target: tf.reduce_mean(tf.cast(tf.equal(tf.greater_equal(logits, 0), tf.cast(target,tf.bool)), tf.float32))
    elif name == 'berr': # Binary error
        return lambda logits, target: tf.reduce_mean(tf.cast(tf.not_equal(tf.greater_equal(logits, 0), tf.cast(target,tf.bool)), tf.float32))
    elif name == 'H_delta_H':
        return lambda logits1, logits2: tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(logits1, axis=1), tf.argmax(logits2, axis=1)), tf.float32))
    elif name == 'mean':
        return lambda logits, domain_id: tf.reduce_mean((domain_id * 2 - 1) * logits)
    else:
        raise NotImplementedError('Unknown metric name: %s' % name)


if __name__ == '__main__':
    # Test the metric
    import numpy as np
    import tensorflow as tf

    # Create some fake data
    logits = tf.random.normal([5,1])
    target = tf.ones_like(logits)
    # print(logits)
    # print(target)

    # Create 'Binary Accuracy'
    metric = create_metric('bacc')
    print(metric(logits, target))

    # Create 'Binary Error'
    metric = create_metric('berr')
    print(metric(logits, target))