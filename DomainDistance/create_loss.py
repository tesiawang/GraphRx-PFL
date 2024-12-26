import tensorflow as tf


def create_loss(name='ce'):
    """
    loss function must be differentiable
    """
    if name == 'ce':
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    elif name == 'bce':
        return tf.keras.losses.BinaryCrossentropy(from_logits=True) # treat 'y_pred' as logits instead of probabilities
    elif name == 'kl_div':
        return tf.keras.losses.KLDivergence()
    elif name == 'mutual_kl_div':
        return lambda logits1, logits2: 0.5 * tf.keras.losses.KLDivergence()(logits1, logits2) + \
                                        0.5 * tf.keras.losses.KLDivergence()(logits2, logits1)
    elif name == 'mean':
        return lambda logits, domain_id: - tf.reduce_mean((domain_id * 2 - 1) * logits)
    else:
        raise NotImplementedError('Unknown loss name: %s' % name)
