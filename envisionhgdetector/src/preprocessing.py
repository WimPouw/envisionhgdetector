class BasicPreprocessing(layers.Layer):
    """
    Basic preprocessing - adds gaussian noise during training.
    """
    def __init__(self, noise_stddev: float = 0.025, **kwargs):
        super().__init__(**kwargs)
        self.noise_stddev = noise_stddev
        
    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.noise_stddev,
                dtype=tf.float32
            )
            inputs = inputs + noise
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config["noise_stddev"] = self.noise_stddev
        return config


class EnhancedPreprocessing(layers.Layer):
    """
    Enhanced preprocessing with derivatives and augmentation.
    """
    def __init__(
        self,
        noise_stddev: float = 0.005,
        jitter_sigma: float = 0.001,
        scale_range: Tuple[float, float] = (0.995, 1.005),
        drop_prob: float = 0.01,
        **kwargs
    ):
        super(EnhancedPreprocessing, self).__init__(**kwargs)
        self.noise_stddev = noise_stddev
        self.jitter_sigma = jitter_sigma
        self.scale_range = scale_range
        self.drop_prob = drop_prob

    def call(self, inputs, training=None):
        x = inputs
        
        # Center features
        x = x - tf.reduce_mean(x, axis=-2, keepdims=True)
        
        # Compute derivatives
        t_deriv = x[:, 1:] - x[:, :-1]
        t_deriv = tf.pad(t_deriv, [[0, 0], [1, 0], [0, 0]])
        
        t_deriv_2 = t_deriv[:, 1:] - t_deriv[:, :-1]
        t_deriv_2 = tf.pad(t_deriv_2, [[0, 0], [1, 0], [0, 0]])
        
        # Concatenate features with derivatives
        x = tf.concat([x, t_deriv, t_deriv_2], axis=-1)
        
        if training:
            # Add noise
            x = x + tf.random.normal(tf.shape(x), stddev=self.noise_stddev)
            
            # Random scaling
            scale = tf.random.uniform([], self.scale_range[0], self.scale_range[1])
            x = x * scale
            
            # Random frame drop
            mask = tf.cast(tf.random.uniform(tf.shape(x)[:2]) > self.drop_prob, x.dtype)
            x = x * mask[:, :, tf.newaxis]
        
        # Normalize
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True) + 1e-8
        x = (x - mean) / std
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'noise_stddev': self.noise_stddev,
            'jitter_sigma': self.jitter_sigma,
            'scale_range': self.scale_range,
            'drop_prob': self.drop_prob
        })
        return config

