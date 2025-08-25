import tensorflow as tf

def build_autoencoder(input_dim, latent_dim=16):
    inputs = tf.keras.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(64, activation='relu')(inputs)
    encoded = tf.keras.layers.Dense(latent_dim, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    outputs = tf.keras.layers.Dense(input_dim)(decoded)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
