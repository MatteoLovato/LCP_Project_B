import mesa_web as mw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import re

from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import layers, losses, metrics
from keras.models import Model
from scipy.interpolate import UnivariateSpline

## Autoencoder class definition
class Autoencoder(Model):
    def __init__(self, encoder_neurons, decoder_neurons, encoder_activations, decoder_activations):
        ## Input check
        if not len(encoder_neurons) == len(encoder_activations):
            raise ValueError('The vector of neuron numbers for the encoder should be the same size as the activations')
        if not len(decoder_neurons) == len(decoder_activations):
            raise ValueError('The vector of neuron numbers for the decoder should be the same size as the activations')

        input_shape = keras.Input(shape=(decoder_neurons[-1],))
        encoded = layers.Dense(encoder_neurons[0], activation=encoder_activations[0])(input_shape)
        for i in range(1, len(encoder_neurons)):
            encoded = layers.Dense(encoder_neurons[i], activation=encoder_activations[i])(encoded)

        ## Define the decoder
        decoded = layers.Dense(decoder_neurons[0], activation=decoder_activations[0])(encoded)
        for i in range(1, len(decoder_neurons)):
            decoded = layers.Dense(decoder_neurons[i], activation=decoder_activations[i])(decoded)
        
        super().__init__(input_shape, decoded)

### Import parameters
cwd = os.getcwd()
dir_names = ['MESA-Web_M07_Z00001', 'MESA-Web_M07_Z002','MESA-Web_M10_Z002','MESA-Web_M10_Z0001', 'MESA-Web_M10_Z00001','MESA-Web_M15_Z0001', 'MESA-Web_M15_Z00001', 'MESA-Web_M30_Z00001', 'MESA-Web_M30_Z002','MESA-Web_M50_Z00001', 'MESA-Web_M50_Z002', 'MESA-Web_M50_Z001', 'MESA-Web_M5_Z002', 'MESA-Web_M5_Z0001', 'MESA-Web_M1_Z00001', 'MESA-Web_M1_Z0001']  # Add more folder names here
column_filter = ['mass','radius', 'initial_mass', 'initial_z', 'star_age', 'logRho','logT','Teff','energy','photosphere_L', 'photosphere_r', 'star_mass','h1','he3','he4']
column_filter_train = ['mass','radius', 'logRho','logT','energy','h1','he3','he4']
n_points = 50
r = np.linspace(0, 1, n_points)

def extract_number(filename):
    match = re.search(r'\d+', filename)  # Find the sequence of digits
    return int(match.group()) if match else float('inf')

linear_train_df = pd.DataFrame()

for i, dir_name in enumerate(dir_names):
    print(f"####\t\tIMPORTING DATA FROM FOLDER {dir_name}\t\t####")
    dir_path = os.path.join(cwd, 'StellarTracks', dir_name)
    filenames = [filename for filename in os.listdir(dir_path) if re.fullmatch(r'profile[0-9]+\.data', filename)]
    filenames = sorted(filenames, key=extract_number)  # Sort the elements according to the number in the name

    for j, filename in enumerate(filenames):
        print(f"####\t\t\tIMPORTING FILE {filename}\t\t\t####")
        file_path = os.path.join(dir_path, filename)
        data = mw.read_profile(file_path)
        profile_df = pd.DataFrame(data)
        filtered_profile_df = profile_df[column_filter].copy()
        train_filtered_profile_df = profile_df[column_filter_train].copy()

        # Normalization process
        tot_radius = filtered_profile_df['photosphere_r']
        norm_radius = (filtered_profile_df['radius'] - filtered_profile_df['radius'].min()) / (tot_radius - filtered_profile_df['radius'].min())

        norm_radius = np.asarray(norm_radius.T)
        log_rho = np.asarray(train_filtered_profile_df['logRho'].T)
        int_log_rho = UnivariateSpline(norm_radius, log_rho, k=2, s=0)(r)

        train_df = pd.DataFrame(data=int_log_rho.T, columns=[f"log_rho_{i}_{j}"])  # in the format _indexOfFolder_IndexOfProfile

        if linear_train_df.empty:
            linear_train_df = train_df
        else:
            linear_train_df = pd.concat([linear_train_df, train_df], axis=1)

print(linear_train_df.shape)
print(linear_train_df)

x_train, x_test = train_test_split(linear_train_df.T, test_size=0.2)
x_train = x_train.values
x_test = x_test.values
print(x_train.shape)
print(x_test.shape)
print(x_train)
print(x_test)

### Autoencoder training
for latent_dim in range(1, 5):
    encoder_neurons = [256, 128, 64, latent_dim]  # Added more hidden layers
    decoder_neurons = encoder_neurons[:-1][::-1]  # Same inverted values for the hidden layers
    decoder_neurons.append(n_points)  # Last value should be original size
    encoder_activations = ['relu'] * len(encoder_neurons)  # With relu better learning
    decoder_activations = ['relu'] * (len(decoder_neurons) - 1)  # Relu for better learning
    decoder_activations.append('sigmoid')  # At least the last value should be sigmoid

    autoencoder = Autoencoder(
        encoder_neurons=encoder_neurons,
        decoder_neurons=decoder_neurons,
        encoder_activations=encoder_activations,
        decoder_activations=decoder_activations
    )

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=[metrics.MeanSquaredError()])

    history = autoencoder.fit(x_train, x_train, epochs=150, shuffle=True, validation_data=(x_test, x_test))

    # Plot training and validation loss
    file_save_dir = os.path.join(os.getcwd(), "Graphs")
    os.makedirs(file_save_dir, exist_ok=True)
    file_save_path = os.path.join(file_save_dir, f"TrainValLoss_dim_{latent_dim}.png")
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f'Training Loss VS Validation Loss - Latent Dim = {latent_dim}')
    plt.legend()
    plt.savefig(file_save_path)
    plt.close()

    decoded_imgs = autoencoder.predict(x_test)

    # Display original and reconstructed signals
    n = min(10, len(x_test))
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(r, x_test[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.plot(r, decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
