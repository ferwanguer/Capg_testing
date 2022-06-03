import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import dice_ml
from dice_ml import Dice
import pandas as pd
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', 70)
pd.set_option('display.width', 2000)


class ModelWrapper:
    def __init__(self, tf_model):
        self.tf_model = tf_model

    def predict_proba(self, X):
        X_array = X.to_numpy().T

        Input_array = np.reshape(X_array.T, (-1, 28, 28))

        output_prediction = self.tf_model.predict(Input_array)

        return output_prediction

    def predict(self, X):
        X_array = X.to_numpy().T
        Input_array = np.reshape(X_array.T, (-1, 28, 28))
        output_prediction = self.tf_model.predict(Input_array)
        classification = np.argmax(output_prediction, axis=1)
        print(f'used predict, {X_array.shape}')
        return classification

    def array_to_df_input(self, X_features, labels):
        """Transform the input (array of arrays, it flattens it) and its labels into
        a dataframe that can be introduced in the DiCE library. Returns the
        dataframe with the labels under the column 'target'

         X_features(# samples,# horizontal pixels, # vertical pixels)
         labels(# columns, 1)"""

        reshaped_x = np.reshape(X_features, (X_features.shape[0], -1))
        features_labels = np.append(reshaped_x, labels, axis=1)
        input_database = pd.DataFrame(features_labels)
        input_database.rename(columns={features_labels.shape[1] - 1: 'target'}, inplace=True)

        return input_database


if __name__ == '__main__':
    print(tf.__version__)

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0

    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=1)

    dddd = ModelWrapper(model)

    database = dddd.array_to_df_input(test_images, test_labels[:, None])



    continuous_features = database.drop('target', axis=1).columns.tolist()

    data_dice = dice_ml.Data(dataframe=database, continuous_features=continuous_features, outcome_name='target')
    model_dice = dice_ml.Model(model=dddd, backend='sklearn', model_type='classifier')

    explainer = Dice(data_dice, model_dice, method='genetic')

    n_samples = 1
    input_datapoint = database[4:5].drop('target', axis=1)
    counterfactuals = explainer.generate_counterfactuals(input_datapoint, total_CFs=n_samples,
                                                         desired_class=3)
    test = counterfactuals.cf_examples_list[0].final_cfs_df
    print(test.shape)
    test_predictions = dddd.predict(test)
    counter_df = counterfactuals.visualize_as_dataframe(show_only_changes=False)

    print(test_predictions)

    arahy = np.reshape(test.to_numpy(),(-1,28,28))
    plt.figure()
    plt.imshow(arahy[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()





    print('End')
