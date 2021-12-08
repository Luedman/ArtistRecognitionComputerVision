import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

# Settings
epochs = 100
batch_size = 16
image_size = (128, 128)
input_shape = image_size + (3,)
seed = 1
test_split = 0.2
class_labels = os.listdir("./data/Art500k/SelectedArtistsSmall")


# Load the data set
def load_data(directory: str, test_split: float = 0.1, preprocess: str = None) -> (tf.data.Dataset, tf.data.Dataset):
    tf_dataset = tf.keras.preprocessing.image_dataset_from_directory(directory,
                                                                     label_mode="categorical",
                                                                     labels='inferred',
                                                                     image_size=image_size,
                                                                     batch_size=batch_size,
                                                                     shuffle=True,
                                                                     class_names=class_labels)

    tf_dataset = tf_dataset.map(lambda x, y: (x / 255, y), num_parallel_calls=tf.data.AUTOTUNE)

    if preprocess.lower() == 'resnet':
        tf_dataset = tf_dataset.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y))

    tf_dataset = tf_dataset.apply(tf.data.experimental.ignore_errors())
    data_set_size = 0
    for _ in tf_dataset:
        data_set_size += 1

    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    split = int(test_split * data_set_size)
    test_set = tf_dataset.take(split)

    training_set = tf_dataset.skip(split)

    data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical", seed=seed),
                                             layers.RandomRotation(0.2, seed=seed)])

    training_set_augmented = training_set.map(lambda x, y: (data_augmentation(x, training=True), y),
                                              num_parallel_calls=tf.data.AUTOTUNE)

    training_set = training_set_augmented.concatenate(training_set_augmented)

    print(f"Dataset loaded. size: {data_set_size}")
    return training_set, test_set


# Create the model, ResNet50 based on image net
def create_model() -> tf.keras.models:
    input_layer = tf.keras.layers.Input(shape=input_shape)
    resnet50_imagenet_model = tf.keras.applications.resnet.ResNet50(include_top=False,
                                                                    weights='imagenet',
                                                                    input_shape=input_shape,
                                                                    pooling='max')

    # for layer in resnet50_imagenet_model.layers[:141]:
    #    layer.trainable = False

    x = resnet50_imagenet_model(input_layer)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(25, activation="softmax")(x)

    model = tf.keras.Model(inputs=[input_layer], outputs=[x])

    adam = tf.keras.optimizers.Adam(learning_rate=1e-6, clipvalue=0.5)

    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Initialize the callbacks used for training
def get_callbacks(label):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./tensorboard_logs/{label}",
                                                          histogram_freq=1,
                                                          write_images=True,
                                                          write_graph=True)
    model_save_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./models/model_" + label,
                                                             save_weights_only=False,
                                                             save_best_only=True),

    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                             patience=10, verbose=1, min_lr=1e-8)

    return tensorboard_callback, model_save_callback, reduce_on_plateau


# Train a newly compiled model
def train_new_model(model: tf.keras.models, training_set: tf.data.Dataset,
                    test_set: tf.data.Dataset, label: str) -> dict:
    model.summary()

    tensorboard_callback, model_save_callback, reduce_on_plateau = get_callbacks(label)

    history = model.fit(training_set, epochs=epochs,
                        callbacks=[tensorboard_callback, model_save_callback, reduce_on_plateau],
                        validation_data=test_set,
                        shuffle=True)
    model.save("./models/model_" + label + "_final")
    return history


# Load and train an existing model
def train_existing_model(model_name: str, epoch_start: int, no_epochs: int,
                         training_set: tf.data.Dataset, test_set: tf.data.Dataset,
                         new_label: str) -> dict:
    model = tf.keras.models.load_model(f'./models/model_{model_name}')
    model.summary()

    tensorboard_callback, model_save_callback, reduce_on_plateau = get_callbacks(new_label)

    history = model.fit(training_set,
                        initial_epoch=epoch_start,
                        epochs=epoch_start + no_epochs,
                        callbacks=[tensorboard_callback, model_save_callback, reduce_on_plateau],
                        validation_data=test_set,
                        shuffle=True)

    return history


# Evaluate a trained model, compute the loss and accuracy on the test/training set
# Create confusion matrices on a class level as well as on the complete datasets
def evaluate_model(model_name: str, training_set: tf.data.Dataset, test_set: tf.data.Dataset):
    model = tf.keras.models.load_model(f'./models/model_{model_name}')
    model.summary()

    eval_dataset(model, training_set, test_set)

    eval_class_confusion_matrix(model, training_set, "training")
    eval_class_confusion_matrix(model, test_set, "test")

    eval_confusion_matrix(model, training_set, "training")
    eval_confusion_matrix(model, test_set, "test")
    return


# Compute the loss and accuracy on the test/training set
def eval_dataset(model: tf.keras.models, training_set: tf.data.Dataset, test_set: tf.data.Dataset) -> None:
    print("Evaluate dataset")
    evaluation_df = pd.DataFrame(columns=['loss', 'accuracy'])

    training_evaluation = model.evaluate(training_set, return_dict=True)
    evaluation_df = evaluation_df.append(pd.DataFrame(data=training_evaluation, index=["training set"]))

    test_evaluation = model.evaluate(test_set, return_dict=True)
    evaluation_df = evaluation_df.append(pd.DataFrame(data=test_evaluation, index=["test set"]))

    evaluation_df.to_csv("./evaluation/evaluation.csv")
    return


# Create confusion matrices per class
def eval_class_confusion_matrix(model: tf.keras.models, data_set: tf.data.Dataset, label: str):
    print(f"Generating class confusion matrix: {label}")
    y_hat = model.predict(data_set)
    y = np.concatenate([y for x, y in data_set], axis=0)

    metric = tfa.metrics.MultiLabelConfusionMatrix(num_classes=25)
    metric.update_state(y, y_hat)
    result = metric.result()

    figure = plt.figure(1)
    figure.set_size_inches(12, 8)
    figure.suptitle(f"Class wise confusion: {label} set")
    for i in range(25):
        heatmap = result.numpy()[i]
        sum_heatmap = sum([sum(x) for x in heatmap])
        ax = figure.add_subplot(5, 5, i + 1)
        plt.imshow(heatmap / sum_heatmap, cmap="Blues")
        for k in [0, 1]:
            for j in [0, 1]:
                text = ax.text(k, j, "%.01f%%" % (heatmap[k, j] / sum_heatmap * 100), ha="center", va="center",
                               color="0.8")
        plt.axis('off')
        plt.title(class_labels[i])
    plt.tight_layout()
    plt.savefig(f"./evaluation/class_wise_confusion_{label}.png", dpi=400)
    plt.close()

    return


def eval_confusion_matrix(model: tf.keras.models, data_set: tf.data.Dataset, label: str):
    print(f"Generating confusion matrix: {label}")
    y_predictions = np.array([])
    y_true_label = np.array([])
    for x, y in data_set:
        y_predictions = np.concatenate([y_predictions, np.argmax(model.predict(x), axis=-1)])
        y_true_label = np.concatenate([y_true_label, np.argmax(y.numpy(), axis=-1)])

    confusion = tf.math.confusion_matrix(labels=y_predictions, predictions=y_true_label, num_classes=25)

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    im = ax.imshow(confusion, cmap="Blues")

    ax.set_xticks(np.arange(len(class_labels)), labels=class_labels, rotation="vertical")
    ax.set_yticks(np.arange(len(class_labels)), labels=class_labels)

    ax.set_title(f"Confusion matrix heatmap {label}")
    fig.tight_layout()
    plt.savefig(f"./evaluation/confusion_matrix_{label}.png", dpi=400)
    plt.close()

    return


# Check whether tensorflow detects any GPUs
def tensorflow_gpu_check():
    print(f"Tensorflow Version: {tf.__version__}")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs found")


if __name__ == "__main__":
    tensorflow_gpu_check()
    list_of_arguments = sys.argv

    training_set, test_set = load_data("./data/Art500k/SelectedArtistsSmall",
                                       test_split=test_split, preprocess='resnet')
    if len(list_of_arguments) == 0:
        main_arg = "create"
    else:
        main_arg = list_of_arguments[1]

    if main_arg == "create":
        model = create_model()
        train_new_model(model, training_set, test_set, "resnet_7")
    elif main_arg == "load":
        train_existing_model(model_name='resnet_7', epoch_start=100, no_epochs=50,
                             training_set=training_set, test_set=test_set, new_label='resnet_7')
    elif main_arg == "eval":
        evaluate_model(model_name='resnet_7', training_set=training_set, test_set=test_set)
