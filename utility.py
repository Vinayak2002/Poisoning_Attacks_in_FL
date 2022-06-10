import os
import random

import cv2
import numpy
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer


def load(image_paths, verbose=-1):
    """
    Expects images for each class in separate directory
    (E.g - all digits in 0 class in the directory named 0).
    :param image_paths: Path to the image
    :param verbose: The number after which to inform the user.
    :return: Tuple of data and labels
    """

    data = list()  # Stores the image data
    labels = list()  # Stores the corresponding labels for the images

    # Iterate over each image path
    for (i, image_path) in enumerate(image_paths):

        # Load the image and extract the class labels
        im_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten()

        # image_path -> "./archive/trainingSample/trainingSample\\0\img_110.jpg".
        # To extract the label, we need to split the path string on the file separator based on os.
        # Here it is \ and split gives list ['add', 'class label', 'image-name'].
        # Access the -2 element of the list will give the label of the image.
        label = image_path.split(os.path.sep)[-2]

        # Scale the Image to [0, 1] and add to list
        data.append(image / 255)
        labels.append(label)

        # Show an update after every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))

    # Return the Data and Labels
    return data, labels


def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    """
    Creates virtual clients and gives them a share of data and labels.

    return: A dictionary with keys as clients names and value as
                data shards - tuple of images and label lists.
    args:
        image_list: a list of numpy arrays of training images
        label_list: a list of binary labels for each image
        num_client: number of federated members (clients)
        initials: the clients' name prefix, e.g, clients_1
    """

    # Client Names List
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # Data is randomly allocated to each client
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    # Data is shared uniformly to clients
    size = len(data) // num_clients
    data_groups = [data[i:i + size] for i in range(0, size * num_clients, size)]

    # The no of data groups created must be equal to the number of clients
    assert (len(data_groups) == len(client_names))

    return {client_names[i]: data_groups[i] for i in range(len(client_names))}


def batch_data(data_group, bs=32, malicious=False):
    """
    Takes in a clients data and create a TensorFlow DataSet (tfds) object from it.
    :param malicious: Whether the given client is malicious or not.
    :param data_group: a data, label constituting a client's data group
    :param bs: batch size
    :return: Tensorflow Dataset object
    """

    # Separate data_group into data and labels lists
    data, label = zip(*data_group)
    if malicious:
        label = poison_data(label)

    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


def poison_data(labels: list):
    """
    Poisons the data by label shuffling.
    :param labels: List of labels to shuffle (poison).
    :return: Poisoned list.
    """
    X = numpy.array(labels)
    numpy.random.shuffle(X)

    return X


def weight_scaling_factor(clients_trn_data, client_name):
    """
    Return the scaling factor for the aggregation model
    :param clients_trn_data: Dictionary containing Client name and data held by that client
    :param client_name: The client for which we need the scaling factor
    :return: Floating Point - scaling factor
    """

    client_names = list(clients_trn_data.keys())

    # Get the batch size
    bs = list(clients_trn_data[client_name])[0][0].shape[0]

    # Calculate the total training data points across clients
    global_count = sum([
        tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names
    ]) * bs

    # Total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() * bs

    return local_count / global_count


def scale_model_weights(weight, scalar):
    """
    Scales the model weights
    :param weight: Given weight of the model
    :param scalar: Scaling factor
    :return: Scaled weight
    """
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    """
    Returns the sum of the listed scaled weights. The is equivalent to scaled avg of the weights.
    :param scaled_weight_list: List - scaled weights
    :return: Float - scaled average of the weights
    """

    avg_grad = list()
    # Average gradient across all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


def test_model(x_test, y_test, model, comm_round, file_name="output.txt"):
    """
    Calculate the accuracy and loss of model mentioned with the given test data and labels.
    :param x_test: Data
    :param y_test: Labels
    :param model: Model that need to be tested.
    :param comm_round: Round number.
    :return: Accuracy and Loss
    """
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # predictions = model.predict(X_test, batch_size=100)
    predictions = model.predict(x_test)
    loss = cce(y_test, predictions)
    acc = accuracy_score(tf.argmax(predictions, axis=1), tf.argmax(y_test, axis=1))
    with open(file_name, 'a') as file:
        file.write('comm_round: {} | global_acc: {:.3%} | global_loss: {}\n'.format(comm_round, acc, loss))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss
