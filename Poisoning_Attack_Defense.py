# Poisoning Attack with defense based on statistical elimination

import tensorflow
from imutils import paths
# from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
from check_updates import check_model
from smlp_model import SimpleMLP
from utility import *

# Path to the MNIST training dataset directory
# img_path = "./archive/trainingSet/trainingSet"
img_path = "./archive/trainingSample/trainingSample"

# Path list for the images using paths object
image_paths = list(paths.list_images(img_path))

# Load the image data and labels
image_list, label_list = load(image_paths, verbose=10000)

# Make the labels binary
lb = LabelBinarizer()
label_list = lb.fit_transform(label_list)

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.1, random_state=42)

# Create clients for FL rounds
clients = create_clients(X_train, y_train, num_clients=10, initial='client')

# Process and Batch the Training Data for each Client
clients_batched = dict()
max_mal_clients = 3
count = 0
for (client_name, data) in clients.items():
    n = random.randint(0, 1)
    if n == 0 or count > max_mal_clients:
        clients_batched[client_name] = batch_data(data)
    else:
        count += 1
        clients_batched[client_name] = batch_data(data, malicious=True)

# Process and Batch the Testing Dataset
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

fl_rounds = 1

# Create Optimizer
learning_rate = 0.01
loss = 'categorical_crossentropy'
metrics = ['accuracy']
# optimizer = tensorflow.keras.optimizers.SGD(lr=learning_rate, decay=(learning_rate / fl_rounds), momentum=0.9)
# optimizer = tensorflow.compat.v1.keras.optimizers.SGD(lr=learning_rate, decay=(learning_rate / fl_rounds),
# momentum=0.9)

# Initialize the Global Model
simple_mlp_global = SimpleMLP()
global_model = simple_mlp_global.build(784, 10)


# Start global training loop
for comm_round in range(fl_rounds):

    # The initial weights of global model will provide the local models with their own local model weights
    global_weights = global_model.get_weights()
    with open("check.txt", 'a') as f1:
        f1.write(str(global_weights) + "\n")
        f1.write(str(type(global_weights))+"\n")
    # A list to store the scaled local model weights after local training
    scaled_local_weight_list = list()

    # Randomize the Client Data - using keys
    client_names = list(clients_batched.keys())
    random.shuffle(client_names)

    # Iterate through each client and create new local model and train it.
    for i, client in enumerate(client_names):
        simple_mlp_local = SimpleMLP()
        local_model = simple_mlp_local.build(784, 10)
        local_model.compile(loss=loss, optimizer='sgd', metrics=metrics)

        # Initialize the Local Model Weight to the Weight of the Global Model
        local_model.set_weights(global_weights)

        # Fit local model with client's local data
        local_model.fit(clients_batched[client], epochs=1, verbose=0)

        # Scale the Local Model Weights and append to list
        scaling_factor = weight_scaling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        if check_model(local_model):
            scaled_local_weight_list.append(scaled_weights)

        # Clear session to free memory after each FL Round
        K.clear_session()

    # Get the average over all the local model weights (simply take the sum of the scaled weights).
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    # Update the Global Model Weights
    global_model.set_weights(average_weights)

    # Test Global Model and print out metrics after each FL Round
    for (X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round, "poisoning_defense_global.txt")

SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(320)
smlp_SGD = SimpleMLP()
SGD_model = smlp_SGD.build(784, 10)

SGD_model.compile(loss=loss, optimizer='sgd', metrics=metrics)

# fit the SGD training data to model
_ = SGD_model.fit(SGD_dataset, epochs=100, verbose=0)

# test the SGD global model and print out metrics
for (X_test, Y_test) in test_batched:
    SGD_acc, SGD_loss = test_model(X_test, Y_test, SGD_model, 1, 'poisoning_defense_no_fl.txt')
