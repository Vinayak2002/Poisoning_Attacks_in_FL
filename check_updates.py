import tensorflow as tf
from smlp_model import SimpleMLP, Sequential
from Poisoning_Attack_Defense import y_train, X_train, loss, metrics, test_batched
from utility import test_model


def check_model(local_model: Sequential):
    SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(320)
    smlp_SGD = SimpleMLP()
    SGD_model = smlp_SGD.build(784, 10)

    SGD_model.compile(loss=loss, optimizer='sgd', metrics=metrics)

    # fit the SGD training data to model
    _ = SGD_model.fit(SGD_dataset, epochs=100, verbose=0)

    # test the SGD global model and print out metrics
    SGD_acc = None
    SGD_loss = None
    for (X_test, Y_test) in test_batched:
        SGD_acc, SGD_loss = test_model(X_test, Y_test, SGD_model, 1, 'check_model.txt')

    SGD_acc_LM = None
    SGD_loss_LM = None
    SGD_model.set_weights(local_model.get_weights())
    for (X_test, Y_test) in test_batched:
        SGD_acc_LM, SGD_loss_LM = test_model(X_test, Y_test, SGD_model, 1, 'check_model_LM.txt')

    if SGD_acc_LM < (0.95 * SGD_acc) and SGD_loss_LM > (1.05 * SGD_loss):
        return False

    return True
