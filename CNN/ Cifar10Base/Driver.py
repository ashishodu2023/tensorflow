from Cifar10base import Cifar10BaseCNN
from Cifar10base import plot_loss


def main():
    cnn = Cifar10BaseCNN()
    (x_train, y_train), (x_test, y_test) = cnn.load_data()
    print("==============Shape of the datasets=============")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train, y_train, x_test, y_test = cnn.preprocess(x_train, y_train, x_test, y_test)
    model = cnn.build_model()
    print("==============Model Summary=====================")
    print(model.summary())
    print("==============Training Model=====================")
    history = cnn.train_model(x_train, y_train)
    plot_loss(history)
    print("==============Predictions========================")
    cnn.predictions(model, x_test, y_test)


if __name__ == '__main__':
    main()
