from DeeperCifar10 import DeeperCifar10CNN
from DeeperCifar10 import plot_loss


def main():
    cnn = DeeperCifar10CNN()
    (x_train, y_train), (x_test, y_test) = cnn.load_data()
    print("==============Shape of the datasets=============")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train, y_train, x_test, y_test = cnn.preprocess(x_train, y_train, x_test, y_test)
    model = cnn.build_model(x_train.shape[1:])
    print("==============Model Summary=====================")
    print(model.summary())
    print("==============Training Model=====================")
    history = cnn.train_model(x_train, y_train,x_test, y_test)
    plot_loss(history)
    print("==============Predictions========================")
    cnn.predictions(model, x_test, y_test)


if __name__ == '__main__':
    main()
