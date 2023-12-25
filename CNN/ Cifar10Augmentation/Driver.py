from Cifar10Augmentation import Cifar10AugmentationCNN,plot_loss


def main():
    cnn = Cifar10AugmentationCNN()
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
    print("==============Savingt Model======================")
    cnn.save_model(model)


if __name__ == '__main__':
    main()
