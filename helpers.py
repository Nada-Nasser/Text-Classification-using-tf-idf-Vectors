from sklearn.model_selection import KFold
from tensorflow.python.keras import datasets, layers, models, optimizers,Input
import matplotlib.pyplot as plt
import keras
from numpy import mean


# normalize data by divide each pixel in each image by 255(the max value for any pixel in the images)
def normalize_image_pixel(images_data):
    images_data = images_data.astype('float32')
    return images_data / 255.0

'''
def prepare_training_and_testing_data():
    # All images have the same square size of 28×28 pixels.
    # the images are grayscale. ->
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

    # reshape dataset to add the number of color channels 
    # (color channel  = 1, because the images are grayscale)
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    train_x, test_x = normalize_image_pixel(train_x), normalize_image_pixel(test_x)

    return (train_x, train_y), (test_x, test_y)
'''


def create_sequential_model(n):
    model = models.Sequential()  # define a Sequential model

    # add n convolutional layers and n max pool layers
    for i in range(n):
        # adding the convolution layer
        # is a 2d layer with shape (3,3)
        # use the activation function rectified linear activation unit (ReLU)
        # this layer summarize the presence of features in an input image
        # results the down sampled feature maps to be the input for next layer
        # each feature map contain the precise position of features in the input image.
        model.add(layers.Conv1D(23, (3, 3), activation="relu", input_shape=(1500, 34933)))

        # add pooling layer after the convolution layer
        # pooling layer create a new set of the same number of pooled feature maps.
        # the pooled feature map size is less than  the input maps.
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten layer: does not change the number of pooled feature map
    # maps is converted to 1d lists (to be the input to the next Dense Layer)
    # Dense layer accept only 1D lists as an input
    model.add(layers.Flatten())

    # It's the only actual layer in the network that is connected to all previous layers.
    model.add(layers.Dense(100, activation='relu'))

    # adding the output layer with 10 nodes (0-9 classes)
    model.add(layers.Dense(10, activation='softmax'))

    # creat a gradient decent optimizer with learning rate equals 0.01
    opt = optimizers.SGD(lr=0.003)
    # compile and build the model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# evaluate a model using k-fold cross-validation
def apply_cross_validation_and_evaluate(data_X, data_Y, nkfold, arc_number):
    scores, histories = list(), list()
    n = 1
    bestAccuracy = 0

    cv = KFold(n_splits=nkfold, shuffle=True, random_state=1)
    # cv is a kfolds cross validator , which is split into k folds
    # each fold splits data into train/test data indices
    # cv shuffles data in our case
    for train_ix, test_ix in cv.split(data_X):
        # select rows for train and test
        trainX, trainY, testX, testY = data_X[train_ix], data_Y[train_ix], data_X[test_ix], data_Y[test_ix]

        # fit model
        model = create_sequential_model(arc_number)  # compile and build CNN model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY))

        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print("accuracy for kfold #", n, 'is ' , (acc * 100.0), '\n\n')
        n += 1  # tells us which kfold we are at

        if acc*100 > bestAccuracy:
            bestAccuracy = acc*100
            bestModel = model

        # stores scores and history
        scores.append(acc)
        histories.append(history)

    print("best accuracy achieved by the cross validation is:" , bestAccuracy)
    return scores, histories, bestModel


def plt_history(histories):
    print("cost plot:")
    for i in range(len(histories)):
        plt.plot(histories[i].history['loss'], color='blue', label='train')
    plt.legend(loc='lower right')
    plt.show()

    print("accuracy plot:")
    for i in range(len(histories)):
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
    plt.legend(loc='lower right')
    plt.show()


def accuracy_summary(scores):
    print('mean Accuracy of all models:', mean(scores) * 100)
