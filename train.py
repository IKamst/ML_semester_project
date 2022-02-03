from matplotlib import pyplot as plt
from tensorflow import config
from tensorflow.keras import models, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import random, cv2
from tensorflow.keras.utils import to_categorical
from load_data import *
import numpy as np

AUG_ROTATION_RANGE = 60
AUG_ZOOM_RANGE = 0
AUG_SHEAR_RANGE = 0

AUG_CHANCE = 0.5


def solve_cudnn_error():
    gpus = config.experimental.list_physical_devices('GPU')
    if gpus:

        try:
            for gpu in gpus:
                config.experimental.set_memory_growth(gpu, True)
            logical_gpus = config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        except RuntimeError as e:
            print(e)


def model_build():
    BaseCnn = MobileNetV2(input_shape=(32, 32, 3), alpha=1.0,  # minimalistic=True,
                          include_top=False, weights='imagenet',  # classes=1,dropout_rate=0.7,
                          )  # 'imagenet' None

    model = models.Sequential()
    model.add(BaseCnn)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def data_generator(dataset, st, ed, batch_size, aug):
    nowinx = st
    while True:
        im_array = []
        lb_array = []
        for i in range(batch_size):
            im_gray = np.array(dataset[nowinx][0], dtype=np.uint8)
            im_bgr = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

            tmp_im_array = np.array(im_bgr)
            tmp_im_array = tmp_im_array / 255

            tmp_im_array = tmp_im_array[np.newaxis, :, :, :]
            lb = dataset[nowinx][1]

            if len(im_array) == 0:
                im_array = tmp_im_array
                lb_array.append(lb)
            else:
                im_array = np.concatenate((im_array, tmp_im_array), axis=0)
                lb_array.append(lb)
            random.shuffle(dataset) if nowinx == ed else nowinx + 1
            print("----dataset shuffled-----") if nowinx == ed else nowinx + 1
            nowinx = st if nowinx == ed else nowinx + 1

        lb_array = to_categorical(np.array(lb_array), 10)

        # augment the data

        if (aug is not None) and (random.random() <= AUG_CHANCE):
            im_array_old = im_array.copy()
            new_array = im_array
            new_array = next(aug.flow(x=new_array, y=None, batch_size=batch_size, shuffle=False,
                                      sample_weight=None, seed=None, save_to_dir=None, save_prefix='',
                                      save_format='png', subset=None))
            im_array = new_array
            '''    
            for i in range(batch_size):
                print("lb  %s"%lb_array[i])
                while(1):
                    cv2.imshow('dst',im_array[i])
                    cv2.imshow('im_array_old',im_array_old[i])

                    if cv2.waitKey(1) == ord('0'):
                        break
                cv2.destroyAllWindows() '''
        yield im_array, lb_array


def plot_accuracy_train_validate(result):
    plt.figure(1)
    plt.plot(result.history['accuracy'])
    plt.plot(result.history['val_accuracy'])
    plt.title('Model accuracy for training set and validation set')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train set', 'Validation set'], loc='upper left')
    plt.show()
    return


# Loop over the test set and for each image use model.predict(image).
# Then check if this prediction is correct and calculate the accuracy over the entire test set.
def determine_accuracy_test_set(test_data, model):
    # TODO Only run after the model has been trained and different hyperparameters have been used to check
    #  performance on the validation set.
    print(len(test_data))
    correct = 0
    for i in range(len(test_data)):
        im_gray = np.array(test_data[i][0], dtype=np.uint8)
        im_bgr = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR) / 255
        yy = list(model.predict(im_bgr.reshape(1, 32, 32, 3))[0])
        num_pre = yy.index(max(yy))
        num_label = test_data[i][1]
        if num_pre == num_label:
            correct += 1
        else:
            print("error_index:", i, "num_pre:", num_pre, "num_label:", num_label)
            '''
            while(1):
                cv2.imshow('im_bgr',im_bgr)
                if cv2.waitKey(1) == ord('0'): # press 0 to close the pic window
                    break
            cv2.destroyAllWindows()
            '''

    accuracy = correct / len(test_data)
    print("Accuracy", accuracy)
    return


def run_CNN():
    solve_cudnn_error()
    aug = ImageDataGenerator(rotation_range=AUG_ROTATION_RANGE, zoom_range=AUG_ZOOM_RANGE, width_shift_range=0,
                             height_shift_range=0, shear_range=AUG_SHEAR_RANGE, horizontal_flip=False,
                             vertical_flip=False,
                             fill_mode="constant", cval=0)

    train_gen = data_generator(train_set, 0, len(train_set) - 1, 30, aug)
    validate_gen = data_generator(test_set, 0, len(validate_set) - 1, 20, None)

    model = model_build()
    model.summary()
    model.compile(optimizer=SGD(learning_rate=0.01),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    reducelrOnplateau = ReduceLROnPlateau(monitor='loss', factor=0.33, patience=7, verbose=1)

    model_file_path = "./model/digit_recognition.hdf5"
    model_checkpointer = ModelCheckpoint(
        filepath=model_file_path,
        monitor='loss',
        verbose=1,
        mode='auto',
        save_best_only=True,
        save_weights_only=False,
    )

    tensorboard = TensorBoard(log_dir='./logs/digit_recognition_logs', update_freq='batch')

    # Fit the model using the training data and check if the performance improves using the validation data.
    his = model.fit_generator(
        generator=train_gen,
        callbacks=[reducelrOnplateau, tensorboard, model_checkpointer],  #
        steps_per_epoch=40,
        validation_data=validate_gen,
        validation_steps=20,
        epochs=100
    )
    return his, model


if __name__ == "__main__":
    img_array, labels = load_data(expand=True, plot=0)
    train_img_array, train_labels, validate_img_array, validate_labels, test_img_array, test_labels = \
        split_train_validate_test(img_array, labels)
    for i in range(len(train_img_array)):
        train_set = list(zip(train_img_array[i], train_labels[i]))
        validate_set = list(zip(validate_img_array[i], validate_labels[i]))
        test_set = list(zip(test_img_array[i], test_labels[i]))
        results, model = run_CNN()
        validation_accuracy = results.history['val_accuracy'][-1]
        print('Validation accuracy:')
        print(validation_accuracy)
        determine_accuracy_test_set(test_set, model)
    plot_accuracy_train_validate(results)

