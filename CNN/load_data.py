import numpy as np
import cv2


def load_data(expand=False,plot=False):  # load mfeat-pix.txt as opencv' format
    with open("mfeat-pix.txt", "r") as f: 
        data = f.read() 

    data = data.replace(" ", "")
    data = list(data.split("\n"))
    img_array = []
    for im in range(2000):
        img = []
        for i in range(16):
            row = []
            for j in range(15):
                row.append(int(data[im][i*15+j]))
            img.append(row)
        if expand:  # expand the img from 16*15 to 32*32
            img = cv2.copyMakeBorder(np.array(img,dtype=np.uint8), 8, 8, 8, 9, cv2.BORDER_CONSTANT, value=0)
        img_array.append(img)  
    img_array = (np.array(img_array, dtype=np.uint8)/6)*255
    
    labels = []
    for i in range(10):
        labels = labels+[i for _ in range(200)]
    
    if plot:
        for im in range(2000):  # plot the images one by one
            print(np.shape(img_array[im]))
            while 1:

                cv2.imshow('img', img_array[im])
                if cv2.waitKey(1) == ord('0'):
                    break
            cv2.destroyAllWindows()
    return img_array, np.array(labels)


# Split the data into training, validation and testing set.
def split_train_validate_test(img_array, labels):
    training_images = []
    training_labels = []
    validation_images = []
    validation_labels = []
    testing_images = []
    testing_labels = []

    train_img_array = img_array[20:100]
    train_labels = labels[20:100]
    validate_img_array = img_array[0:20]
    validate_labels = labels[0:20]
    test_img_array = img_array[100:200]
    test_labels = labels[100:200]
    for i in range(200, 2000, 200):
        train_img_array = np.concatenate((train_img_array,img_array[i+20:i+100]),axis=0)
        train_labels = np.concatenate((train_labels,labels[i+20:i+100]),axis=0)
        validate_img_array = np.concatenate((validate_img_array,img_array[i:i+20]),axis=0)
        validate_labels = np.concatenate((validate_labels,labels[i:i+20]),axis=0)
        test_img_array = np.concatenate((test_img_array,img_array[i+100:i+200]),axis=0)
        test_labels = np.concatenate((test_labels,labels[i+100:i+200]),axis=0)
    training_images.append(train_img_array)
    training_labels.append(train_labels)
    validation_images.append(validate_img_array)
    validation_labels.append(validate_labels)
    testing_images.append(test_img_array)
    testing_labels.append(test_labels)
    for fold in range(1, 5):
        train_img_array = img_array[0:fold*20]
        train_labels = labels[0:fold*20]
        validate_img_array = img_array[fold*20:(fold+1)*20]
        validate_labels = labels[fold*20:(fold+1)*20]
        test_img_array = img_array[100:200]
        test_labels = labels[100:200]
        if not fold == 4:
            train_img_array = np.concatenate((train_img_array, img_array[(fold+1)*20:100]), axis=0)
            train_labels = np.concatenate((train_labels, labels[(fold+1)*20:100]), axis=0)
        for i in range(200, 2000, 200):
            train_img_array = np.concatenate((train_img_array, img_array[i:i+fold*20]), axis=0)
            train_labels = np.concatenate((train_labels, labels[i:i+fold*20]), axis=0)
            validate_img_array = np.concatenate((validate_img_array, img_array[i+fold*20:i+(fold+1)*20]), axis=0)
            validate_labels = np.concatenate((validate_labels, labels[i+fold*20:i+(fold+1)*20]), axis=0)
            if not fold == 4:
                train_img_array = np.concatenate((train_img_array, img_array[i+(fold+1)*20:i+100]), axis=0)
                train_labels = np.concatenate((train_labels, labels[i+(fold+1)*20:i+100]), axis=0)
            test_img_array = np.concatenate((test_img_array, img_array[i+100:i+200]), axis=0)
            test_labels = np.concatenate((test_labels, labels[i+100:i+200]), axis=0)
        training_images.append(train_img_array)
        training_labels.append(train_labels)
        validation_images.append(validate_img_array)
        validation_labels.append(validate_labels)
        testing_images.append(test_img_array)
        testing_labels.append(test_labels)
    return training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels
