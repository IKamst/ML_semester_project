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


# Split the data into training, validation en testing set.
# The first 50 images of each digit are for training, the next 50 for testing
# and then 100 for testing.
# TODO randomise this? and how many for each set?
# 80 train, 20 validation -> 5 fold cross validation
def split_train_validate_test(img_array, labels):
    train_img_array=img_array[0:50]
    train_labels=labels[0:50]
    validate_img_array=img_array[50:100]
    validate_labels=labels[50:100]
    test_img_array=img_array[100:200]
    test_labels=labels[100:200]
    for i in range(200,2000,200):
        train_img_array=np.concatenate((train_img_array,img_array[i:i+50]),axis=0)
        train_labels=np.concatenate((train_labels,labels[i:i+50]),axis=0)
        validate_img_array=np.concatenate((validate_img_array,validate_img_array[i+50:i+100]),axis=0)
        validate_labels=np.concatenate((validate_labels,validate_labels[i+50:i+100]),axis=0)
        test_img_array=np.concatenate((test_img_array,img_array[i+100:i+200]),axis=0)
        test_labels=np.concatenate((test_labels,labels[i+100:i+200]),axis=0)
    return train_img_array,train_labels,validate_img_array, validate_labels, test_img_array,test_labels

# def split_train_validate_test(img_array, labels):
#     training_images = []
#     training_labels = []
#     validation_images = []
#     validation_labels = []
#     testing_images = []
#     testing_labels = []
#     for fold in range(5):
#         train_img_array = img_array[0:50]
#         train_labels = labels[0:50]
#         validate_img_array = img_array[50:100]
#         validate_labels = labels[50:100]
#         test_img_array = img_array[100:200]
#         test_labels = labels[100:200]
#         for i in range(200, 2000, 200):
#             train_img_array = np.concatenate((train_img_array, img_array[i:i+50]), axis=0)
#             train_labels = np.concatenate((train_labels, labels[i:i+50]), axis=0)
#             validate_img_array = np.concatenate((validate_img_array, validate_img_array[i+50:i+100]), axis=0)
#             validate_labels = np.concatenate((validate_labels, validate_labels[i+50:i+100]), axis=0)
#             test_img_array = np.concatenate((test_img_array, img_array[i+100:i+200]), axis=0)
#             test_labels = np.concatenate((test_labels, labels[i+100:i+200]), axis=0)
#         training_images.append(train_img_array)
#         training_labels.append(train_labels)
#         validation_images.append(validate_img_array)
#         validation_labels.append(validate_labels)
#         testing_images.append(test_img_array)
#         testing_labels.append(test_labels)
#     return training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels
