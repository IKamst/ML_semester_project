import numpy as np
import cv2

def load_data(expand=False,plot=False):  # load mfeat-pix.txt as opencv' format
    
    with open("mfeat-pix.txt", "r") as f: 
        data = f.read() 

    data=data.replace(" ", "")
    data=list(data.split("\n"))
    img_array=[]
    for im in range(2000):
        img=[]
        for i in range(16):
            row=[]
            for j in range(15):
                row.append(int(data[im][i*15+j]))
            img.append(row)
        if expand: # expand the img from 16*15 to 32*32
            img = cv2.copyMakeBorder(np.array(img,dtype=np.uint8), 8, 8, 8, 9, cv2.BORDER_CONSTANT, value=0) # 添加边框
        img_array.append(img)  
    img_array=(np.array(img_array,dtype=np.uint8)/6)*255
    
    labels=[]
    for i in range(10):
        labels=labels+[i for _ in range(200)]
    
    if plot:
        for im in range(2000):  #plot the img one by one
            print(np.shape(img_array[im]))
            while(1):

                cv2.imshow('img',img_array[im])
                if cv2.waitKey(1) == ord('0'):
                    break
            cv2.destroyAllWindows()
    return img_array,np.array(labels)

def split_train_test(img_array,labels):# the first 100 is for training, the last 100 is for testing
    train_img_array=img_array[0:100]
    train_labels=labels[0:100]
    test_img_array=img_array[100:200]
    test_labels=labels[100:200]
    for i in range(200,2000,200):
        print(i)
        train_img_array=np.concatenate((train_img_array,img_array[i:i+100]),axis=0)
        train_labels=np.concatenate((train_labels,labels[i:i+100]),axis=0)
        test_img_array=np.concatenate((test_img_array,img_array[i+100:i+200]),axis=0)
        test_labels=np.concatenate((test_labels,labels[i+100:i+200]),axis=0)
    return train_img_array,train_labels,test_img_array,test_labels

if __name__ == "__main__":
    img_array,labels=load_data(expand=True)
    train_img_array,train_labels,test_img_array,test_labels=split_train_test(img_array,labels)