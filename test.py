from tensorflow.keras.models import load_model
from tensorflow import config
from load_data import *
from train import solve_cudnn_error
import numpy as np
import cv2

if __name__ == "__main__":
    
    img_array,labels=load_data(expand=True,plot=0)
    _,_,test_img_array,test_labels=split_train_test(img_array,labels)
    test_set=list(zip(test_img_array, test_labels))
    
    solve_cudnn_error()
    
    correct=0
    model=load_model("model/digit_recognition.h5")

    for i in range(len(test_set)):
        im_gray = np.array(test_set[i][0],dtype=np.uint8)
        im_bgr=cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)/ 255
        yy = list(model.predict(im_bgr.reshape(1,32,32,3))[0])
        num_pre=yy.index(max(yy))
        num_label=test_set[i][1]
        if num_pre==num_label:
            correct+=1
        else:
            print("error_index:",i,"num_pre:",num_pre,"num_label:",num_label)
            '''
            while(1):
                cv2.imshow('im_bgr',im_bgr)
                if cv2.waitKey(1) == ord('0'): # press 0 to close the pic window
                    break
            cv2.destroyAllWindows()
            '''

    correct_rate=correct/len(test_set)
    print("correct_rate",correct_rate)