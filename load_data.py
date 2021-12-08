import numpy as np
import cv2

def load_data():  # load mfeat-pix.txt as opencv' format
    
    with open("mfeat-pix.txt", "r") as f:  # 打开文件
        data = f.read()  # 读取文件

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
        img_array.append(img)  
    img_array=np.array(img_array,dtype=np.uint8)/6

    for im in range(2000):  #plot the img one by one
        while(1):
            cv2.imshow('org',img_array[im])
            if cv2.waitKey(1) == ord('0'):
                break
        cv2.destroyAllWindows()
    return img_array

if __name__ == "__main__":
    mm=load_data()