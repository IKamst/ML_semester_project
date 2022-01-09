from tensorflow import config
from tensorflow.keras import models,layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import random,cv2
from tensorflow.keras.utils import to_categorical
from load_data import *
import numpy as np

AUG_ROTATION_RANGE = 60
AUG_ZOOM_RANGE = 0.8
AUG_SHEAR_RANGE = 20

AUG_CHANCE = 1

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
    BaseCnn = MobileNetV2(input_shape=(32, 32, 3), alpha=1, #minimalistic=True,
                               include_top=False,weights='imagenet',#classes=1,dropout_rate=0.7,
                              )#'imagenet' None

    model=models.Sequential()
    model.add(BaseCnn)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10,activation = 'softmax')) 
    
    return model
def data_generator(dataset,st,ed,batch_size, aug):
    nowinx = st
    while True:
        im_array = []
        lb_array = []
        for i in range(batch_size):
            im=[]
            im_gray = np.array(dataset[nowinx][0],dtype=np.uint8)
            im_bgr=cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)
            
            tmp_im_array = np.array(im_bgr) 
            tmp_im_array = tmp_im_array / 255 

            tmp_im_array = tmp_im_array[np.newaxis,:,:,:] 
            lb=dataset[nowinx][1]
            
            if len(im_array) == 0:
                im_array = tmp_im_array
                lb_array.append(lb)
            else:
                im_array = np.concatenate((im_array,tmp_im_array),axis=0) 
                lb_array.append(lb)
            random.shuffle(dataset) if nowinx==ed else nowinx+1
            print("----dataset shuffled-----") if nowinx==ed else nowinx+1
            nowinx = st if nowinx==ed else nowinx+1 
            
        lb_array = to_categorical(np.array(lb_array),10)

        # augment the data

        if (aug is not None) and (random.random() <= AUG_CHANCE):
            im_array_old = im_array.copy()
            new_array = im_array
            new_array = next(aug.flow(x=new_array,y=None,batch_size = batch_size, shuffle=False, 
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
        yield(im_array,lb_array)


        
if __name__ == "__main__":
    
    img_array,labels=load_data(expand=True,plot=0)
    train_img_array,train_labels,test_img_array,test_labels=split_train_test(img_array,labels)
    train_set=list(zip(train_img_array, train_labels))
    test_set=list(zip(test_img_array, test_labels))

    solve_cudnn_error()
    aug = ImageDataGenerator(rotation_range = AUG_ROTATION_RANGE, zoom_range = AUG_ZOOM_RANGE, width_shift_range = 0,
                             height_shift_range = 0, shear_range = AUG_SHEAR_RANGE, horizontal_flip =False, vertical_flip =False,
                             fill_mode = "constant", cval=0)

    train_gen = data_generator(train_set,0,len(train_set)-1,30,aug)
    validate_gen = data_generator(test_set,0,len(test_set)-1,20,None) 

   
    model=model_build()
    model.summary()
    model.compile(optimizer = SGD(lr = 0.01),
                  loss = 'categorical_crossentropy', metrics = ['accuracy'])

    reducelrOnplateau=ReduceLROnPlateau(monitor='loss',factor=0.33,patience=7,verbose=1)

    model_file_path="./model/digit_recognition.hdf5"
    model_checkpointer = ModelCheckpoint(
        filepath=model_file_path,
        monitor='loss', 
        verbose=1,
        mode='auto',
        save_best_only=True,
        save_weights_only=False,
    )

    tensorboard = TensorBoard(log_dir='./logs/digit_recognition_logs', update_freq='batch')

    his = model.fit_generator(
        generator = train_gen, 
        callbacks=[reducelrOnplateau,tensorboard,model_checkpointer],#
        steps_per_epoch = 40, 
        validation_data = validate_gen, 
        validation_steps = 20, 
        epochs = 100
    )

