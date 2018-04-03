import os
import numpy as np
import random
from PIL import Image
import cv2

def get_dataset() :

    PATH = './dataset_10'
    image_list_train = []
    label_list_train = []
    image_list_test = []
    label_list_test = []
    dictionary = {}

    label=0
    for dirs in sorted(os.listdir(os.path.join(PATH))) :
        if os.path.isdir(os.path.join(PATH,dirs)):
            print('Species :', dirs.split('_')[1],'; Code :',dirs.split('_')[0],'; i : ',label)
            dictionary[label] = dirs
            n=0
            for file in sorted(os.listdir(os.path.join(PATH,dirs))) :
                if file.split(".")[-1] == "jpg":

                    img = Image.open(os.path.join(os.path.join(PATH,dirs,file)))
                    img = img.convert("RGB")

                    img_arr = np.asarray(img)
                    img_arr = img_arr[62:962,190:1090,:]
                    # img_arr = resize_imgs(img_arr,900)
                    print(np.shape(img_arr))

                    for i in range(0,601,300):
                        for j in range(0,301,300):
                            print(i,j)
                            start_y = j  # Titik pengambilan pixel
                            start_x = i
                            new_img_arr = img_arr[start_x:start_x + 300, start_y:start_y + 300, :]
                            image_list_train.append(new_img_arr)
                            label_list_train.append(label)  # PENTING

                    for j in range(0,601,300):
                        start_y = 600  # Titik pengambilan pixel
                        start_x = j
                        new_img_arr = img_arr[start_x:start_x + 300, start_y:start_y + 300, :]
                        image_list_test.append(new_img_arr)
                        label_list_test.append(label)  # PENTING
                    n+=1
                if n==3 : break
            label+=1

    image_list_train = np.array(image_list_train)
    image_list_test = np.array(image_list_test)
    label_list_train = np.array(label_list_train)
    label_list_test = np.array(label_list_test)

    return image_list_train, label_list_train, image_list_test, label_list_test,dictionary

def resize_imgs(imgs,dims) :
    dim = (dims,dims)
    resized = []
    for img in imgs :
        resized.append(cv2.resize(img, dim, interpolation=cv2.INTER_AREA))

    return np.array(resized)
#
# if __name__ == '__main__':
#     x_train, y_train, x_test, y_test, dictionary = get_dataset()
    # img = Image.fromarray(x_train[0])
    # img.show()
    # img = Image.fromarray(x_train[1])
    # img.show()
    # img = Image.fromarray(x_train[2])
    # img.show()
    # img = Image.fromarray(x_test[0])
    # img.show()
    # img = Image.fromarray(x_test[1])
    # img.show()
    # img = Image.fromarray(x_test[2])
    # img.show()
