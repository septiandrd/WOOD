from keras.optimizers import Adam
from keras.applications import InceptionV3
from keras.models import load_model
from keras.utils import to_categorical
from keras import losses,metrics
import Dataset_10
import os
import h5py
import numpy as np

if __name__ == '__main__':

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'ResNet_10class_lr.h5'
    weight_name = 'ResNet_10class_weight_lr.h5'
    model_path = os.path.join(save_dir, model_name)
    weight_path = os.path.join(save_dir, weight_name)

    x_train,y_train,x_test,y_test,dictionary = Dataset_10.get_dataset()

    x_train_resized = Dataset_10.resize_imgs(x_train,150)
    x_test_resized = Dataset_10.resize_imgs(x_test,150)

    y_train = to_categorical(y_train,num_classes=10)
    y_test = to_categorical(y_test,num_classes=10)

    # model = load_model(model_path)

    model = InceptionV3(include_top=True,weights=None,classes=10)

    opt = Adam(lr=2e-5)
    model.compile(optimizer=opt,loss=losses.categorical_crossentropy,metrics=[metrics.categorical_accuracy])
    model.fit(x_train_resized,y_train,epochs=30,batch_size=15)
    model.save(model_path)
    model.save_weights(weight_path)

    score1 = model.evaluate(x_train_resized, y_train, batch_size=10)
    score2 = model.evaluate(x_test_resized, y_test, batch_size=10)
    print("Score data training : ",score1)
    print("Score data testing : ",score2)