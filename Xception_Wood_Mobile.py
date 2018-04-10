from keras.optimizers import Adam
from keras.applications import Xception
from keras.utils import to_categorical
from keras import losses,metrics
from keras.callbacks import ModelCheckpoint
import Dataset_mobile
import os

if __name__ == '__main__':

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'xceptionMobile_wood_model.h5'
    model_path = os.path.join(save_dir, model_name)

    x_train, y_train, x_test, y_test, dictionary = Dataset_mobile.read_data()

    x_train_resized = Dataset_mobile.resize_imgs(x_train)
    x_test_resized = Dataset_mobile.resize_imgs(x_test)

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    model = Xception(include_top=True, weights=None, classes=2)

    opt = Adam(lr=5e-6)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])
    model.fit(x_train_resized,y_train,epochs=20,batch_size=6)
    model.save(model_path)

    score1 = model.evaluate(x_train_resized, y_train, batch_size=6)
    score2 = model.evaluate(x_test_resized, y_test, batch_size=6)
    print(score1)
    print(score2)