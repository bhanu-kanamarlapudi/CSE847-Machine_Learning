import numpy as np
import pandas as pd
import tensorflow as tf
import copy
import pre_process
from sklearn.model_selection import train_test_split
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape
from keras.models import Sequential
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import Sequence
import pickle
import os


class Generator(Sequence):
    def __init__(self, df, batch_size=16, subset="train", shuffle=False, info={}):
        super().__init__()
        self.indexes = np.arange(len(self.df))
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subset = subset
        self.info = info

        # self.data_path = path
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, 9, 9, 1))
        y = np.empty((self.batch_size, 81, 1))
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        for i, f in enumerate(self.df['quizzes'].iloc[indexes]):
            self.info[index * self.batch_size + i] = f
            X[i,] = (np.array(list(map(int, list(f)))).reshape((9, 9, 1)) / 9) - 0.5
        if self.subset == 'train':
            for i, f in enumerate(self.df['solutions'].iloc[indexes]):
                self.info[index * self.batch_size + i] = f
                y[i,] = np.array(list(map(int, list(f)))).reshape((81, 1)) - 1
        if self.subset == 'train':
            return X, y
        return X


def get_data(file):
    data = pd.read_csv(file)
    quizzes_ = data['quizzes']
    solutions_ = data['solutions']
    input_ = []
    target_ = []
    for i in quizzes_:
        input_.append(np.array([int(j) for j in i]).reshape((9, 9, 1)))

    for i in solutions_:
        target_.append(np.array([int(j) for j in i]).reshape((81, 1)) - 1)

    input_ = np.array(input_)
    target_ = np.array(target_)
    input_ = input_ / 9
    input_ -= 0.5
    x_train, x_test, y_train, y_test = train_test_split(input_, target_, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def get_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(9, 9, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense((81 * 9)))
    model.add(Reshape((-1, 9)))
    model.add(Activation('softmax'))
    return model


def norm(a):
    return (a / 9) - 0.5


def denorm(a):
    return (a + 0.5) * 9


def sudoku_solver(sample):
    sample_raw = copy.copy(sample)
    while 1:
        out = model.predict(sample_raw.reshape((1, 9, 9, 1))).squeeze()
        pred = np.argmax(out, axis=1).reshape((9, 9)) + 1
        prob = np.around(np.max(out, axis=1).reshape((9, 9)), 2)
        sample_raw = denorm(sample_raw).reshape((9, 9))
        mask = (sample_raw == 0)
        if mask.sum() == 0:
            break
        prob_new = prob * mask
        ind = np.argmax(prob_new)
        x, y = (ind // 9), (ind % 9)
        val = pred[x][y]
        sample_raw[x][y] = val
        sample_raw = norm(sample_raw)
    return pred


def test_accuracy(input, target):
    correct = 0
    for i, feat in enumerate(input):
        pred = sudoku_solver(feat)
        true = target[i].reshape((9, 9)) + 1
        if abs(true - pred).sum() == 0:
            correct += 1
    return correct / input.shape[0]


def solve_sudoku(game):
    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9, 9, 1))
    game = norm(game)
    game = sudoku_solver(game)
    return game


# train_x, test_x, train_y, test_y = get_data('C:\\Users\\bhanu\\Desktop\\Masters\\projects\\Sudoku_AI\\src\\sudoku
# .csv') filename = 'cnn_trained_model.sav' if os.path.exists(filename): model = pickle.load(open(filename,
# 'rb')) else:
model = get_model()
#    pickle.dump(model, open(filename, 'wb'))
data = pd.read_csv('sudoku.csv')
train_idx = int(len(data) * 0.95)
data = data.sample(frac=1).reset_index(drop=True)
training_generator = Generator(data.iloc[:train_idx], subset="train", batch_size=640)
validation_generator = Generator(data.iloc[train_idx:], subset="train", batch_size=640)
adam = tf.optimizers.Adam(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)
filepath1 = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
filepath2 = "best_weights.hdf5"
checkpoint1 = ModelCheckpoint(filepath1, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
checkpoint2 = ModelCheckpoint(filepath2, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, min_lr=1e-6)
callbacks_list = [checkpoint1, checkpoint2, reduce_lr]
model.fit(training_generator, validation_data=validation_generator, epochs=1, verbose=1, callbacks=callbacks_list)
model.load_weights('best_weights.hdf5')
# model.fit(train_x, train_y, batch_size=32, epochs=2)
# test_acc = test_accuracy(test_x[:100], test_y[:100])
# print(test_acc)
game = '''
          0 8 0 0 3 2 0 0 1
          7 0 3 0 8 0 0 0 2
          5 0 0 0 0 7 0 3 0
          0 5 0 0 0 1 9 7 0
          6 0 0 7 0 9 0 0 8
          0 4 7 2 0 0 0 5 0
          0 2 0 6 0 0 0 0 9
          8 0 0 0 9 0 3 0 5
          3 0 0 8 2 0 0 1 0
      '''

game = solve_sudoku(game)
# board_Extractor = pre_process.Extractor('Sudoku.jpg')
# grid = board_Extractor.image_preprocess()
# game = solve_sudoku(grid)
print('solved puzzle:\n')
print(game)
print(np.sum(game, axis=0))
print(np.sum(game, axis=1))
