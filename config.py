import tensorflow as tf
import datetime

class Config():
    custom_directory = r"D:\year-end-project\models\final\1-50-resized-128x128_10"
    custom_directory_colab = "/content/drive/MyDrive/custom"
    custom_directory_32x32 = r"D:\year-end-project\models\final\1-50-resized-32x32"
    custom_directory_32x32_10 = r"D:\year-end-project\models\final\1-50-resized-32x32_10"
    custom_directory_28x28 = r"D:\year-end-project\models\final\1-50-resized-28x28"
    kmnist_directory = r"D:\year-end-project\models\final\kkanji"
    batch_size = 32
    epochs_5 = 5
    epochs_10 = 10
    datagen_percent = (0.7, 0.15, 0.15)
    rotation_range = 60
    width_shift_range=0.2
    height_shift_range=0.2
    shear_range=0.2
    test_size=0.15
    copy_num = 50
    sample_size = 10
    num_channels = 1
    epsilon = 1e-7
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5
    alpha = 0.0005
    epochs = 100
    optimizer = tf.keras.optimizers.Adam()
    devnagri_train = r"D:\year-end-project\models\final\My Preprocessed dataset\train"
    devnagri_test = r"D:\year-end-project\models\final\My Preprocessed dataset\test"
    devnagri_train_10 = r"D:\year-end-project\models\final\devnagri_10\test"
    devnagri_test_10 = r"D:\year-end-project\models\final\devnagri_10\train"

