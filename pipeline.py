import pickle
from config import Config
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np
import cv2
from tqdm import tqdm

class CreateDataset:
    def __init__(self,dataset,augment = False,copy = False):
        self.dataset = dataset
        self.to_augment = augment
        self.to_copy = copy

    def load_data(self,file_path):
        with open(os.path.join(Config.kmnist_directory,file_path), 'rb') as f:
            data = pickle.load(f)
        return data
    
    def resize(self,shape,images):
        resized_images = []
        print('Resizing to', shape)
        for single_image in tqdm(images, desc="Resizing Images"):
            resized_image = cv2.resize(single_image, shape)
            resized_images.append(cv2.bitwise_not(resized_image))
        return np.array(resized_images)

    def extract_from_directory(self,directory):
        subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

        class_folders = os.listdir(directory)
        class_labels = {label: index for index, label in enumerate(class_folders)}
        labels_reverse = {index: label for index, label in enumerate(class_folders)}

        data = []
        labels = []

        for subdirectory in tqdm(subdirectories, desc="Processing Directories"):
            class_path = os.path.join(directory, subdirectory)
            class_label = class_labels[subdirectory]

            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                input_image = np.expand_dims(input_image, axis=2)
                data.append(input_image)
                labels.append(class_label)

        return (data,labels)

    def extract_devnagri(self):
        if self.dataset == 'devnagri':
            print("images extracted from ",Config.devnagri_train)
            train_images,train_labels = self.extract_from_directory(Config.devnagri_train)
            test_images,test_labels = self.extract_from_directory(Config.devnagri_test)
        elif self.dataset == 'devnagri_10':
            print("images extracted from ",Config.devnagri_train_10)
            train_images,train_labels = self.extract_from_directory(Config.devnagri_train_10)
            test_images,test_labels = self.extract_from_directory(Config.devnagri_test_10)
        self.images = np.concatenate([train_images, test_images], axis=0)
        self.labels = np.concatenate([train_labels, test_labels], axis=0)
        self.images = self.resize((32,32),self.images)
        with open("D:/year-end-project/models/final/devanagri.pkl",'wb') as f:
            pickle.dump({"images" : self.images,"labels" : self.labels},f)
        print("Shape of Custom images:", self.images.shape)
        print("Shape of Custom labels:", self.labels.shape)

    def extract_custom(self,size):
        if size == 'normal':
            data,labels = self.extract_from_directory(Config.custom_directory)
        elif size == '32x32':
            data,labels = self.extract_from_directory(Config.custom_directory_32x32)
        elif size == '32x32_10':
            data,labels = self.extract_from_directory(Config.custom_directory_32x32_10)
        elif size == '28x28':
            data,labels = self.extract_from_directory(Config.custom_directory_28x28)
        self.images = np.array(data)
        self.labels = np.array(labels)
        print("Shape of Custom images:", self.images.shape)
        print("Shape of Custom labels:", self.labels.shape)

    def extract_mnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        con_images = np.concatenate([x_train, x_test], axis=0)
        self.images = np.expand_dims(con_images, axis=-1)
        self.labels = np.concatenate([y_train, y_test], axis=0)

        print("Shape of MNIST images:", self.images.shape)
        print("Shape of MNIST labels:", self.labels.shape)

    
    def extract_kmnist(self):
        train_images = self.load_data('train_images.pkl')
        train_labels = self.load_data('train_labels.pkl')
        test_images = self.load_data('test_images.pkl')
        test_labels = self.load_data('test_labels.pkl')


        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        con_images = np.concatenate([train_images, test_images])  # Assuming you meant to concatenate train_images and test_images
        self.images = np.expand_dims(con_images, axis=-1)
        self.labels = np.concatenate([train_labels, test_labels])

        print("Shape of Kmnist images:", self.images.shape)
        print("Shape of Kmnist labels:", self.labels.shape)

    def datagenerator(self):
        print("Augmenting")
        print(f"Images Shape: {self.images.shape}")
        print(f"Labels Shape: {self.labels.shape}")
        
        datagen = ImageDataGenerator(
            rotation_range=Config.rotation_range,
            width_shift_range=Config.width_shift_range,
            height_shift_range=Config.height_shift_range,
            shear_range=Config.shear_range,
            horizontal_flip=False,
            fill_mode='nearest'
        )
        datagen.fit(self.images)
        augmented_generator = datagen.flow(self.images, self.labels, batch_size=Config.batch_size, shuffle=False)

        self.augmented_images, self.augmented_labels = [], []
        num_batches = len(augmented_generator)
        
        progress_bar = tqdm(total=num_batches, desc="Augmenting Images")

        for _ in range(num_batches):
            images_batch, labels_batch = augmented_generator.__next__()
            self.augmented_images.append(images_batch)
            self.augmented_labels.append(labels_batch)
            progress_bar.update(1)
        progress_bar.close()
        
        self.augmented_images = np.concatenate(self.augmented_images)
        self.augmented_images = np.reshape(self.augmented_images, (-1, self.augmented_images.shape[1], self.augmented_images.shape[2], 1))

        self.augmented_labels = np.concatenate(self.augmented_labels)
        
        print(f"Augmented Images Shape: {self.augmented_images.shape}")
        print(f"Augmented Labels Shape: {self.augmented_labels.shape}")
        
        self.images = np.concatenate([self.images, self.augmented_images])
        self.labels = np.concatenate([self.labels, self.augmented_labels])
           
        print(f"Combined Images Shape: {self.images.shape}")
        print(f"Combined Labels Shape: {self.labels.shape}")
        
    def description(self):
        unique_labels, label_counts = np.unique(self.labels, return_counts=True)
        num_unique_labels = len(unique_labels)

        print("Unique labels:", unique_labels)
        print("Label counts:", label_counts)
        print("Number of unique labels:", num_unique_labels)

    def copy(self):
        if self.to_copy == True:
            print("Copy")
            self.images = np.tile(self.images, (Config.copy_num, 1, 1, 1))
            self.labels = np.tile(self.labels, Config.copy_num)
        else: 
            print("No copy")

        print(f"Copied Images Shape: {self.images.shape}")
        print(f"Copied Labels Shape: {self.labels.shape}")

    def show(self, sample_size=25):
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))  
        indices = random.sample(range(len(self.images)), sample_size)
        for i, idx in enumerate(indices):
            row = i // 5
            col = i % 5
            
            image = self.images[idx].squeeze()  
            label = self.labels[idx]

            axes[row, col].imshow(image, cmap='gray') 
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()
        
    def split(self, random_state=None):
        if self.to_augment:
            self.datagenerator()
        self.copy()
        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(self.images,self.labels , test_size=Config.test_size, random_state=random_state)    
        print(f"Training Set:   Images shape={self.train_images.shape}, Labels shape={self.train_labels.shape}")
        print(f"Test Set:       Images shape={self.test_images.shape}, Labels shape={self.test_labels.shape}")
                
        self.train = tf.data.Dataset.from_tensor_slices((self.train_images.reshape((-1, self.train_images.shape[1] * self.train_images.shape[2])).astype(np.float32) / 255.0, self.train_labels.astype(np.int32)))
        self.test = tf.data.Dataset.from_tensor_slices((self.test_images.reshape((-1, self.test_images.shape[1] * self.test_images.shape[2])).astype(np.float32) / 255.0, self.test_labels.astype(np.int32)))

    def run(self):
        if self.dataset.lower() == "custom":
            self.extract_custom('normal')
        elif self.dataset.lower() == "custom32x32":
            self.extract_custom('32x32')
        elif self.dataset.lower() == "custom32x32_10":
            self.extract_custom('32x32_10')
        elif self.dataset.lower() == "custom28x28":
            self.extract_custom('28x28')
        elif self.dataset.lower() == "kmnist":
            self.extract_kmnist()
        elif self.dataset.lower() == "mnist":
            self.extract_mnist()
        elif self.dataset.lower() == "devnagri" or self.dataset.lower() == "devnagri_10":
            if os.path.exists('D:/year-end-project/models/devanagri.pkl'):
                with open("D:/year-end-project/models/devanagri.pkl", 'rb') as f:
                        data = pickle.load(f)
                self.images = data['images']
                self.labels = data['labels']
            elif os.path.exists('D:/year-end-project/models/devanagri_10.pkl'):
                with open("D:/year-end-project/models/devanagri_10.pkl", 'rb') as f:
                        data = pickle.load(f)
                self.images = data['images']
                self.labels = data['labels']
            else:
                self.extract_devnagri()
        else:
            print("Enter Valid Dataset.")
        self.show()
        self.description()
        self.split(random_state=42)

    def get_train_test(self):
        return ((self.train_images,self.train_labels),(self.test_images,self.test_labels))
    
    def save_to_pickle(self, file_path):
        (X_train, y_train), (X_test , y_test) = self.get_train_test()
        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

if __name__ == '__main__':
    choice = input("Enter your choice (custom, kmnist, devnagri, mnist): ").lower()
    dataset = CreateDataset(choice)
    dataset.run()
    if not os.path.exists(choice):
        os.makedirs(choice)
    print(os.path.join(choice,str(choice)+'.pkl'))
    dataset.save_to_pickle(os.path.join(choice,str(choice)+'.pkl'))