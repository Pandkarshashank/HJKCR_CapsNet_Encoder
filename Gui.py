from tkinter import * 
import cv2
from model import predict,show_image,recover,test_accuracy,get_data,get_model_name
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

X_train,y_train,X_test,y_test,testing,testing_dataset_size,labels = get_data('Custom')

cap = cv2.VideoCapture(1)

def show_frame():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    resized_image = cv2.resize(cv2image, (400,400))
    img = Image.fromarray(resized_image)
    imgtk = ImageTk.PhotoImage(image=img)
    l_cam.imgtk = imgtk
    l_cam.configure(image=imgtk)
    l_cam.after(10, show_frame)

def predict_image(img):
    model = recover(selected_option.get())
    prediction =  predict(model, tf.expand_dims(img, axis=0))
    return prediction

def capture_image():
    _, frame = cap.read()
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_img = np.array(gray_img)
    _, img = cv2.threshold(gray_img, 90, 255, cv2.THRESH_BINARY)
    model_name = get_model_name(selected_option.get())
    if model_name == 'Kmnist' :
        resized_img = cv2.resize(img,(28,28))
        img = cv2.bitwise_not(img)
    elif model_name == 'Custom':
        resized_img = cv2.resize(img,(32,32))
    cv2.imwrite("captured_image.jpg", resized_img)
    cv2.imshow("captured_image",img)
    img = np.expand_dims(resized_img,axis= -1)
    img = img.astype(np.float32)
    print('predicting....')
    predicted_label = predict_image(img)
    l_prediction.config(text="Prediction: "+str(predicted_label[0]))
    if model_name == 'Custom':
        l_label.config(text = labels[predicted_label[0]])
    

root = Tk()

root.geometry("1200x800")

l = Label(root, text="Handwritten Japanese Kanji Character Recognition using Different Pruning Algorithms",
          font="Helvetica 20 bold", bg="#f0f0f0", fg="#333333", pady=20)
l.pack(fill=X)

f1 = Frame(root, bg="#f7f7f7", bd=2, relief=SUNKEN)
f1.pack(side=LEFT, fill=Y, padx=10, pady=10)

l_left = Label(f1, text="Model Selection Frame", font="Helvetica 16 bold", bg="#f7f7f7", fg="#333333")
l_left.pack(pady=20, padx=50)

options = ["Kmnist", "Kmnist_Magnitude", "Kmnist_Similarity","Custom", "Custom_Magnitude", "Custom_Similarity"]
selected_option = StringVar(f1)
selected_option.set(options[0])

dropdown = OptionMenu(f1, selected_option, *options)
dropdown.config(font="Helvetica 14", bg="#ffffff", fg="#333333")
dropdown.pack(pady=50, padx=30)

l_prediction = Label(f1, text="", font=("Helvetica", 14, "bold"), bg="#f7f7f7", fg="#333333")
l_prediction.pack(pady=50, padx=30)

l_label = Label(f1, text="", font=("Helvetica", 14, "bold"), bg="#f7f7f7", fg="#333333")
l_label.pack(pady=60, padx=30)

f2 = Frame(root, bg="#f7f7f7", bd=2, relief=SUNKEN)
f2.pack(side=RIGHT, fill=Y, padx=10, pady=10)

l_right = Label(f2, text="Camera Frame", font="Helvetica 16 bold", bg="#f7f7f7", fg="#333333")
l_right.pack(pady=20)

l_cam = Label(f2, bg="#ffffff", width=600, height=400)
l_cam.pack(pady=10)

btn_capture = Button(f2, text="Capture", command=capture_image, font="Helvetica 14", bg="#4CAF50", fg="#ffffff",
                     activebackground="#45a049", width=20, height=2)
btn_capture.pack(pady=20)

l_captured = Label(f2, bg="#ffffff", width=600, height=400)
l_captured.pack(pady=10)

cap = cv2.VideoCapture(0)

show_frame()

root.mainloop()

cap.release()
