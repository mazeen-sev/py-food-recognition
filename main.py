import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import cv2
from imageai.Detection import ObjectDetection  
import argparse
import os
import pytesseract
# iterating through the items found in the image  

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
file_path = "input/66f.jpg"

ocr_text="empty"

window1 = Tk()

l1 = tk.Label(window1,text='Food Scanner Alpha  ver 0.00001',width=50)  
l1.grid(row=0,column=0,columnspan=4)


l3 = tk.Label(window1,text="detected text is "+ocr_text,width=100)  
l3.grid(row=4,column=0,columnspan=4)


img1 = Image.open(file_path)


img1 = img1.resize((500,500))

img1 = ImageTk.PhotoImage(img1)

Label(window1, image = img1).grid(row = 3, column = 0, padx = 5, pady = 5)


def detect_object(img):
    return img

img2 = img1
Label(window1, image=img2).grid(row=3, column=1, padx = 5, pady = 5)


b1 = Button(window1, text='browse', width=20,command = lambda:upload_file())
b1.grid(row=1,column=0,columnspan=5)

b2 = Button(window1, text='live', width=20,command = lambda:start_camera())
b2.grid(row=2,column=0,columnspan=5)

def detect_object2(img):
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "model/retinanet_resnet50_fpn_coco-eeacb38b.pth")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/retinanet_resnet50_fpn_coco-eeacb38b.pth
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , file_path), output_image_path=os.path.join(execution_path , "output/2_detected.jpg"), minimum_percentage_probability=40)

    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")

def print_images():
    global img1 
    global img2
    img1 = Image.open(file_path)
    detect_object(img1)
    img1 = img1.resize((500,500))

    img1 = ImageTk.PhotoImage(img1)

    Label(window1, image = img1).grid(row = 3, column = 0, padx = 5, pady = 5)

    detect_object2(img1)

    img2 = Image.open("output/2_detected.jpg")
    ocr_text = "empty"
    img2=img2.resize((500,500))
    img2 = ImageTk.PhotoImage(img2)
    
    Label(window1, image=img2).grid(row=3, column=1, padx = 5, pady = 5)

    l3 = tk.Label(window1,text="detected text is"+ocr_text,width=100)  
    l3.grid(row=4,column=0,columnspan=4)


def start_camera():
    obj_detect = ObjectDetection()
    obj_detect.setModelTypeAsRetinaNet()
    obj_detect.setModelPath("model/retinanet_resnet50_fpn_coco-eeacb38b.pth")
    obj_detect.loadModel()

    cam_feed = cv2.VideoCapture(0)
    cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
    cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)

    while True:    
        ret, img = cam_feed.read()   
        annotated_image, preds = obj_detect.detectObjectsFromImage(input_image=img,
                        output_type="array",
                        display_percentage_probability=False,
                        display_object_name=True)

        cv2.imshow("", annotated_image)
        if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
            break

    cam_feed.release()
    cv2.destroyAllWindows()

def upload_file():
    global file_path
    file_path = tk.filedialog.askopenfilename()
    print_images()


window1.mainloop()
