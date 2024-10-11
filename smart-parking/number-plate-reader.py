import cv2
import mediapipe as mp;
import numpy as np;
import pandas as pd;
import easyocr as es;
import qrcode;
import matplotlib.pyplot as plt;

def read_file(file_path: str):
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error: Could not read the image {file_path}")
        return None
    return image

def format_gray(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    # print("gray scale is", gray);
    return gray;

def classifier(gray_image):
    frame = None
    plate_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml");
    plates = plate_classifier.detectMultiScale(gray_image, scaleFactor= 1.1, minNeighbors=5, minSize=(30,30));
    for (x,y,w,h) in plates:
        # drawing the bounding boxes
        cv2.rectangle(gray_image, (x,y), (x+w, y+h), (0, 255, 0), 2);
        plate_number = gray_image[y:y+h, x:x+w];
        frame = plate_number;
        # print("plate numner", plate_number)
        # return plate_number; 
    # cv2.imshow("Plate number detection", gray_image);
    # cv2.waitKey(0)
    # cv2.destroyAllWindows();
    return frame;
    
def number_plate_extraction(frame):
    reader = es.Reader(['en']);
    plate_value = reader.readtext(frame, detail=0)[-1];
    return reader, plate_value
    
def qr_code_generation(number_plate, value):
    qr = qrcode.QRCode(version=2, box_size=10, border=4);
    qr.add_data(number_plate);
    qr.make(fit=True)
    qrImage = qr.make_image(fill_color="black", back_color="white");
    print(type(qrImage));
    plt.imshow(qrImage)
    plt.show()
    qr_code_np = np.array(qrImage);
    print(type(qr_code_np));
    qr_code_np = cv2.cvtColor(qr_code_np, cv2.COLOR_RGB2BGR);
    writeValue = f"{value}.png";
    cv2.imwrite(f"{value}.png", qr_code_np);
    return writeValue;
    
def qr_code_scanner_and_verification(value):
    qr_detector = cv2.QRCodeDetector();
    image = cv2.imread(value);
    msg, bbox, _ = qr_detector.detectAndDecode(image);
    print(bbox);
    if msg == value:
        print(f"QR code {msg} verified successfully");
        cv2.putText(image, f'sucessfully authorized: ', (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200, 0), 1)
        value = cv2.putText(image, msg, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200, 0), 1)
        plt.imshow(value)
        
    
    
    

image = read_file(file_path="/Users/charleso/Desktop/computations/geneAlgo/src/smart-parking/car2.jpg");
gray = format_gray(image=image)
frame = classifier(image);

reader, plate_value = number_plate_extraction(frame)
writeValue = qr_code_generation(plate_value, reader);
qr_code_scanner_and_verification(writeValue);
plt.show()
