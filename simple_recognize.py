import numpy as np
import pickle
import cv2
import os
import model as embedding
from imutils import paths
import imutils
import argparse
import torch

# custruct the argument parser and pars the arguments for imgepath

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagePath", required=True,
				help="path to input image")
args = vars(ap.parse_args())

imagePath = args["imagePath"]
savename = imagePath.split('.')[-2]
print(savename)

# load face detection model

protoPath = "./model_paths/deploy.prototxt.txt"
modelPath = "./model_paths/res10_300x300_ssd_iter_140000.caffemodel"

recognizerPickle = "./output/SimpleRecognizer.pickle"
labelPickle = "./output/SimpleLabel.pickle"

predictedImg = "./predictedImg"


# loading face detection model
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load embedding model
embedder = embedding.InceptionResnetV1(pretrained="vggface2").eval()

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(recognizerPickle, "rb").read())
label = pickle.loads(open(labelPickle, "rb").read())

image = cv2.imread(imagePath)
(h,w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

detector.setInput(blob)
detections = detector.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    
    # extract the confidence (i.e., probalility) associated with the prediction
    confidence = detections[0, 0, i, 2]
    
    # fillter out weak detections
    if confidence > 0.2:
        
        # compute the (x ,y) - coordinates of the bounding box for the face
        
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # extract the face ROI
        face = image[startY:endY , startX:endX]
        (fH ,fW) = face.shape[:2]
        
        # ensure the facce width and height are sufficently large
        if fW < 20 or fH < 20:
            print("[Error] - Face size in Image not sufficent to get Embeddings : ", imagePath)
            continue
        

        try:
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(160, 160), (0, 0, 0), swapRB=True, crop=False)
        except:
            print("[Error] - Face size in Image not sufficent to get Embeddings : ", imagePath)
            continue
        
        faceTensor = torch.tensor(faceBlob)
        faceEmbed = embedder(faceTensor)
        flattenEmbed = faceEmbed.squeeze(0).detach().numpy()
        
        array = np.array(flattenEmbed).reshape(1,-1)
        
        # perform classification to recognize the face
        
        preds = recognizer.predict_proba(array)[0]
        
        j = np.argmax(preds)
        
        proba = preds[j]
        name = label.classes_[j]
        
        #draw the bunding box of the face along with the associated probability
        
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        
# save image predicte foler
cv2.imwrite("{}/{}.png".format(predictedImg, savename), image)

# show the output image
cv2.imshow(savename, image)
cv2.waitKey(0)