# import the necessary packages

from imutils import paths
import numpy as np
import imutils
import cv2
import torch
import pickle
from tqdm import tqdm
import os

from facenet_pytorch import MTCNN
from PIL import Image

# importing pre-trined FaceNet model in pytorch
import model as embedding

# face detection model to detect faces
mtcnn = MTCNN(image_size=160)

# load FaceNet 'vggface2' model to get embeddings for faces
embedder = embedding.InceptionResnetV1(pretrained='vggface2').eval()

# paths to save pickle file
currentDir = os.getcwd()

# images folder that contains different folders names with faces images
dataset = os.path.join(currentDir , "dataset")

# path to save embeddings
embeddingPickle = os.path.join(currentDir, "output/FinalEmbeddings.pickle")

# getting all image patha
imagePaths = list(paths.list_images(dataset))

print("Total number of images : ", len(imagePaths))

# create lists to append ImgPaths/names/imageIDs/boxs/embeddings
ImgPaths = []
names = []
imageIDs = []
boxs = []
embeddings = []

# initlize the total number of faces processed
total = 0

# loop over the image ImgPaths
for (i, imagePath) in enumerate(tqdm(imagePaths)):

    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    imageID = imagePath.split(os.path.sep)[-1].split('.')[-2]

    # loading image
    image = Image.open(imagePath)

    try:
        # detecting face
        img_cropped = mtcnn.detect(image)
        # rectangle coordenates of the face
        box = img_cropped[0][0]
        # converted into int type
        box = box.astype('int')

        # resizing the cropped image as 160x160
        face = mtcnn(image)
    except:
        print("[Error] resizing : ", imagePath)
        continue

    # getting embedding vector 512-d for face
    faceEmbed = embedder(face.unsqueeze(0))
    flattenEmbed = faceEmbed.squeeze(0).detach().numpy()

    ImgPaths.append(imagePath)
    imageIDs.append(imageID)
    names.append(name)
    boxs.append(box)
    embeddings.append(flattenEmbed)
    total += 1

# save all embeddings / ImgPaths / imageIDs / names / boxs
# dump all to  disk as pickle file
print("[INFO] serializing {} encodings ....".format(total))
data = {"paths":ImgPaths, "names":names, "imageIDs":imageIDs, "boxs":boxs, "embeddings":embeddings}
f = open(embeddingPickle , "wb")
f.write(pickle.dumps(data))
f.close()
