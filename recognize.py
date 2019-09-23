"""
First run this 'extractEmbeddings.py' for extract embeddings
Second run this 'trainModel.py' for traine classification mode
Third run this file  along with image path to recognize facess in image

Note:

1) We are using state of the art Face Detection Model called Retina-Face to detect facess acuuretly

2) This Face Recognition model detects only trained facess

3) We also give some thresholds to identify facess that are not in Traing Set as 'None'

4) We give 'None' to those facess

"""

import numpy as np
import pickle
import cv2
import os
import model as embedding
import imutils
import argparse
import torch

# we save 'RetinaFace' model at 'models/retinaface'
# we load retinaface model to detect facess
import torch.backends.cudnn as cudnn
from models.retinaface.config import cfg
from models.retinaface.prior_box import PriorBox
from models.retinaface.py_cpu_nms import py_cpu_nms
from models.retinaface.retinaface import RetinaFace
from models.retinaface.box_utils import decode , decode_landm
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from PIL import Image
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

currentDir = os.getcwd()

# paths to embedding pickle file
embeddingPickle = os.path.join(currentDir, "output/FinalEmbeddings.pickle")

# path to save recognizer pickle file
recognizerPickle = os.path.join(currentDir, "output/FinalRecognizer.pickle")

# path to save labels pickle file
labelPickle = os.path.join(currentDir, "output/FinalLabel.pickle")

# path to save prdictedImages
predictedImg = os.path.join(currentDir, "predictedImg")
if not os.path.exists(predictedImg):
    os.mkdir(predictedImg)

# Use argparse to get image path on commend line



# loading 'RetinaFace' weights to detect facess
trained_model_path = "models/retinaface/weights/Final_Retinaface.pth"
cpu = True
confidence_threshold = 0.05
top_k = 5000
nms_threshold = 0.3
keep_top_k = 750
save_image_path = "predictedImg"
vis_threshold = 0.6

### check_keys

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    #print('Missing keys:{}'.format(len(missing_keys)))
    #print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    #print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

### remove_prefix
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


### load_model
def load_model(model, pretrained_path, load_to_cpu):
    #print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

torch.set_grad_enabled(False)

#net and model
net = RetinaFace(phase="test")
net = load_model(net , trained_model_path, cpu)
net.eval()
print("Finished loading model!")
cudnn.benchmark = True
device = torch.device("cpu" if cpu else "cuda")
net = net.to(device)

resize = 1

# load embedding model
embedder = embedding.InceptionResnetV1(pretrained="vggface2").eval()

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(recognizerPickle, "rb").read())
label = pickle.loads(open(labelPickle, "rb").read())
# loading embeddings pickle
data = pickle.loads(open(embeddingPickle, "rb").read())

COLORS = np.random.randint(0, 255, size=(len(label.classes_), 3), dtype="uint8")

Embeddings = np.array(data["embeddings"])
names = np.array(data["names"])
print("Embeddings ", Embeddings.shape)
print("Names ", names.shape)
#print("Labels ", labels.shape)

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def detectFacess(path):
    image_path = path
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #print(img_raw)
    img_raw_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    imageName = image_path.split('/')[-1].split('.')[-2]

    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    print('net forward time: {:.4f}'.format(time.time() - tic))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    for b in dets:
        if b[4] < vis_threshold:
            continue
        boxes = np.array(b[0:4])
        boxes = boxes.astype('int')

        (startX , startY, endX, endY) = boxes

        face = img_raw_rgb[startY:endY , startX:endX]

        try:
            #print("yes-1")
            faceRead = Image.fromarray(face)
            faceRead = faceRead.resize((160, 160), Image.ANTIALIAS)
            faceRead = F.to_tensor(faceRead)
            #print("yes-2")
        except:
            print("[Error] - resizing face ")
            continue
        #print(faceRead.shape)

        # getting embeddings for croped faces
        faceEmbed = embedder(faceRead.unsqueeze(0))
        flattenEmbed = faceEmbed.squeeze(0).detach().numpy()
        #print(flattenEmbed.shape)

        # predectiong class
        array = np.array(flattenEmbed).reshape(1,-1)
        # perform classification to recognize the face
        preds = recognizer.predict_proba(array)[0]

        j = np.argmax(preds)

        proba = preds[j]
        name = label.classes_[j]
        #print(name)

        result = np.where(names == name)
        resultEmbeddings = Embeddings[result]

        dists = []
        for emb in resultEmbeddings:
            d = distance(emb, flattenEmbed)
            dists.append(d)
        #print(dists)
        distarray = np.array(dists)
        #print(distarray)
        min_dist = np.min(distarray)
        max_dist = np.max(distarray)
        #print("Name : ",name)
        #print("min dist : ",min_dist)
        #print("max dist : ", max_dist)

        if proba >= 0.5:

            if (min_dist < 0.75 and max_dist < 1.4) or (min_dist < 0.5) or (proba == 1 and min_dist <= 0.5):

                print("dist name ", name)
                print("min dist : ",min_dist)
                print("max dist : ", max_dist)


                color = [int(c) for c in COLORS[j]]

                cv2.rectangle(img_raw, (startX, startY), (endX, endY), color, 2)

                text = "{}: {:.2f}".format(name, proba)
                cv2.putText(img_raw,text, (startX, startY - 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            else:

                print("________________missing______________")
                print("dist name ", name)
                print("min dist : ",min_dist)
                print("max dist : ", max_dist)
                print("probability :",proba)

                name = "NONE"

                color = (255, 255, 255)

                cv2.rectangle(img_raw, (startX, startY), (endX, endY), color, 2)

                text = "{}".format(name)
                cv2.putText(img_raw,text, (startX, startY - 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            name = "NONE"

            color = (255, 255, 255)

            cv2.rectangle(img_raw, (startX, startY), (endX, endY), color, 2)

            text = "{}".format(name)
            cv2.putText(img_raw,text, (startX, startY - 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # save image predicte foler
    cv2.imwrite("{}/{}.png".format(predictedImg, imageName), img_raw)
    #im = Image.open("{}/{}.png".format(predictedImg,imageName))
    #return im

    cv2.imshow(imageName, img_raw)
    cv2.waitKey(0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagePath", required=True, help="Image path to recognize facess")
    args = vars(ap.parse_args())

    imagePath = args["imagePath"]
    currentDirImage = os.getcwd()

    ImageDir = os.path.join(currentDirImage , imagePath)

    if not os.path.exists(ImageDir):
        print("Image not exists")

    #print("image path: ",ImageDir)
    readImg = plt.imread(ImageDir)
    #print("shape :", readImg.shape)

    detectFacess(imagePath)
