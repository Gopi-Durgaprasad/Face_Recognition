# Face_Recognition

This is a repository for Face_Recognition using FaceNet Inception Resnet (V1) model in pytorch and using state of the art Face Detection model called Retina Faces

Started with Simple Face Recognition model to understand.

### Existing Research

FaceNet: A Unified Embedding for Face Recognition and Clustering: [link](https://arxiv.org/pdf/1503.03832.pdf)

RetinaFace: Single-stage Dense Face Localisation in the Wild : [link](https://arxiv.org/pdf/1905.00641.pdf)

**Github Repositorys**

FaceNet Implementation : [link](https://github.com/davidsandberg/facenet)

FaceNet PyTorch Implementation : [link](https://github.com/timesler/facenet-pytorch)

RetinaFace PyTorch Implementation : [link](https://github.com/biubug6/Pytorch_Retinaface)


### Get Started

We are stating with pre-trained FaceNet model implemented in PyTorch [link](https://github.com/timesler/facenet-pytorch)

From this repo, we download pre-trained weights and models.

ModelName : VGGFace2 [link](https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py)

Weights : [link](https://drive.google.com/uc?export=download&id=1TDZVEBudGaEd5POR5X4ZsMvdsh1h68T1)

### Challenges
Face Recognition there are mainly three challenges.

1. Detecting Faces
2. Get Embeddings for faces
3. Train model Clustering/Classification/Similarly to recognize the Face

**1. Detecting Faces:**
<p>The first main challenge is Detecting Faces from a given image, there are many models to detect faces in an image</p>

*  The first main challenge is Detecting Faces from a given image.
*  There are so many models to detect faces in this [link](https://github.com/StarStyleSky/awesome-face-detection)
*  We are using RetinaFace (sota) Face Detection model

**2. Get Embeddings for faces**
<p>The second challenge is getting embedding for the detected faces, we are using FaceNet Model for getting Embeddings for faces.</p>

* Trine a model using FaceNet architecture / Download pre-train model and train.
* FaceNet Implementation :[link](https://github.com/timesler/facenet-pytorch)
* Pretrained Model :[link](https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py)
* Pretrainde Weights :[link](https://drive.google.com/uc?export=download&id=1TDZVEBudGaEd5POR5X4ZsMvdsh1h68T1)

**3. Train model Clustering/Classification/Similarly to recognize the Face**
<p>There are three many ways to train a model to recognize the face </p>

* Using Clustering
* Using Classification
* Using Similarity Matrix

### Simple Model

<p> We are build simple model form this [link](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/)</p>

<p> Run this files </p>

**1. Extract Embeddings:** `$python3 Simple_extractEmbeddings.py`

**2. Train Model:** `$python3 Simple_trainModel.py`

**3. Recognize Face:** `$python3 Simple_recognize.py --image kohili-sachin-dhoni.jpg`

**Input Image:** kohili-sachin-dhoni.jpg

![kohili-sachin-dhoni.jpg](kohili-sachin-dhoni.jpg?raw=true "kohili-sachin-dhoni.jpg")

**Output Image:**

![kohili-sachin-dhoni.png](predictedImg/kohili-sachin-dhoni.png?raw=true "kohili-sachin-dhoni.png")
