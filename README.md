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
<p> The first main challenge is Detecting Faces from a given image, there are many models to detect faces in an image </p>

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

* <b> For more detailed understanding and code : [simple_EDA.ipynb](https://github.com/Gopi-Durgaprasad/Face_Recognition/blob/master/simple_EDA.ipynb) </b><br>
* <b> TSNE Visulization for Embeddings </b>

<p float="left">
  <img src="output/tsne-1.png" width="400"/>
  <img src="output/tsne-2.png" width="400"/> 
</p>

<p> Observe TSNE Visulization of 512-d Embeddings those are well clustered </p><br>

* <b> Distance (Best threshold for the verification </b><br>
  <p> Using this Distance threshold we can easily desided threshold to Recognize faces</p>
<p  align="center">
  <img src="output/threshold.png"/> 
</p>
<p> Observe Distance threshold is 0.74 and it gives 0.98 Accuracy, to recognize faces using simple Euclident distance </p><br>

* <b>Distance distributions of positive and negative pairs</b><br>
<p>
  <img src="output/distance_pos_neg.png"/> 
</p>
<p> Observe distributions of positive and negative pairs, almost well separated using threshold 0.74 wich gives high Accuracy</p><br>

<p> Run this files </p>

**1. Extract Embeddings:** `$python3 Simple_extractEmbeddings.py`

**2. Train Model:** `$python3 Simple_trainModel.py`

**3. Recognize Face:** `$python3 Simple_recognize.py --image kohili-sachin-dhoni.jpg`

<table  align="center">
        <tr  align="center" >
          <td><b>Input Image</b></td>
          <td><b>Output Image</b></td>
        </tr>
        <tr>
            <td><img src="kohili-sachin-dhoni.jpg"/></td>
            <td><img src="predictedImg/kohili-sachin-dhoni.png"/></td>
       </tr>
<table>
  
## Retina Faces (state of the art) Face Detection Model

- For accurate face detection, we are using Retina Face model to detect faces.
- Retina Faces is state of the art model to detect faces

* <b> For more detailed understanding and code : [retinaface.ipynb](https://github.com/Gopi-Durgaprasad/Face_Recognition/blob/master/retinaface.ipynb) </b><br>

**Face Detection Outputs:**

<img src="output/group_of_people.png"/>
<img src="output/group_of_people-2.png"/>
