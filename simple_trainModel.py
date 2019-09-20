from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import pickle
import numpy as np

# paths to embedding pickle file
embeddingPickle = "./output/SimpleEmbeddings.pickle"

# path to recognizer pickle file
recognizerPickle = "./output/SimpleRecognizer.pickle"

# path to labels pickle file
lebelPickle = "./output/SimpleLabel.pickle"

# loading embeddings pickle
data = pickle.loads(open(embeddingPickle, "rb").read())

# encode the labels
label = LabelEncoder()
labels = label.fit_transform(data["names"])

# getting embeddings
Embeddings = np.array(data["embeddings"])

# train the model used to accept the 512-d embeddings of the face and 
# then produce the actual face recognition

recognizer = KNeighborsClassifier(n_neighbors= 2, metric='euclidean', weights="distance")
#recognizer = SVC(probability=True)
recognizer.fit(Embeddings, labels)

# write the actual face recognition model to disk
f = open(recognizerPickle, "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(lebelPickle,"wb")
f.write(pickle.dumps(label))
f.close()