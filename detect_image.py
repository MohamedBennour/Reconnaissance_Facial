# importation des bibliothèques
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import argparse
import numpy as np

# déclarer les arguments de argparse 
parser = argparse.ArgumentParser(description="detection d'une image")
parser.add_argument('-i', '--image', required=True, help="Chemin de l'image")

args = vars(parser.parse_args())

# importer l'architecture et les poids du modèle de détection de visage
architecture = r"face_detector\deploy.prototxt"
poids = r"face_detector\weights.caffemodel"
net = cv2.dnn.readNet(architecture, poids)

# importer le model de détection de mask
model = load_model(r"model\face_model.h5")

# charger l'image d'entrée
image =cv2.imread(args['image'])
(h, l) = image.shape[:2]
# {h:hauteur , l:largeur}

# prétraitement d'image
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300,300), mean=(104, 177, 123))

# obtenir les détections de visage
net.setInput(blob)
detection = net.forward()

for i in range(detection.shape[2]):
    confidence = detection[0, 0, i, 2] 
    if confidence > 0.5:

        #calculer les délimitations de la boîte
        boxes = detection[0, 0, i, 3:] * np.array([l,h, l,h]) 

        (gauche, haut, droite, bas) = boxes.astype('int') 
      
        (gauche, haut) = (max(0, gauche)), max(0, haut) 
        (droite, bas) = (min(l-1, droite), min(h-1, bas))
        
        face = image[haut:bas , gauche:droite]

      
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, dsize=(224, 224))
        face = img_to_array(face)      
        face = preprocess_input(face)  
        face = np.expand_dims(face, axis=0)              
        
        (mohamed, other, youssef) = model.predict(face)[0]

        # déterminer la classe de chaque prédiction
        if mohamed > youssef and mohamed > other:
            label = "mohamed"
            couleur = (0, 255, 0)
        elif youssef > mohamed and youssef > other:
            label = "youssef"
            couleur =(0, 255, 0)
        else:
            label = "Other"
            couleur = (0, 0, 255)

        # tracer un rectangle autour des visages 
        cv2.rectangle(image, (gauche, haut), (droite, bas), couleur, 2)
        # afficher la classe du visage
        cv2.putText(image, label, (gauche, haut-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, couleur, 2)   
cv2.imshow("output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
