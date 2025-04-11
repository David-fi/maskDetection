from mtcnn.mtcnn import MTCNN
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import load_model
from skimage.io import imread
from skimage.transform import resize

# Load the best model
model = load_model('best_model.keras')

#class names 0,1,2
class_names = ['No Mask', 'Mask', 'Incorrect']

def MaskDetection(path_to_images, image_size=(128, 128), n_samples=4):
    #intialize the face detector 
    detector = MTCNN()
    #list of all images in the file given
    all_images = [f for f in os.listdir(path_to_images) if f.lower().endswith(('.jpg', '.jpeg'))]
    #randomly pull out the (n samples)4 from the spefic folder
    selected = random.sample(all_images, min(n_samples, len(all_images)))

    #wide figure where we show the selected images
    plt.figure(figsize=(15, 5))

    #loop over each image that was selected
    for i, img_name in enumerate(selected):
        #the full path path
        img_path = os.path.join(path_to_images, img_name)
        #load the image, numpy array
        image = imread(img_path)
        #using MTCNN i can detect the faces in the image 
        results = detector.detect_faces(image)
        #create a subplot for the current image
        plt.subplot(1, n_samples, i + 1)
        #show the image 
        plt.imshow(image)
        #get axis to draw the boxes and labels on the boxes
        ax = plt.gca()

        #for images with multiple faces we loop through all those detected in the image
        for face in results:
            #pull the face bounding coordinates
            x, y, w, h = face['box']
            #crop the face from the full image
            face_img = image[y:y+h, x:x+w]
            #skip ones which might cause runtime errors like zero-size or corrupted
            if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                continue
            
            #resize the face image to the input size used by the CNN
            resized_face = resize(face_img, image_size, anti_aliasing=True)
            #predict the mask status using the CNN
            pred = model.predict(np.expand_dims(resized_face, axis=0), verbose=0)[0]
            #determine the index of which class is most likely
            predicted_idx = np.argmax(pred)
            #for confidence scores grab the actual prob for the given prediction
            confidence = pred[predicted_idx]
            #formatted the label, class name and confidence score
            label = f"{class_names[predicted_idx]} ({confidence*100:.1f}%)"

            # Draw bounding box around the face
            rect = plt.Rectangle((x, y), w, h, fill=False, color='lime', linewidth=2)
            ax.add_patch(rect)
            #aboce the bounding box put the label
            ax.text(x, y - 10, label, color='white', fontsize=12, backgroundcolor='black')
        #withouth axis ticks it looks better
        plt.axis('off')
        #title is the image name
        plt.title(f"{img_name}")
    #adjust layout for cleaner look
    plt.tight_layout()
    #display
    plt.show()