"""
Το συγκεκριμένο αρχείο δείχνει πως μπορεί να χρησιμοποιηθεί το αρχείο
Detectron.py και στη συνέχεια να αποθηκευτεί το αποτέλεσμα.
"""

import glob
import time

#Import everything from file Detectron.py
from Detectron import *

#Select model ('COCO','Cityscapes','Crosswalk','Traffic_Sign','Safety_Cones') 
# and model_type ('OD' , 'IS' , 'P' , 'SS') 

Models = [{'model':'Cityscapes', 'model_type': 'P'},
          {'model':'COCO', 'model_type': 'P'},
          {'model':'Traffic_Sign', 'model_type': 'OD'},
          {'model':'COCO', 'model_type': 'OD'},
          {'model':'Crosswalk', 'model_type': 'OD'},
          {'model':'Safety_Cones', 'model_type': 'OD'}]

#Specify image folder 
folder = r''

#Import image paths from camera 0 and 1
images = []

for i in ['0','1','2','3','4','5']: #specify camera id
    for path in glob.glob(folder + '/*_Cam'+i+'_*.jpg'):
        images.append(path)

#Specify output destination 
directory = r''

projects = []

#Import each model's checpoints
for k in Models:
         detector = Detector(model= k['model'], model_type = k['model_type'])
         projects.append(detector)
    
for imagePath in images : 
    
    start_time = time.time()

    img = cv2.imread(imagePath) 

    #Rotate image (optional)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) #90 degrees clockwise rotation 

    #specify image name
    split_path = imagePath.split('/')
    N = len(split_path)
    split_image_name = split_path[N-1].split('_') #image name 

    stream_name = split_image_name[0]+'_'+split_image_name[1]+'_'+split_image_name[2]
    image_id = split_image_name[4]
    Cam_id = split_image_name[5]

    name = stream_name+'_'+image_id+'_'+Cam_id

    outputs = []
    
    for detector in projects:
        #Create predictions for current image
        out = detector.onImage(img)
        outputs.append({'detector': detector, 'outputs': out})

    for row in outputs:
        #Drow predictions 
        output = row['detector'].output(img,row['outputs'],name,directory)
        img = output
    
    out_name = name+'_output.jpg' 
    cv2.imwrite(os.path.join(directory, out_name), img) #save the output
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

    

   

