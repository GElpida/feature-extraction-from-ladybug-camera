"""
Το συγκεκριμένο αρχείο υπολογίζει τα κέντρα βάρους των μασκών που εξάγονται από το Detectron.py και
τα αποθηκεύει με κατάλληλο τρόπο σε αρχείο CSV ώστε να μπορούν να χρησιμοποιηθούν ως δεδομένα εισόδου 
των αρχείων 3d_intersection.py και 3d_intersection_WGS84.py.
"""

import glob
import cv2
import csv
import numpy as np

#Specify output folder 
folder = r''

#Import masks paths
masks = []

for path in glob.glob(folder + '/*_mask.jpg'):
      masks.append(path)

point = []
point_name = '' #Specify a name

for path in masks:

    info = {}

    #Export information from mask's name and store them in a list
    split_path = path.split('/')
    N = len(split_path)
    split_image_name = split_path[N-1].split('_')
    
    stream_name = split_image_name[0]+'_'+split_image_name[1]+'_'+split_image_name[2]
    image_id = split_image_name[3]
    Cam_id = split_image_name[4]
    
    for i in [0,1,2,3,4,5]:
     if Cam_id == 'Cam'+str(i):
          Cam_id = i
   
    info.update({'point_name':point_name, 'stream_name':stream_name, 'image_id':image_id, 'Cam_id':Cam_id})

    mask = cv2.imread(path,0) 

    #Reset image rotation (optional)
    mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE) #90 degrees counter-clockwise rotation 
    
    #Calculate centroid coordinates and store them in a list
    M = cv2.moments(mask)
    
    x_center = int(M["m10"] / M["m00"])
    y_center = int(M["m01"] / M["m00"])
    
    #In case of rectified images x_center = 'Rectified_x[px]' y_center = 'Rectified_y[px]'
    info.update({'Raw_x[px]':x_center, 'Raw_y[px]': y_center, 'Rectified_x[px]': '', 'Rectified_y[px]': ''})

    point.append(info)

#Save list as a csv file
csv_name = folder + '/'+point_name+'.csv' #specify csv name

with open(csv_name, 'w') as csvfile:
     keys = point[0].keys()
     writer = csv.DictWriter(csvfile, fieldnames=keys)
     writer.writeheader()
     for row in point:
          writer.writerow(row)
