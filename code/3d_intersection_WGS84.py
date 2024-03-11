"""
Το συγκεκριμένο αρχείο υπολογίζει τις συντεταγμένες ενός σημείου στο WGS84 και το ΕΓΣΑ 87. 
Βασική προϋπόθεση, να είναι γνωστές οι διορθωμένες εικονοσυντεταγμένες του 
σημείου σε 2 διαφορετικές εικόνες. Δεν θα πρέπει να υπάρχουν διαφορετικά σημεία με 
το ίδιο όνομα στη λίστα με τα σημεία.
"""

import csv 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

#Import point image coordinates
point_path = '' #specify csv path

with open(point_path, 'r') as file:
  points = list(csv.DictReader(file, delimiter = ','))

#Import Camera Exteriore Orientation Parameters 
CSV_path = '../data/Ladybug5_plus/Ladybug5_plus_EOP.csv' #specify csv path

with open(CSV_path, 'r') as file:
    Cam_EOP = list(csv.DictReader(file, delimiter = '\t'))

#Import Camera Interior Orientation Parameters 
CSV_path = '../data/Ladybug5_plus/IOP_Ladybug5_plus.csv' #specify csv path

with open(CSV_path, 'r') as file:
  IOP = list(csv.DictReader(file, delimiter = '\t'))

#Import Exterior Orientation Parameters
CSV_path = '../data/Ladybug5_plus/Kalamata_20220413-02_TBC_TMX_export_panorama.csv' #specify csv path

with open(CSV_path,'r') as file:
  EOP = list(csv.DictReader(file, delimiter = '\t'))

#Import Ladybug's Stream Specification file
CSV_path = '../data/Ladybug5_plus/Ladybug_Stream_Specifications.csv' #specify csv path

with open(CSV_path, 'r') as file:
  Stream_file = list(csv.DictReader(file, delimiter = '\t'))

#Add stream and image id keys in EOP file
for row in EOP:
    for id in Stream_file:
        split = row['panorama_file_name'].split('_')
        row.update({'stream_id': split[1],'image_id': split[2]})  

        #Add stream name key in EOP file
        if row['stream_id'] == id['stream_id']:
            row.update({'stream_name':id['stream_name']})

#Import everything from file transformations.py
from transformations import *

point_list = []
Ladybug_center = []
p_name = []

#Create a list with point names
for po in points:
   name = po['point_name']
   if name not in p_name:
      p_name.append(name)

#Find a 3d line from the given point and Sencor's center in WGS84
for name in p_name:
  
  camera_ids = []
  image_ids = []
  line = []
  x1 = []
  y1 = []
  z1 = []
  
  for po in points:
    if po['point_name']==name :
      cam = po['Cam_id']
      camera_ids.append(cam)
      image_ids.append(po['image_id'])

      #Compute x,y,z coordinates in Sencor's frame
      for param in IOP:
        if param['Cam_id'] == cam : 
          ppx = float(param['PPx[px]']) #principal point's x image coordinate in pixels
          ppy = float(param['PPy[px]']) #principal point's y image coordinate in pixels
          c = float(param['F[px]']) #focal length in pixels 

      z = 1 #m
      x = (float(po['Rectified_x[px]'])-ppx)*(z/c) #m
      y = (float(po['Rectified_y[px]'])-ppy)*(z/c) #m
          
      #Calculate transformation matrix from Sensor's frame (Virtual Pinhole Camera Model) to Ladybug's frame
      for param in Cam_EOP:
            if param['Cam_id'] == cam :
              Rx = float(param['Rx[rad]'])
              Ry = float(param['Ry[rad]'])
              Rz = float(param['Rz[rad]'])
              Tx = float(param['Tx[m]'])
              Ty = float(param['Ty[m]'])
              Tz = float(param['Tz[m]'])

              Rt_local = get_T_ZYX_matrix(Rx,Ry,Rz,Tx,Ty,Tz)
      
      #Compute x,y,z coordinates in Ladybug's frame
      x2,y2,z2 = apply_transformation(Rt_local,x,y,z)
      #Create transformation matrix to WGS 84 :
      for param in EOP : 
          if param['stream_name'] == po['stream_name'] and param['image_id'] == po['image_id']:
              phi = float(param['latitude[deg]'])
              lamda = float(param['longitude[deg]'])
              h = float(param['altitude_ellipsoidal[m]'])
              Rx = float(param['roll[deg]'])*pi/180 
              Ry = float(param['pitch[deg]'])*pi/180 
              Rz = pi - float(param['heading[deg]'])*pi/180
              
              #Ladybug's Center in WGS'84
              Xc,Yc,Zc = convert2XYZ(phi,lamda,h)
              
              #Ladybug's Center in EGSA 87 (Greek Grid)
              Xgr_c,Ygr_c = WGS84_2_EGSA87(Xc,Yc,Zc)
              Zgr_c = 0

              if [Xgr_c,Ygr_c,Zgr_c,param['image_id']] not in Ladybug_center:
                Ladybug_center.append([Xgr_c,Ygr_c,Zgr_c,param['image_id']])

              #Transformation matrix to WGS 84:
              Rt = get_T_ZYX_matrix(Rx,Ry,Rz,Xc,Yc,Zc)

      """
      3D equation of straight line in cartesian form: 
      (x-x1)/a = (y-y1)/b = (z-z1)/c

      where : a = x2-x1 , b = y2-y1 , c = z2-z1
      """

      #From Ladybug's frame to WGS 84:
      x2,y2,z2 = apply_transformation(Rt,x2,y2,z2)
      xpp,ypp,zpp = apply_transformation(Rt,Tx,Ty,Tz) #Principal point's coordinates in WGS 84 

      a = x2-xpp
      b = y2-ypp
      c = z2-zpp

      line.append([a,b,c])
      x1.append(xpp)
      y1.append(ypp)
      z1.append(zpp)
  
  if len(x1) < 2:
      continue
  
  else :

    #Find the Intersection of Two Lines in WGS 84
          
    """
    For Line 1 : 
    x = x1_1 + a1*s
    y = y1_1 + b1*s
    z = z1_1 + c1*s

    For Line 2 : 
    x = x1_2 + a2*t
    y = y1_2 + b2*t
    z = z1_2 + c2*t

    Intersection :
    x1_1 + a1*s = x1_2 + a2*t
    y1_1 + b1*s = y1_2 + b2*t
    z1_1 + c1*s = z1_2 + c2*t
    """

    #Line 1 :
    x1_1 = x1[0]
    y1_1 = y1[0]
    z1_1 = z1[0]
    a1 = line[0][0]
    b1 = line[0][1]
    c1 = line[0][2]

    #Line 2 :
    x1_2 = x1[1]
    y1_2 = y1[1]
    z1_2 = z1[1]
    a2 = line[1][0]
    b2 = line[1][1]
    c2 = line[1][2]

    s = (b2*(x1_2-x1_1)-a2*(y1_2-y1_1))/(b2*a1-b1*a2)
    t = (b1*(x1_1-x1_2)-a1*(y1_1-y1_2))/(b1*a2-b2*a1)
    k = z1_1+c1*s-z1_2-c2*t

    if k < 10 :
      
      X1 = x1_1 + a1*s
      Y1 = y1_1 + b1*s
      Z1 = z1_1 + c1*s

      X2 = x1_2 + a2*t
      Y2 = y1_2 + b2*t
      Z2 = z1_2 + c2*t

      X = (X1+X2)/2
      Y = (Y1+Y2)/2
      Z = (Z1+Z2)/2
      
      #Calculate altitude and distance

      D1 = np.sqrt((X-x1_1)**2+(Y-y1_1)**2)
      print('Οριζόντια αποσταση',name,'από Ladybug ('+image_ids[0]+') :',"%.1f" % D1,'m')
      D2 = np.sqrt((X-x1_2)**2+(Y-y1_2)**2)
      print('Οριζόντια αποσταση',name,'από Ladybug ('+image_ids[1]+') :',"%.1f" % D2,'m')
      
      #Convert to φ,λ,h:
      phi,lamda,h = convert2phi_lamda_h(X,Y,Z)
      print(name,' in WGS 84 :',phi,lamda,h)

      phi_1,lamda_1,h1 = convert2phi_lamda_h(x1_1,y1_1,z1_1)
      phi_2,lamda_2,h2 = convert2phi_lamda_h(x1_2,y1_2,z1_2)

      dh1 = h-h1
      print('Υψομετρική Διαφορά',name+'- Ladybug ('+image_ids[0]+') :',"%.1f" % dh1,'m')
      dh2 = h-h2
      print('Υψομετρική Διαφορά',name+'- Ladybug ('+image_ids[1]+') :',"%.1f" % dh2,'m')

      point_list.append([X,Y,Z,name])

      #Convert to EGSA 87:
      x_gr,y_gr = WGS84_2_EGSA87(X,Y,Z)
      z_gr = 0
      print(name,' in EGSA 87 :',x_gr,y_gr,z_gr)
  
      #Visualize lines:

      plt.rcParams["figure.figsize"] = [10, 10]
      plt.rcParams["figure.autolayout"] = True
      fig = plt.figure()
      ax = fig.add_subplot(projection="3d")

      ax.scatter(x1_1, y1_1, z1_1, c='green', s=10)
      ax.scatter(X1, Y1, Z1, c='red', s=10)
      
      x, y, z = [x1_1, X1], [y1_1, Y1], [z1_1, Z1]
      ax.plot(x, y, z, color='black')
      ax.text(x1_1,y1_1, z1_1, 'Cam'+camera_ids[0])

      ax.scatter(x1_2, y1_2, z1_2, c='green', s=10)
      ax.scatter(X2, Y2, Z2, c='red', s=10)
      
      x, y, z = [x1_2, X2], [y1_2, Y2], [z1_2, Z2]
      ax.plot(x, y, z, color='black')
      ax.text(x1_2,y1_2, z1_2, 'Cam'+camera_ids[1])
      
      x, y, z = [x1_1+a1, x1_2+a2], [y1_1+b1, y1_2+b2], [z1_1+c1, z1_2+c2]
      ax.scatter(x, y, z, c='b', s=10)

      ax.text(X,Y,Z,name)

      ax.set_xlabel("x")
      ax.set_ylabel("y")
      ax.set_zlabel("z")

      plt.show()

    else : print('No intersection')

#Save list as a csv file
csv_name = point_path+'_WGS84_and_EGSA87_2.csv' #specify csv name

with open(csv_name, 'w') as csvfile:
     keys = ['point_name','φ[deg]','λ[deg]','h[m]','x[m]','y[m]']
     writer = csv.DictWriter(csvfile, fieldnames=keys)
     writer.writeheader()
     for row in point_list:
        X = row[0]
        Y = row[1]
        Z = row[2]
        #Convert to φ,λ,h:
        phi,lamda,h = convert2phi_lamda_h(X,Y,Z)
        #Convert to EGSA 87:
        X,Y = WGS84_2_EGSA87(X,Y,Z)
        writer.writerow({'point_name':row[3],'φ[deg]':phi,'λ[deg]':lamda, 'h[m]':h,'x[m]':X,'y[m]':Y})

#Visualaze all points in EGSA 87: 
plt.rcParams["figure.figsize"] = [20, 20]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

for point in point_list:
  #Convert to EGSA 87:      
  x, y, z = point[0],point[1],point[2]
  x,y = WGS84_2_EGSA87(x,y,z)
  z = 0
  ax.scatter(x, y, z, c='red', s=10)
  ax.text(x,y, z, point[3])

for point in Ladybug_center:
  x, y, z = point[0],point[1],point[2]
  ax.scatter(x, y, z, c='green', s=10)
  ax.text(x,y, z, 'L_'+point[3])

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title('Points in EGSA 87')

plt.show()
