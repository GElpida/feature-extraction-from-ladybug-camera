"""
Το συγκεκριμένο αρχείο υπολογίζει τις συντεταγμένες ενός σημείου αρχικά στο σύστημα της
Ladybug και κατόπιν στο WGS 84 και το προβολικό σύστημα ΕΓΣΑ 87. Βασική προϋπόθεση, να 
είναι γνωστές οι διορθωμένες εικονοσυντεταγμένες του σημείου σε 2 εικόνες που έχουν 
ληφθεί την ίδια χρονική στιγμή (ίδιο stream name και image id) από διαφορετικούς αισθητήρες
της Ladybug. 
"""

import csv
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

#Import point image coordinates
point_path = '' #specify csv path

with open(point_path, 'r') as file:
  points = list(csv.DictReader(file, delimiter = ','))

#Import Camera Exterior Orientation Parameters 
CSV_path = '../data/Ladybug5_plus/Ladybug5_plus_EOP.csv' #specify csv path

with open(CSV_path, 'r') as file:
    Cam_EOP = list(csv.DictReader(file, delimiter = '\t'))

#Import Camera Interior Orientation Parameters 
CSV_path = '../data/Ladybug5_plus/IOP_Ladybug5_plus.csv' #specify csv path

with open(CSV_path, 'r') as file:
  IOP = list(csv.DictReader(file, delimiter = '\t'))

#Import Ladybug Exterior Orientation Parameters
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
streams = []
image_ids = []
point_names = []

#Create a list with stream names
for po in points:
   stream = po['stream_name']
   if stream not in streams:
      streams.append(stream)

#Create a list with image ids
for po in points:
   id = po['image_id']
   if id not in image_ids:
      image_ids.append(id)

#Create a list with point names
for po in points:
   name = po['point_name']
   if name not in point_names:
      point_names.append(name)

#Find a 3d line from the given point and Sencor's center in Ladybug's frame
for stream in streams:
  for id in image_ids:
    
    #Create a list with all points of a local Ladybug frame
    points_local = []

    for name in point_names:

      cam_id = []
      line = []
      x1 = []
      y1 = []
      z1 = []
      
      for po in points:
        if po['point_name']==name and po['image_id']==id and po['stream_name']==stream:
          cam = po['Cam_id']
          cam_id.append(cam)

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
          x,y,z = apply_transformation(Rt_local,x,y,z)

          """
          3D equation of straight line in cartesian form: 
          (x-x1)/a = (y-y1)/b = (z-z1)/c

          where : a = x2-x1 , b = y2-y1 , c = z2-z1
          """

          a = x-Tx
          b = y-Ty
          c = z-Tz

          line.append([a,b,c])
          x1.append(Tx)
          y1.append(Ty)
          z1.append(Tz)
      
      if len(x1) < 2:
        continue
      
      else :

        #Find the Intersection of Two Lines in Ladybug's frame
              
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

        #Line 2:
        x1_2 = x1[1]
        y1_2 = y1[1]
        z1_2 = z1[1]
        a2 = line[1][0]
        b2 = line[1][1]
        c2 = line[1][2]

        s = (b2*(x1_2-x1_1)-a2*(y1_2-y1_1))/(b2*a1-b1*a2)
        t = (b1*(x1_1-x1_2)-a1*(y1_1-y1_2))/(b1*a2-b2*a1)
        k = z1_1+c1*s-z1_2-c2*t

        if abs(k) < 0.1 :
          
            Xp_1 = x1_1 + a1*s
            Yp_1 = y1_1 + b1*s
            Zp_1 = z1_1 + c1*s

            Xp_2 = x1_2 + a2*t
            Yp_2 = y1_2 + b2*t
            Zp_2 = z1_2 + c2*t

            Xp = (Xp_1+Xp_2)/2
            Yp = (Yp_1+Yp_2)/2
            Zp = (Zp_1+Zp_2)/2

            points_local.append([Xp,Yp,Zp,name])

            #Visualize lines:

            plt.rcParams["figure.figsize"] = [10, 10]
            plt.rcParams["figure.autolayout"] = True
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            ax.scatter(x1_1, y1_1, z1_1, c='green', s=10)
            ax.scatter(Xp_1, Yp_1, Zp_1, c='red', s=10)
            
            x, y, z = [x1_1, Xp_1], [y1_1, Yp_1], [z1_1, Zp_1]
            ax.plot(x, y, z, color='black')
            ax.text(x1_1,y1_1, z1_1, 'Cam'+cam_id[0])

            ax.scatter(x1_2, y1_2, z1_2, c='green', s=10)
            ax.scatter(Xp_2, Yp_2, Zp_2, c='red', s=10)

            x, y, z = [x1_2, Xp_2], [y1_2, Yp_2], [z1_2, Zp_2]
            ax.plot(x, y, z, color='black')
            ax.text(x1_2,y1_2, z1_2, 'Cam'+cam_id[1])

            x, y, z = [x1_1+a1, x1_2+a2], [y1_1+b1, y1_2+b2], [z1_1+c1, z1_2+c2]
            ax.scatter(x, y, z, c='b', s=10)

            ax.text(Xp,Yp,Zp,name)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            plt.show()
            
            #Convert point coordinates from Ladybug's frame to WGS 84
            #Tranformation matrix to WGS 84:
            for param in EOP : 
              if param['stream_name'] == stream and param['image_id'] == id:
                  phi = float(param['latitude[deg]'])
                  lamda = float(param['longitude[deg]'])
                  h = float(param['altitude_ellipsoidal[m]'])
                  Rx = float(param['roll[deg]'])*pi/180 
                  Ry = float(param['pitch[deg]'])*pi/180 
                  Rz = pi - float(param['heading[deg]'])*pi/180 

                  #Ladybug's Center in WGS'84
                  x0,y0,z0 = convert2XYZ(phi,lamda,h)
                  if [x0,y0,z0,id] not in Ladybug_center:
                     Ladybug_center.append([x0,y0,z0,id])

                  Rt = get_T_ZYX_matrix(Rx,Ry,Rz,x0,y0,z0)
                    
            X,Y,Z = apply_transformation(Rt,Xp,Yp,Zp)

            point_list.append([X,Y,Z,name])
        
            phi,lamda,h = convert2phi_lamda_h(X,Y,Z)
            print(name,' in WGS 84 :',phi,lamda,h)

            #Convert to EGSA 87
            Xgr,Ygr = WGS84_2_EGSA87(X,Y,Z)
            print(name,' in EGSA 87 :',Xgr,Ygr)

        else : print(name,' : No intersection')
    
    #Visualaze points in local Ladybug frame
    if len(points_local)>0:

      plt.rcParams["figure.figsize"] = [20, 20]
      plt.rcParams["figure.autolayout"] = True
      fig = plt.figure()
      ax = fig.add_subplot(projection="3d")
      
      for point in points_local: 
        x, y, z = point[0],point[1],point[2]
        ax.scatter(x, y, z, c='red', s=10)
        ax.text(x,y, z, point[3])

      ax.scatter(0, 0, 0, c='green', s=10)
      ax.text(0,0,0, 'Ladybug')

      ax.set_xlabel("x")
      ax.set_ylabel("y")
      ax.set_zlabel("z")
      ax.set_title('Points in Ladybug frame ('+id+')')
      
      #Calculate altitude and distance 
      for point in points_local: 
        x, y, z = point[0],point[1],point[2]

        D = np.sqrt(x**2+y**2)
        dz = z 

        print('Οριζόντια αποσταση ',point[3],'από Ladybug :',"%.1f" % D,'m')
        print('Υψομετρική διαφορά ',point[3]+'-Ladybug :',"%.1f" % dz,'m')
      plt.show()

      #Save point list as a csv file
      csv_name = point_path+'_'+stream+'_'+id+'.csv' #specify csv name

      with open(csv_name, 'w') as csvfile:
          keys = ['point_name','XL[m]','YL[m]','ZL[m]','X_wgs84[m]','Y_wgs84[m]','Z_wgs84[m]','φ_wgs84[deg]','λ_wgs84[deg]','h_wgs84[m]','x_egsa87[m]','y_egsa87[m]']
          writer = csv.DictWriter(csvfile, fieldnames=keys)
          writer.writeheader()

          for row in points_local:
              #Convert to WGS 84
              X,Y,Z = apply_transformation(Rt,row[0],row[1],row[2])

              #Convert to φ,λ,h
              phi,lamda,h = convert2phi_lamda_h(X,Y,Z)

              #Convert to EGSA 87
              Xgr,Ygr = WGS84_2_EGSA87(X,Y,Z)

              writer.writerow({'point_name':row[3],'XL[m]':row[0],'YL[m]':row[1],'ZL[m]':row[2],'X_wgs84[m]':X,'Y_wgs84[m]':Y,'Z_wgs84[m]':Z,'φ_wgs84[deg]':phi,'λ_wgs84[deg]':lamda,'h_wgs84[m]':h,'x_egsa87[m]':Xgr,'y_egsa87[m]':Ygr})

#Save point list as a csv file
csv_name = point_path+'_WGS84_and_EGSA87_1.csv' #specify csv name

with open(csv_name, 'w') as csvfile:
     keys = ['point_name','φ[deg]','λ[deg]','h[m]','x[m]','y[m]']
     writer = csv.DictWriter(csvfile, fieldnames=keys)
     writer.writeheader()

     for row in point_list:
        #Convert to φ,λ,h
        phi,lamda,h = convert2phi_lamda_h(row[0],row[1],row[2])

        #Convert to EGSA 87
        Xgr,Ygr = WGS84_2_EGSA87(row[0],row[1],row[2])

        writer.writerow({'point_name':row[3],'φ[deg]':phi,'λ[deg]':lamda,'h[m]':h,'x[m]':Xgr,'y[m]':Ygr})

#Visualaze all points in EGSA 87: 
plt.rcParams["figure.figsize"] = [20, 20]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

for point in point_list:
  x, y, z = point[0],point[1],point[2]

  #Convert to EGSA 87
  x,y = WGS84_2_EGSA87(x,y,z)
  z=0

  ax.scatter(x, y, z, c='red', s=10)
  ax.text(x,y, z, point[3])

for point in Ladybug_center:
  x, y, z = point[0],point[1],point[2]

  #Convert to EGSA 87
  x,y = WGS84_2_EGSA87(x,y,z)
  z=0

  ax.scatter(x, y, z, c='green', s=10)
  ax.text(x,y, z, 'L_'+point[3])

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title('Points in EGSA 87')

plt.show()
