"""
Το συγκεκριμένο αρχείο περιέχει συναρτήσεις απαραίτητες για την εκτέλεση των αρχείων 3d_intersection.py 
και 3d_intersection_WGS84.py και θα πρέπει να βρίκεται στον ίδιο φάκελο με τα αρχεία αυτά.
"""

import numpy as np
import math  

pi = math.pi
sin = math.sin
cos = math.cos

#Convert (φ,λ,h) to (Χ,Υ,Ζ)
def convert2XYZ(phi,lamda,h):
    
  #WGS84 Parameters :
  a = 6378137.000
  e2 = 0.0066943799
  
  N = a/math.sqrt(1-e2*pow(sin(phi*pi/180),2))       
  X = (N + h)*cos(phi*pi/180)*cos(lamda*pi/180)
  Y = (N + h)*cos(phi*pi/180)*sin(lamda*pi/180)
  Z = ((1-e2)*N + h)*sin(phi*pi/180)

  return [X,Y,Z]

#Convert (Χ,Υ,Ζ) to (φ,λ,h)
def convert2phi_lamda_h(X,Y,Z):
    
  #WGS84 Parameters :
  a = 6378137.000
  e2 = 0.0066943799
  e_tonos2 = 0.0067394964
  
  if X*Y != 0 :
    lamda = math.atan(Y/X) #rad
    if X<0:
      lamda+=pi
    elif X>0 and Y<0:
      lamda+=2*pi

  elif X*Y == 0 and Y == 0:
    if X >= 0 :
      lamda = 0
    else :
      lamda = (3*pi)/2
  
  elif X*Y == 0 and X == 0 and Y!=0:
    if Y<0 : 
      lamda = pi
    else : 
      lamda = 2*pi

  #Normalize lamda to (-pi,pi)
  if lamda > pi or lamda == -pi:
    lamda = -pi + abs(lamda+pi)% (abs(-pi) + abs(pi))
  if lamda <-pi or lamda == pi:
    lamda = pi - abs(lamda+pi)% (abs(-pi) + abs(pi))

  lamda = lamda*180/pi #degrees

  #Compute φ :
  phi_ini = math.atan((Z*(1+e_tonos2))/math.sqrt((X**2)+(Y**2))) #Initial value of φ
  dphi = 1

  while abs(dphi) > 0.0001:
      N = a/math.sqrt(1-e2*pow(sin(phi_ini),2))
      phi = math.atan((Z+e2*N*sin(phi_ini))/math.sqrt((X**2)+(Y**2)))
      dphi = phi-phi_ini
      phi_ini = phi

  h = (Z/sin(phi))-(1-e2)*N #m
  phi = phi*180/pi #degrees
  
  return [phi,lamda,h]

#Convert WGS 84 to EGSA 87:
def WGS84_2_EGSA87(X,Y,Z) :
   
  #Translation vector :
  DX = 200 #m
  DY = -75 #m
  DZ = -246 #m
    
  #Translate to EGSA 87 :
  X = X+DX
  Y = Y+DY
  Z = Z+DZ

  #GRS80 Parameters :
  a = 6378137.000
  e2 = 0.006694380
  e_tonos2 = 0.006739497

  #Compute φ,λ,h to GRS80 :
    
  if X*Y != 0:
    lamda = math.atan(Y/X) #rad
    if X<0:
      lamda+=pi
    elif X>0 and Y<0:
      lamda+=2*pi

  elif X*Y == 0 and Y == 0:
    if X >= 0 :
      lamda = 0
    else :
      lamda = (3*pi)/2
  
  elif X*Y == 0 and X == 0 and Y!=0:
    if Y<0 : 
      lamda = pi
    else : 
      lamda = 2*pi

  #Normalize lamda to (-pi,pi)
  if lamda > pi or lamda == -pi:
    lamda = -pi + abs(lamda+pi)% (abs(-pi) + abs(pi))
  if lamda <-pi or lamda == pi:
    lamda = pi - abs(lamda+pi)% (abs(-pi) + abs(pi))
        
  #Compute φ :
  phi_ini = math.atan((Z*(1+e_tonos2))/math.sqrt((X**2)+(Y**2))) #Initial value of φ
  dphi = 1

  while abs(dphi) > 0.0001:
        N = a/math.sqrt(1-e2*pow(sin(phi_ini),2))
        phi = math.atan((Z+e2*N*sin(phi_ini))/math.sqrt((X**2)+(Y**2)))
        dphi = phi-phi_ini
        phi_ini = phi #rad

  h = (Z/sin(phi))-(1-e2)*N #m
  
  #TM 87 parametes:
  lamda_0 = 24*pi/180 #rad
  c = 500000 #m
  m0 = 0.9996

  #Calculate x,y (E,N) :
  t = np.tan(phi)
  h_2 = e_tonos2*(cos(phi)**2)
  Dl = lamda - lamda_0
  
  #GRS 80 parameters:
  A0 = 1.000006345
  A1 = -2.5188441*(10**-3)
  A2 = 5.2871167*(10**-6)
  A3 = -1.0357890*(10**-8)

  k = 6367408.748 #m

  #Calculate S_phi 
  S_phi = k*(A0*phi+A1*sin(2*phi)+(A2/2)*sin(4*phi)+(A3/3)*sin(6*phi))
  
  n0 = ((Dl**2)/2)*sin(phi)*cos(phi)
  n1 = ((Dl**4)/24)*sin(phi)*(cos(phi)**3)*(5-(t**2)+9*h_2+4*(h_2**2))
  n2 = ((Dl**6)/720)*sin(phi)*(cos(phi)**5)*(61-58*(t**2)+(t**4)+270*h_2-330*(t**2)*h_2
        +445*(h_2**2)+324*(h_2**3)-680*(t**2)*(h_2**2)+88*(h_2**4)-600*(t**2)*(h_2**3)-192*(t**2)*(h_2**4))
  n3 = ((Dl**8)/40320)*sin(phi)*(cos(phi)**7)*(1385-3111*(t**2)+543*(t**4)-(t**6))
  
  y = m0*S_phi+m0*N*(n0+n1+n2+n3) #m

  e0 = Dl*cos(phi)
  e1 = (((Dl**3)*(cos(phi)**3))/6)*(1-(t**2)+h_2)
  e2 = (((Dl**5)*(cos(phi)**5))/120)*(5-18*(t**2)+(t**4)+14*h_2-58*(t**2)*h_2+13*(h_2**2)+4*(h_2**3)
        -64*(t**2)*(h_2**2)-24*(t**2)*(h_2**3))
  e3 = (((Dl**7)*(cos(phi)**7))/5040)*(61-479*(t**2)+179*(t**4)-(t**6))

  x = m0*N*(e0+e1+e2+e3)+c #m

  return [x,y]

#Convert EGSA 87 to WGS 84:
def EGSA87_2_WGS84(x,y):
  #TM 87 parametes:
  lamda_0 = 24*pi/180 #rad
  c = 500000 #m
  m0 = 0.9996

  #GRS 80 parameters:
  a = 6378137.000
  e2 = 0.006694380
  e_tonos2 = 0.006739497

  A0 = 1.000006345
  A1 = -2.5188441*(10**-3)
  A2 = 5.2871167*(10**-6)
  A3 = -1.0357890*(10**-8)

  k = 6367408.748 #m

  #Calculate phi_tonos:
  S_phi = y/m0
  phi_ini = S_phi/(k*A0)
  dphi = 1

  while abs(dphi) > 0.0001:
    phi_tonos = S_phi/(k*A0)-(A1*sin(2*phi_ini))/A0 - (A2*sin(4*phi_ini))/(2*A0) - (A3*sin(6*phi_ini))/(3*A0) 
    dphi = phi_tonos - phi_ini
    phi_ini = phi_tonos
  
  E_tonos = x-c
  t_tonos = math.tan(phi_tonos)
  h_tonos2 = e_tonos2*(cos(phi_tonos)**2)
  N_tonos = a/math.sqrt(1-e2*pow(sin(phi_tonos),2))
  ro_tonos = a*(1-e2)/math.sqrt((1-e2*pow(sin(phi_tonos),2))**3)
  
  #Compute phi in GRS 80:
  p1 = -t_tonos*(E_tonos**2)/(2*ro_tonos*N_tonos*(m0**2))
  p2 = (t_tonos*(E_tonos**4)/(24*ro_tonos*(N_tonos**3)*(m0**4)))*(5+3*(t_tonos**2)
        +h_tonos2-4*(h_tonos2**2)-9*(t_tonos**2)*h_tonos2)
  p3 = -(t_tonos*(E_tonos**6)/(720*ro_tonos*(N_tonos**5)*(m0**6)))*(61+90*(t_tonos**2)
        +45*(t_tonos**4)+46*h_tonos2-252*(t_tonos**2)*h_tonos2-3*(h_tonos2**2)+100
        *(h_tonos2**3)-66*(t_tonos**2)*(h_tonos2**2)-90*(t_tonos**4)*h_tonos2+88
        *(h_tonos2**4)+225*(t_tonos**4)*(h_tonos2**2)+84*(t_tonos**2)*(h_tonos2**3)
        -192*(t_tonos**2)*(h_tonos2**4))
  p4 = (t_tonos*(E_tonos**8)/(40320*ro_tonos*(N_tonos**7)*(m0**8)))*(1385+3633*(t_tonos**2)
        +4095*(t_tonos**4)+1575*(t_tonos**6))

  phi = (phi_tonos+p1+p2+p3+p4)*180/pi #degrees

  #Compute lamda in GRS 80:
  l1 = E_tonos/(N_tonos*m0)
  l2 = -(1/6)*((E_tonos/(N_tonos*m0))**3)*(1+2*(t_tonos**2)+h_tonos2)
  l3 = (1/120)*((E_tonos/(N_tonos*m0))**5)*(5+6*h_tonos2+28*(t_tonos**2)-3*(h_tonos2**2)
        +8*(t_tonos**2)*h_tonos2+24*(t_tonos**4)-4*(h_tonos2**3)+4*(t_tonos**2)
        *(h_tonos2**2)+24*(t_tonos**2)*(h_tonos2**3))
  l4 = -(1/5040)*((E_tonos/(N_tonos*m0))**7)*(61+662*(t_tonos**2)+1320*(t_tonos**4)
         +720*(t_tonos**6))

  lamda = (lamda_0 + (1/cos(phi_tonos))*(l1+l2+l3+l4))*180/pi #degrees

  #Calculate X,Y,Z in GRS 80:
  h = 0

  N = a/math.sqrt(1-e2*pow(sin(phi*pi/180),2))       
  X = (N + h)*cos(phi*pi/180)*cos(lamda*pi/180)
  Y = (N + h)*cos(phi*pi/180)*sin(lamda*pi/180)
  Z = ((1-e2)*N + h)*sin(phi*pi/180)

  #Translation vector :
  DX = -200 #m
  DY = 75 #m
  DZ = 246 #m
    
  #Translate to WGS 84 :
  X = X+DX
  Y = Y+DY
  Z = Z+DZ

  phi,lamda,h = convert2phi_lamda_h(X,Y,Z)
  return [phi,lamda]

#Create a 4x4 EulerXYZ transformation matrix
def get_T_matrix(Rx, Ry, Rz, Tx, Ty, Tz):

    r11 = cos(Ry)*cos(Rz)
    r12 = (sin(Rx)*sin(Ry)*cos(Rz)) + (cos(Rx)*sin(Rz))
    r13 = -(cos(Rx)*sin(Ry)*cos(Rz)) + (sin(Rx)*sin(Rz))
    r21 = -(cos(Ry)*sin(Rz))
    r22 = -(sin(Rx)*sin(Ry)*sin(Rz)) + (cos(Rx)*cos(Rz))
    r23 = (cos(Rx)*sin(Ry)*sin(Rz)) + (sin(Rx)*cos(Rz))
    r31 = sin(Ry)
    r32 = -(sin(Rx)*cos(Ry))
    r33 = cos(Rx)*cos(Ry)

    Rt = np.array([[r11,r12,r13, Tx],[r21,r22,r23,Ty],[r31,r32,r33,Tz],[0,0,0,1]])
    return Rt

#Create a 4x4 EulerZYX transformation matrix
def get_T_ZYX_matrix(Rx, Ry, Rz, Tx, Ty, Tz):

    r11 = cos(Ry)*cos(Rz)
    r12 = sin(Rx)*sin(Ry)*cos(Rz)-cos(Rx)*sin(Rz)
    r13 = cos(Rx)*sin(Ry)*cos(Rz)+sin(Rx)*sin(Rz)
    r21 = cos(Ry)*sin(Rz)
    r22 = sin(Rx)*sin(Ry)*sin(Rz)+cos(Rx)*cos(Rz)
    r23 = cos(Rx)*sin(Ry)*sin(Rz)-sin(Rx)*cos(Rz)  
    r31 = -sin(Ry)
    r32 = sin(Rx)*cos(Ry)
    r33 = cos(Rx)*cos(Ry)

    Rt = np.array([[r11,r12,r13, Tx],[r21,r22,r23,Ty],[r31,r32,r33,Tz],[0,0,0,1]])
    return Rt

#Apply transformation 
def apply_transformation(Rt,X,Y,Z):
   
   r = np.dot(Rt, np.asarray([X,Y,Z,1]))

   return [r[0],r[1],r[2]]
