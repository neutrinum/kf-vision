import cv2 as cv
import numpy as np 
from numpy.linalg import inv
import itertools
import scipy as sp
import scipy.linalg as spla
#import control
from collections import deque

import warnings





class KalmanFilter:
    def __init__(self,x0,A,B,C,Kfx,N):
        self.x_pred = x0
        self.x_filt = x0
        self.A = A
        self.B = B
        self.C = C
        self.Kfx = Kfx
        self.N = N # number of predictions
        self.Phix, self.Gamma_B = self.condensate(A,B,C,N)
        self.nz = C.shape[0]
        self.nx = A.shape[0]
        self.nu = B.shape[1]
    
    def condensate(self,A,B,C,N):
        nz = C.shape[0]
        nx = A.shape[0]
        #nu = B.shape[1]

        Phix = np.zeros((nz*N,nx))
        Gamma_B = np.zeros((nz*N,nx*N))

        for i in range(N):
            Phix[i*nz:(i+1)*nz,:]= C @ self.matMul(A,(i+1))
            for j in range(i+1):
                Gamma_B[i*nz:(i+1)*nz,(i-j)*nx:(i-j+1)*nx] = C @ self.matMul(A,j)
    
        return Phix, Gamma_B

    def filter(self,y):
        x_pred, C, Kfx = self.x_pred, self.C, self.Kfx

        #innovation
        y_pred = C @ x_pred
        innov = y - y_pred

        # filter
        self.x_filt = x_pred + Kfx @ innov

        return self.x_filt

    def predict(self,u,constr):
        x_filt, A, B = self.x_filt, self.A, self.B
        Phix, Gamma_B, N = self.Phix, self.Gamma_B, self.N

        # Next state prediction
        self.x_pred = A @ x_filt + B * u

        # Border constraint
        if abs(self.x_pred[0,0]) > constr:
            np.clip(self.x_pred[0],-constr,constr,out=self.x_pred[0])
            self.x_pred[1,0]=0 # speed update
            self.x_pred[4,0]=0 # disturbance update
        if abs(self.x_pred[2,0]) > constr:
            np.clip(self.x_pred[2],-constr,constr,out=self.x_pred[2])
            self.x_pred[3,0]=0
            self.x_pred[5,0]=0
        
        # N_pred next output predictions
        u_N = np.tile(B * u,(N,1))
        y_N = Phix @ x_filt + Gamma_B @ u_N
        #y_N = Gamma_B @ u_N

        # Border constraint
        np.clip(y_N,-constr,constr,out=y_N)

        return self.x_pred,y_N

    def matMul(self,M,n):
        if n == 0:
            return np.eye(M.shape[0])
        else:
            return M @ self.matMul(M,n-1)



def nothing(x):
    pass


def hlsRange(img,colorRange,kernel):
    
    hls =  cv.cvtColor(img,cv.COLOR_BGR2HLS)
    binary = cv.inRange(hls, colorRange[0], colorRange[1])
    #kernel = np.ones((5,5),np.uint8)
    
    binary = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    binary = cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel)

    return binary


def findLongestLines(lines,n):

    lengths=[]
    if lines is not None:
        for line in lines:
            lengths.append(cv.norm(line[0][0:2]-line[0][2:4]))

    nMax = n if len(lengths)>n else len(lengths)

    ind = np.argpartition(lengths, -nMax)[-nMax:]

    if lines is not None:
        return lines[ind], np.argmin(np.array(lengths)[ind])

    else:
        return lines, None


def intersectLines(a,b):
    # extract 2 endpoints of each segment
    a1,a2 = a[0:2],a[2:4]
    b1,b2 = b[0:2],b[2:4]

    # define vector from end to end of each segment
    va = a1-a2
    vb = b1-b2

    # cross prod to check if they intersect close
    cross = va[0]*vb[1] - va[1]*vb[0]
    #print(cross,' ',cross/(cv.norm(va)*cv.norm(vb)))
    
    if abs(cross/(cv.norm(va)*cv.norm(vb))) < 1e-3:
        return None,None,None # parallel lines

    ab = a1-b1
    t = (ab[0]*vb[1] - ab[1]*vb[0])/cross
    u = (ab[0]*va[1] - ab[1]*va[0])/cross

    p = a1 - t*va
    
    # cross_a = a1[0]*a2[1] - a1[1]*a2[0]
    # cross_b = b1[0]*b2[1] - b1[1]*b2[0]

    # p = (cross_a*vb - cross_b*va)/cross

    # print(t,' ',u)
    # print(p)
    return p,t,u


def findIntersections(lines):

    if lines is None:
        return None,None
    if len(lines)<2:
        return None,None

    inter = []
    square = []
    for pair in itertools.combinations(lines,2):
        i,t,u = intersectLines(pair[0][0],pair[1][0])
        if i is not None:
            inter.append(i)
            if abs(t*u)<1.6: # not tested in enough cases, maybe should increase to 1.5
                square.append(1)
            else:
                square.append(0)

    return np.array(inter),np.array(square)


def orderPoints(points):
    # sort the points according to their x (horizontal) coordinate
    xSort = points[np.argsort(points[:,0]),:]

    # take 2 left and 2 right points
    left = xSort[:2, :]
    right = xSort[2:, :]

    # sort by y coordinate
    lSort = left[np.argsort(left[:,1]),:]
    rSort = right[np.argsort(right[:,1]),:]

    pointsSorted = np.array([lSort[0],rSort[0],rSort[1],lSort[1]])

    return pointsSorted


def projectOnImg(p3,img,R,t,intrinsics,distCoeffs,color2=255):
        vec, _ = cv.projectPoints(p3, R, t, intrinsics, distCoeffs)
        p2 = ( int(vec[0,0,0]), int(vec[0,0,1]))

        cv.circle(img, p2, 3, (255,color2,0), 3)
        return img


def drawCoordAxis(img,R,t,intrinsics,distCoeffs):
    vecZ, _ = cv.projectPoints(np.array([(0., 0., side/2)]), R, t, intrinsics, distCoeffs)
    vecY, _ = cv.projectPoints(np.array([(0., side/2, 0.)]), R, t, intrinsics, distCoeffs)
    vecX, _ = cv.projectPoints(np.array([(side/2, 0., 0.)]), R, t, intrinsics, distCoeffs)
    orig, _ = cv.projectPoints(np.array([(0., 0., 0.)]), R, t, intrinsics, distCoeffs)


    po = ( int(orig[0,0,0]), int(orig[0,0,1]))
    pz = ( int(vecZ[0,0,0]), int(vecZ[0,0,1]))
    py = ( int(vecY[0,0,0]), int(vecY[0,0,1]))
    px = ( int(vecX[0,0,0]), int(vecX[0,0,1]))

    cv.line(img, po, pz, (0,0,0), 3)
    cv.line(img, po, pz, (255,0,0), 2)
    cv.line(img, po, py, (0,0,0), 3)
    cv.line(img, po, py, (0,255,0), 2)
    cv.line(img, po, px, (0,0,0), 3)
    cv.line(img, po, px, (0,0,255), 2)

    return img


def createBirdSight(pMeas=None,pPred=None,mode=0):
    bird_sight = np.ones((int(side*0.6),int(side*0.6),3), np.uint8)*255
    cv.rectangle(bird_sight,(50,50),(int(side*0.55),int(side*0.55)),(0,0,0),2)
    cv.rectangle(bird_sight,(65,65),(int(side*0.55)-15,int(side*0.55)-15),(0,0,0))

    if pMeas is not None:
        offset=np.array([side*0.3,side*0.3]).reshape((2,1))

        if mode == 0:
            pMeas = np.flipud(pMeas)
            pPred = np.flipud(pPred)
            #np.clip(pPred,-230,230,out=pPred)

            cv.circle(bird_sight,tuple(pMeas.astype('int')+offset),3,(0,255,0),3)
            cv.circle(bird_sight,tuple(pPred.astype('int')+offset),3,(0,0,255),3)

        if mode == 1:
            for i in range(len(pMeas)-1):
                pm1 = np.flipud(pMeas[i].astype('int')+offset)
                pm2 = np.flipud(pMeas[i+1].astype('int')+offset)
                cv.line(bird_sight, tuple(pm1), tuple(pm2), (255-i*3,0,255),2)

                pp1 = np.flipud(pPred[i].astype('int')+offset)
                pp2 = np.flipud(pPred[i+1].astype('int')+offset)
                cv.line(bird_sight, tuple(pp1), tuple(pp2), (255-80+i,180-i*3,0),2)
        
        if mode == 2:
            for i in range(len(pMeas)):
                pm1 = np.flipud(pMeas[i].astype('int')+offset)
                cv.circle(bird_sight, tuple(pm1), 3, (255-i*4,255,0),3)

                pp1 = np.flipud(pPred[i].astype('int')+offset)
                cv.circle(bird_sight, tuple(pp1), 3, (255-i*4,0,255),3)

        return bird_sight

    else:
        return bird_sight
        


def c2dzoh(A,B,Ts):
 
    nx,nu = B.shape
    # M = [A B; zeros(nu,nx) zeros(nu,nu)]
    M = np.concatenate((np.concatenate((A,B),axis=1),np.zeros((1,nx+nu))))
    Phi = spla.expm(M*Ts)

    Abar = Phi[0:nx,0:nx]
    Bbar = Phi[0:nx,nx:nx+nu]

    return Abar,Bbar
    

def c2dstoch(A,B,Ts):
    
    # nx = size(A,1);
    nx = A.shape[0]
    nx2 = nx+nx

    # M = [-A' G*G'; zeros(nx,nx) A]
    M = np.concatenate((np.concatenate((-A.T,B@B.T),axis=1),
        np.concatenate((np.zeros((nx,nx)),A),axis=1)))

    Phi = spla.expm(M*Ts)
    #print(np.round(Phi,decimals=1))
    Abar = Phi[nx:nx2,nx:nx2].T
    Qbar = Abar @ Phi[0:nx,nx:nx2]

    return Qbar


def buildBcont(R):
    # gravity vector in table coordinate frame

    # R.T * [0,0,1].T
    gTable = R.T[:,-1]
 
    xComponent = gTable[0] # sin(theta_x)
    yComponent = gTable[1] # sin(theta_y)

    aX = 5*xComponent/7 # acceleration in X
    aY = 5*yComponent/7 # acceleration in Y
    B = np.array([0,aX,0,aY]).reshape((4,1))

    return B



# Config
np.set_printoptions(precision=6, suppress=True)


# Object and world parameters
side = 1000. # side of square
rBall = 83.33 # radius of ball
ballIn = False # ball detected (this is for the start of the program)

g = side/0.3*9.81 # gravity in pixels/s^2


# Initial parameters (rotation and translation)
R = np.array([1.,1.,0.]).reshape((3,1))
Rmat = np.eye(3)
t = np.array([0.,0.,1000.]).reshape((3,1))
			
# # camera quaternion
# qx = -0.2195502388	
# qy = -0.7090353848	
# qz = 0.4171316898	
# qw = 0.5244689401
# # table when impact
# # qx = -0.07679927896
# # qy = 0.04870233627
# # qz = 0.732674413
# # qw = 0.6744762099
# q = np.array([qz,qy,qz]).reshape(1,3)
# qnorm = cv.norm(q)
# theta_cam = 2*np.arctan2(qnorm,qw)
# qRodrigues = q*theta_cam
# Rcam,_ = cv.Rodrigues(qRodrigues)
# Rcam = np.array([[0.,  0., -1.],[ 1., 0., 0.],[0., -1., 0.]])
#Rgnd = np.array([[0.,  1., 0.],[ 0., 0., 1.],[1., 0., 0.]])
Rcam = np.array([[  -0.48323,  0.48883, -0.72631],
    [     0.875,  0.24193, -0.41934],
    [ -0.029269, -0.83816, -0.54464]])
Rgnd = np.eye(3)



'''
# Linear state space
'''
# system
Ts = 0.02
Ac = np.zeros((4,4))
Ac[0,1]=1
Ac[2,3]=1
Bc = buildBcont(Rgnd @ Rcam @ Rmat) # B must be built every iteration
Abar,Bbar = c2dzoh(Ac,Bc,Ts) # discretize
C = np.zeros((2,4))
C[0,0]=1
C[1,2]=1


# noise modelling ############################################
sigma_w = 1 # disturbance noise (unkown control actions)
sigma_v = 0.005 # measurement noise (error from image analysis)
beta = .5 # time constant of the estimated disturbance

# extended system 
Bd = np.zeros((4,2)) # disturbance input matrix
Bd[1,0]=1
Bd[3,1]=1

Ace = np.concatenate((np.concatenate((Ac,Bd),axis=1),
    np.concatenate((np.zeros((2,4)),-np.eye(2)*beta),axis=1)),axis=0)
Bce = np.concatenate((Bc,np.zeros((2,1))))
Ce = np.concatenate((C,np.zeros((2,2))),axis=1)

Aebar,Bebar = c2dzoh(Ace,Bce,Ts)

# noise covariance matrices
Rv = np.eye(2)*sigma_v**2
Rbar = Rv/Ts

Bu = np.zeros((6,2)) # noise (e) input matrix (continuous)

Bu[1,0] = sigma_w
Bu[3,1] = sigma_w
Bu[4,0] = sigma_w*np.sqrt(2*beta)
Bu[5,1] = sigma_w*np.sqrt(2*beta)

Qbar = c2dstoch(Ace,Bu,Ts)
#Qbar = np.diag([0,0,0,0,1,1])
print('Qbar\n',np.round(Qbar,3))

# kalman filter gains
P = spla.solve_discrete_are(Aebar.T,Ce.T,Qbar,Rbar)


Re = Ce @ P @ Ce.T + Rbar
Kfx = P @ Ce.T @ inv(Re)
print(np.round(Kfx,2))


# # Kalman Filter setup
Npred = 10
x0 = np.zeros((6,1))
x_pred = x0
KF = KalmanFilter(x0,Aebar,Bebar,Ce,Kfx,Npred)

ballTable = np.zeros((3,1)) # initial position estimate of ball



# Camera and objects parameters

height = 240 # image processing size
width = 320

f = width/2 # 90 deg FOV

    # intrinsic parameters camera matrix
intrinsics = np.array([[f,0,width/2],[0,f,height/2],[0,0,1]],dtype="double")

centerModel = np.array([side/2.,side/2.,0])
modelPoints = np.array([[0.,0.,0.],[0.,side,0.],[.99*side,side,0.],[.99*side,0.,0.]])-centerModel
distCoeffs = np.zeros((4,1))
centerImg = np.array([width/2,height/2],dtype=np.int)

buffLen = 80
y_measBuff = deque([],maxlen=buffLen)
y_filtBuff = deque([],maxlen=buffLen)



# Image processing parameters

bird_sight = createBirdSight()   
bird_sight2 = createBirdSight() 

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))


'''
# Image analysis and main program
'''

cap = cv.VideoCapture('data/icub_table_test_1.mp4')
nFrame = 1 # initial frame
cap.set(1,nFrame)



# img = cv.imread('data/icubtable5.png')

prnt = True
while (True):

    ret, frame = cap.read()

    if ret:
        img = cv.resize(frame,(width, height), interpolation = cv.INTER_AREA)
    else:
        break

    imgDraw = img.copy()

    t1 = cv.getTickCount()
    # 0.3 0.025

    blr = cv.GaussianBlur(img,(3,3),0)

    ## Red Region
    #redRange1 = [np.array([0,5,100]), np.array([10,200,255])]
    redRange2 = [np.array([165,30,100]), np.array([180,200,255])]
    #binRed1 = hlsRange(blr,redRange1,kernel)
    binRed = hlsRange(blr,redRange2,kernel)
    #binRed = cv.bitwise_or(binRed1,binRed2)
    _,cntsRed,_ = cv.findContours(binRed,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    binDraw = binRed.copy()
    if cntsRed:
        hullsRed = [cv.convexHull(cnt) for cnt in cntsRed]

        maxAreaIndx = 0
        if len(hullsRed)>1:
            maxAreaIndx = np.argmax([cv.contourArea(hull) for hull in hullsRed])

        hullApprox = [cv.approxPolyDP(hullsRed[maxAreaIndx],int(width*0.01)+1,True)]
        nPointsHull = hullApprox[0].shape[0]
        # print(hullRed)

        cv.drawContours(binDraw,hullApprox,0,(127,127,127),5)
        # cv.putText(binDraw,str(hullApprox[0].shape),(10,10),cv.FONT_HERSHEY_PLAIN,1,[255,255,255])   

    segments = []


    ## Green Region
    greenRange = [np.array([35,30,80]), np.array([65,220,255])]
    binGreen = hlsRange(blr,greenRange,kernel)

    _,cntsGreen,_ = cv.findContours(binGreen,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    if cntsGreen and cntsRed: # if there is the ball and table
        x,y,w,h = cv.boundingRect(cntsGreen[0])
        (xc,yc),radius = cv.minEnclosingCircle(cntsGreen[0])

        if nPointsHull>4: # if more than 4 vertices are detected (ball at corner or part of table out of FOV)
            dist2Green = [cv.norm(np.array([xc,yc])-hullPt) for hullPt in hullApprox[0]]
            #segments1 = pickSegments1(hullApprox,)
            close2Green = np.argpartition(dist2Green,2)[:2]

            # if ball is close to a pair of vertices (ball at corner)
            if np.sum(np.array(dist2Green)[close2Green])/2 < radius+int(0.04*width): 
                #print(close2Green)
                for i in range(len(hullApprox[0])): # create segments from points
                    if i not in close2Green or (i+1)%nPointsHull not in close2Green:
                        segments.append( np.c_[hullApprox[0][i,0].reshape(1,2), 
                            hullApprox[0][(i+1)%nPointsHull,0].reshape(1,2)] ) 

    if cntsRed: # case in which there is no ball or it is not at one corner
        if not segments: # create segments from points
            for i in range(len(hullApprox[0])): # create segments from points
                segments.append( np.c_[hullApprox[0][i,0].reshape(1,2), 
                    hullApprox[0][(i+1)%nPointsHull,0].reshape(1,2)] )
        segments = np.array(segments).reshape(len(segments),1,4)
    else:
        segments = None

    # remove undesired segments
    sidesSquare,_ = findLongestLines(segments,4)


    # intersections of 4 (or less) longest lines, and square definition
    intersecRed,isSquareRed = findIntersections(sidesSquare)
    if intersecRed is not None and sum(isSquareRed)==4:
        squareRed = intersecRed[isSquareRed==1] # points of square
        modelPointsImg = orderPoints(squareRed) # sorted points of square
    else:
        modelPointsImg = None # Here Kalman Filter could estimate the position of the table
                              # For this, the model should be extended to include the angles of the table
                              # However the current implementation uses this as a warning 
                              # that indicates when the quality of the data is bad 
                              # for example the red surface is clipped due to extreme proximity to camera 
                              # or it is too out of the field of view
        warnings.warn('Poor data quality')
        print('bad',nFrame)

    # Find R and t
    #timePnP1 = cv.getTickCount()
    if modelPointsImg is not None: # if it is None, should use previous estimate or 0 for first case
        ret,R,t = cv.solvePnP(modelPoints,modelPointsImg,intrinsics,distCoeffs,cv.SOLVEPNP_AP3P)
            #rvec=R,tvec=t,useExtrinsicGuess=True,flags=cv.SOLVEPNP_ITERATIVE)
        #timePnP2 = cv.getTickCount()
        #dtPnP = (timePnP2 - timePnP1)/cv.getTickFrequency()
        #print(dtPnP)
        Rmat,_ = cv.Rodrigues(R)


    ## Ball localization

    if cntsGreen:
        if not ballIn:
            ballIn = True

        # Find ball projection vectors and angle between them
        centerBallImg = np.array([xc, yc]) - centerImg
        normCenterBall = cv.norm(centerBallImg)
        if normCenterBall != 0:
            centerBallImgUnit = centerBallImg/normCenterBall  
        else:
            centerBallImgUnit = np.array([1,0]) # when ball is in the center of the img

        p1 = centerBallImg - centerBallImgUnit * (radius-.5)
        p2 = centerBallImg + centerBallImgUnit * (radius-.5)

        v1 = np.append(p1,f)
        v1 = v1 / cv.norm(v1)
        v2 = np.append(p2,f)
        v2 = v2 / cv.norm(v2)

        angleV = np.arccos(np.dot(v1,v2))#/(cv.norm(v1)*cv.norm(v2)))
        kAngle = np.sqrt(1+1/np.tan(angleV/2)**2)
        # print(angleV/2*180/np.pi)
        v = v1+v2
        v = v/cv.norm(v)

        # print(Rmat)
        v_ext = np.dot(Rmat.T,v.T)
        # print(v_ext,' ',v_ext[2])
        o_ext = np.squeeze(-np.dot(Rmat.T,t))
        # print(o_ext,' ',o_ext[2])
        rBall = o_ext[2]/(1-kAngle*v_ext[2])
        #print(rBall)
        ballTable = o_ext+kAngle*rBall*v_ext

        # measurement of X and Y position on the board
        y_meas = ballTable[0:2].reshape(2,1)
    else:
        # if couldn't obtain data, take previous estimate from KF or place it at center

        if not ballIn: # guess at position 0
            y_meas = ballTable[0:2].reshape(2,1) 
        else: # take previous KF prediction
            ballTable = np.concatenate(((Ce @ x_pred).reshape((1,2)),np.array(rBall).reshape((1,1))),axis=1)
            y_meas = Ce @ x_pred
        



    # # Kalman filter

    x_filt = KF.filter(y_meas)
    y_filt = Ce @ x_filt

    Bc = buildBcont(Rgnd @ Rcam @ Rmat)
    Bce = np.concatenate((Bc,np.zeros((2,1))))
    _,Bebar = c2dzoh(Ace,Bce,Ts)
    KF.B = Bebar

    x_pred,y_N = KF.predict(-g,(side-rBall)/2)




    # Draw results
    # imgDraw = img.copy()

    if cntsGreen:

        # Future Npred predictions
        for n in range(Npred):
            pred = np.zeros((1,3))
            #print(pred)
            pred[0,0:2] = y_N[2*n:2*(n+1)].reshape(1,2)
            pred[0,2]=rBall
            imgDraw = projectOnImg(pred,imgDraw,R,t,intrinsics,distCoeffs,255-n*25)


        # Surface estimated pose
        imgDraw = drawCoordAxis(imgDraw,R,t,intrinsics,distCoeffs)


        # Info of ball
        ball_center_in_table = ballTable.reshape(1,3)
        v_ball1, _ = cv.projectPoints(ball_center_in_table, R, t, intrinsics, distCoeffs)
        v_ball2, _ = cv.projectPoints(ball_center_in_table-np.array([(0,0,rBall)],dtype='float64'), R, t, intrinsics, distCoeffs)
        pe1 = ( int(v_ball1[0,0,0]), int(v_ball1[0,0,1]))
        pe2 = ( int(v_ball2[0,0,0]), int(v_ball2[0,0,1]))

        # center to table cyan line
        cv.line(imgDraw, pe1, pe2, (255,255,0), 3)
        # cv.putText(imgDraw,'c',(pe1[0]+5,pe1[1]+5),cv.FONT_HERSHEY_PLAIN,1,[0,0,0],3)
        # cv.putText(imgDraw,'c',(pe1[0]+5,pe1[1]+5),cv.FONT_HERSHEY_PLAIN,1,[255,255,0])
        # cv.putText(imgDraw,'b',(pe2[0]+5,pe2[1]+5),cv.FONT_HERSHEY_PLAIN,1,[0,0,0],3)
        # cv.putText(imgDraw,'b',(pe2[0]+5,pe2[1]+5),cv.FONT_HERSHEY_PLAIN,1,[255,255,0])


        #cv.drawContours(imgDraw, cnts, -1, (0,255,0), 3)
        # bounding box
        cv.rectangle(imgDraw,(x,y),(x+w,y+h),(0,255,0),2)
        #cv.circle(imgDraw,centerCircle,radius,(255,0,0),2)

        # vertices of ellipse
        # cv.circle(imgDraw, tuple(p1.astype(int)+centerImg), 3, [0,0,0],2)
        # cv.circle(imgDraw, tuple(p1.astype(int)+centerImg), 3, [0,128,0])
        # cv.circle(imgDraw, tuple(p2.astype(int)+centerImg), 3, [0,0,0],2)
        # cv.circle(imgDraw, tuple(p2.astype(int)+centerImg), 3, [0,128,0])


    # Red surface
    lsd = cv.createLineSegmentDetector() # just for its function of drawing
    if segments is not None:
        imgDraw = lsd.drawSegments(imgDraw,sidesSquare)

    if intersecRed is not None:
        for v in range(len(intersecRed)):
            colorDraw = int(255*isSquareRed[v])
            cv.circle(imgDraw, tuple(intersecRed[v].astype('int')), 3, [colorDraw,colorDraw,colorDraw])
        if modelPointsImg is not None: # 4 points from the square
            for i in range(len(modelPointsImg)):
                cv.putText(imgDraw,str(i),tuple(modelPointsImg[i].astype('int')+5),cv.FONT_HERSHEY_PLAIN,1.5,[0,0,0],3)
                cv.putText(imgDraw,str(i),tuple(modelPointsImg[i].astype('int')+5),cv.FONT_HERSHEY_PLAIN,1.5,[255,255,255])

    
    # Upper view
    kPred=0
    if cntsGreen:
        #bird_sight = createBirdSight(ballTable[0:2].reshape((2,1))/2,y_N[2*kPred:2*(kPred+1)]/2) 
        bird_sight = createBirdSight(y_meas.reshape((2,1))/2, y_filt/2)

        y_measBuff.append(y_meas.reshape((2,1))/2)
        y_filtBuff.append(y_filt/2)

    else:
        bird_sight = createBirdSight()   
        bird_sight2 = createBirdSight()  

    if len(y_measBuff) > 1:
            bird_sight2 = createBirdSight(y_measBuff, y_filtBuff,1)


    t2 = cv.getTickCount()

    cv.imshow('frame',frame)
    cv.imshow('img',imgDraw)
    #cv.imshow('green',binGreen)
    cv.imshow('red',binDraw)
    #cv.imshow('longest',imgLongest)
    cv.imshow('bird sight',bird_sight2)


    t3 = cv.getTickCount()

    t12 = (t2-t1)/cv.getTickFrequency()
    t23 = (t3-t2)/cv.getTickFrequency()

    print(t12,' ',t23)

    print(nFrame)


    nFrame += 1
    # if nFrame>159:
    #     cv.waitKey(1000)
    #     print(Rcam @ Rmat)


    k = cv.waitKey(3) & 0xFF
    if k == 27:
        break


# cv.imwrite('E:/MScEE/Neurorobotics/results/result11a.png',bird_sight2)
# cv.imwrite('E:/MScEE/Neurorobotics/results/result11b.png',imgDraw)
#cv.imwrite('E:/MScEE/Neurorobotics/results/table1_img.png',imgDraw)
cv.waitKey(0)
cap.release()
cv.destroyAllWindows()

