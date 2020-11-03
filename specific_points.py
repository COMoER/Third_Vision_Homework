import numpy as np
import cv2 as cv

def resize(img,scale=0.4):
    return cv.resize(img,None,fx=scale,fy=scale)
def size(img):
    return (img.shape[1],img.shape[0])
def imshow(img,name='out',scale=0.4,delay=0):
    cv.imshow(name,resize(img,scale))
    cv.waitKey(delay)

class camera(object):
    def __init__(self):
        self.cameraMatrix=np.zeros((3,3))
        self.distcoeffs=np.zeros(5)
    def default_arguments(self):

        #用标准参数
        self.cameraMatrix=np.array([[1114.1804893712708,0.0,1074.2415297217708],
                           [0.0,1113.4568392254073,608.6477877664104],
                           [0.0,0.0,1.0]])

        #设定无畸变
        self.distcoeffs=np.float32([0,0,0,0,0])
    def Pnp(self,p1,p,dpt):  #p1,p是像素坐标系，p应变换为世界（基准相机）坐标系(3D)
        #变换为相机坐标
        p_tmp=cv.undistortPoints(p,self.cameraMatrix,self.distcoeffs)
        p_cam=np.zeros((p.shape[0],3))
        for i in range(p.shape[0]):
            dp=dpt[int(p[i,1])-1,int(p[i,0])-1]
            p_cam[i]=np.float32([p_tmp[i,:,0]*dp,p_tmp[i,:,1]*dp,dp])
        #print(p1[0])
        _,rvec,tvec=cv.solvePnP(p_cam,p1,self.cameraMatrix,self.distcoeffs)
        #imp,_=cv.projectPoints(p_cam,rvec,tvec,self.cameraMatrix,self.distcoeffs)
        #print(imp[0])
        #rM,_=cv.Rodrigues(rvec)
        #T=np.concatenate((rM,tvec.reshape(-1,1)),axis=1)
        return rvec,tvec
    def reproject(self,img,dpt,rvec,tvec,new_size,x,y):
        H, W, _ = img.shape
        #获得每一点的坐标
        #meshgrid可以生成一系列（x，y）坐标
        psx,psy=np.meshgrid(np.arange(1,W+1,1),np.arange(1,H+1,1),sparse=False,indexing='xy')
        ps=np.stack((psx,psy),axis=2)
        psf=np.float32(ps).reshape(-1,2)
        #计算每一点相机坐标系下坐标，dpt是深度矩阵
        wp=cv.undistortPoints(psf,self.cameraMatrix,self.distcoeffs).reshape(-1,2)
        wp=np.concatenate((wp,np.ones((H*W,1))),axis=1)
        wp=wp*dpt.reshape(-1,1)
        #获得各点在目标坐标系中的像素坐标
        imp,_=cv.projectPoints(wp,rvec,tvec,self.cameraMatrix,self.distcoeffs)
        imp=imp.reshape(H,W,2)
        imp[np.isnan(imp)]=0  #去除预测为nan的点，全部投影到(0,0)点
        imp=imp.astype(np.int)
        #平移
        imp[:,:,0]+=x
        imp[:,:,1]+=y
        out=np.zeros((new_size[1],new_size[0],3),dtype=np.uint8)  #图片是unint8类型

        #将原点的颜色值复制到对应坐标上
        out[imp[:,:,1],imp[:,:,0]]=img[psy-1,psx-1]


        return out
    def getStitch(self,dpts,imgs,new_size=(6000,2500),x=1400,y=500):
        cc=Stitch(imgs)

        kps,descs=cc.getkeypoints()
        #cc.drawKeyPointsAndShow()
        # 特征点匹配
        matcher=cv.BFMatcher()
        matches_mr = matcher.knnMatch(descs[1], descs[0],2)
        matches_ml = matcher.knnMatch(descs[2], descs[0],2)
        #得到剔除误点后的对应点
        p1,p2= self.getp(matches_mr, (kps[1],kps[0]),(imgs[1],imgs[0]),radio=0.6)
        p3,p4= self.getp(matches_ml, (kps[2],kps[0]),(imgs[2],imgs[0]),radio=0.65)

        rvec_1,tvec_1=self.Pnp(p2,p1,dpts[1])  #得到右边图像相机坐标系（设为世界坐标系)到中间图像相机坐标系的变换矩阵
        rvec_2,tvec_2=self.Pnp(p4,p3,dpts[2])  #得到左边图像相机坐标系（设为世界坐标系)到中间图像相机坐标系的变换矩阵

        #第一张转换
        warp_img=self.reproject(imgs[1],dpts[1],rvec_1,tvec_1,new_size,x,y)
        #imshow(warp_img,scale=0.2)

        #第二张转换
        warp_img_1=self.reproject(imgs[2],dpts[2],rvec_2,tvec_2,new_size,x,y)
        #imshow(warp_img_1,scale=0.2)

        #合并
        result=warp_img_1+warp_img
        #imshow(result,scale=0.2)
        #将中间图拼上去
        H,W,_=imgs[0].shape
        result[y:y+H,x:x+W,:]=imgs[0]
        imshow(result, scale=0.2)
        return result

    def getp(self,matches,kps,imgs,radio=0.6,thresh=3.0,view=False):
        better_matches=[]
        for m in matches:
            if m[0].distance/m[1].distance<radio:  #最近与次近距离比例小于radio时判断为可行对应点
                better_matches.append(m[0])

        if view:
            img = np.concatenate((imgs[0], imgs[1]), axis=1)
            cv.drawMatches(imgs[0], kps[0], imgs[1], kps[1], better_matches,img)
            imshow(img,scale=0.3)
            cv.destroyAllWindows()
        ml1 = lambda x: kps[0][x.queryIdx].pt
        ml2 = lambda x: kps[1][x.trainIdx].pt
        p1=np.float32([ ml1(m) for m in better_matches])
        p2=np.float32([ ml2(m) for m in better_matches])
        #用对极约束剔除误点
        H,status=cv.findFundamentalMat(p1,p2,cv.RANSAC,thresh)
        p1=np.float32([p1[i] for i in range(len(status)) if status[i]])
        p2=np.float32([p2[i] for i in range(len(status)) if status[i]])
        return p1,p2
    def calibrate(self,N=24,chess_size = (6, 9),length = 17,img_size = (2208, 1242)):

        corners=[]
        wps=[]
        for i in range(N):

            img=cv.imread('./calib/%d_orig.jpg'%i)
            img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            flag,corner=cv.findChessboardCorners(img,chess_size)
            if flag==False:
                continue
            flag,corner=cv.find4QuadCornerSubpix(img,corner,(5,5))
            if flag==False:
                continue
            print(i)
        #world postion
            wp=[]
            for y in range(1,chess_size[1]+1):
                for x in range(1,chess_size[0]+1):
                    wp.append((x*length,y*length,0))
            corners.append(corner)
            wp=np.float32(wp)
            wps.append(wp)

        cv.calibrateCamera(wps,corners,img_size,self.cameraMatrix,self.distcoeffs,flags=cv.CALIB_FIX_PRINCIPAL_POINT)
        print(self.cameraMatrix)
        print(self.distcoeffs)
class Stitch(object):
    def __init__(self,imgpair,nfeatures=500,scaleFactor=2):
        '''
        :param imgpair: 只供三张图片，顺序是中右左
        :param nfeatures:
        :param scaleFactor:
        '''
        self.reset(imgpair,nfeatures,scaleFactor)

    def reset(self,imgpair,nfeatures=500,scaleFactor=2):
        # Imgpair could only hold two images
        assert (len(imgpair) == 3)
        self.kps=[]
        self.describs = []
        self.imgset = imgpair
        self.orb = cv.ORB_create(nfeatures, scaleFactor)
    def getkeypoints(self):
        for img in self.imgset:
            img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            (kp,desc)=self.orb.detectAndCompute(img_gray,None)
            self.kps.append(kp)
            self.describs.append(desc)
        return self.kps,self.describs
    def drawKeyPointsAndShow(self,flag=True):
        flags=None
        if flag:  #能画出范围
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        for img0,kp in zip(self.imgset,self.kps):
            img=img0.copy()
            cv.drawKeypoints(img0, kp,img,flags=flags)
            cv.imshow('out',resize(img))
            cv.waitKey(0)
        cv.destroyAllWindows()
    def getbettermatches(self,matches,radio=0.4):
        better_matches=[]
        for m in matches:
            if m.distance * radio <= matches[0].distance:
                better_matches.append(m)
        return better_matches
    def getH(self,better_matches,kps,thresh=4.0):
        ml1 = lambda x: kps[0][x.queryIdx].pt
        ml2 = lambda x: kps[1][x.trainIdx].pt
        p1=np.float32([ ml1(m) for m in better_matches])
        p2=np.float32([ ml2(m) for m in better_matches])
        assert (len(better_matches)>4)
        (H,status)=cv.findHomography(p1,p2,cv.RANSAC,thresh)  #1->2
        return(better_matches,H,status)
    def H_move(self,H,x=0,y=0):
        #dst(x,y) = src((M11x+M12y+M13)/(M31x+M32y+M33), (M21x+M22y+M23)/(M31x+M32y+M33))
        #平移目标像素
        H[0,:]+=H[2,:]*x

        H[1,:]+=H[2,:]*y

        return H
    def getStitch(self,radio=0.5,thresh=4.0,new_size=(5000,2500),x=1400,y=500):
        self.matcher = cv.BFMatcher()
        # 给特征点排序
        matches_mr = sorted(self.matcher.match(self.describs[1], self.describs[0]), key=lambda x: x.distance)
        matches_ml = sorted(self.matcher.match(self.describs[2], self.describs[0]), key=lambda x: x.distance)
        assert(radio<=1 and radio>0)
        better_matches_mr=self.getbettermatches(matches_mr,radio)
        better_matches_ml=self.getbettermatches(matches_ml,radio)
        # 获得变换矩阵
        _,H1,_=self.getH(better_matches_mr,[self.kps[1],self.kps[0]],thresh)
        _,H2,_=self.getH(better_matches_ml,[self.kps[2],self.kps[0]],thresh)

        #中心矩阵左上角的位置(x,y)
        H1=self.H_move(H1,x,y)#right
        H2=self.H_move(H2,x,y)#left

        #第一张转换
        warp_img=cv.warpPerspective(self.imgset[1],H1,new_size)
        #imshow(warp_img,scale=0.2)
        #第二张转换
        warp_img_1=cv.warpPerspective(self.imgset[2],H2,new_size)
        #imshow(warp_img_1,scale=0.2)
        #合并
        result=warp_img_1+warp_img
        #imshow(result,scale=0.2)
        #将中间图拼上去
        H,W,_=self.imgset[0].shape
        result[y:y+H,x:x+W,:]=self.imgset[0]
        #imshow(result, scale=0.2)
        return result
