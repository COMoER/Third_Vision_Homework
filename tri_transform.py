import numpy as np
import cv2 as cv
def resize(img,scale=0.4):
    return cv.resize(img,None,fx=scale,fy=scale)
def size(img):
    return (img.shape[1],img.shape[0])
def imshow(img,name='out',delay=0,scale=0.4):
    cv.imshow(name,resize(img,scale))
    cv.waitKey(delay)

class camera(object):
    def __init__(self,cameraMatrix):
        self.cameraMatrix=cameraMatrix

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
        H1=self.H_move(H1,x,y)
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


def main():
    cameraMatrix=np.array([[1114.1804893712708,0.0,1074.2415297217708],
                           [0.0,1113.4568392254073,608.6477877664104],
                           [0.0,0.0,1.0]])
    cam=camera(cameraMatrix)
    dpts=[]
    imgs=[]
    #load
    for i in range(3):
        dpts.append(np.load('./stereo-data/%d_dpt.npy'%i))
        imgs.append(cv.imread('./stereo-data/%d_orig.jpg'%i))

    cc = Stitch(imgs)  #中，右，左
    cc.getkeypoints()
    #cc.drawKeyPointsAndShow()
    #透视变换法
    new_img=cc.getStitch(radio=0.4)
    imshow(new_img, scale=0.2)
    cv.imwrite('out.jpg',new_img)


if __name__=='__main__':
    main()