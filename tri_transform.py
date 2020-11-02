
from specific_points import *


def load():
    dpts=[]
    imgs=[]
    #load
    for i in range(3):
        dpts.append(np.load('./stereo-data/%d_dpt.npy'%i))
        imgs.append(cv.imread('./stereo-data/%d_orig.jpg'%i))
    return dpts,imgs
def main():

    dpts,imgs=load()
    cc = Stitch(imgs)  #中，右，左
    cc.getkeypoints()
    #cc.drawKeyPointsAndShow()
    #透视变换法
    new_img=cc.getStitch(radio=0.4)
    imshow(new_img, scale=0.2)
    cv.imwrite('out.jpg',new_img)

def main_2():
    dpts,imgs=load()
    cameraMatrix=np.array([[1114.1804893712708,0.0,1074.2415297217708],
                           [0.0,1113.4568392254073,608.6477877664104],
                           [0.0,0.0,1.0]])
    cam=camera()
    cam.default_arguments()
    out_img=cam.getStitch(dpts,imgs)
    cv.imwrite('out_2.jpg',out_img)
if __name__=='__main__':
    main_2()