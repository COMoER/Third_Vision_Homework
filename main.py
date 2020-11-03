
from specific_points import *


def load():
    dpts=[]
    imgs=[]
    #load
    for i in range(3):
        dpts.append(np.load('./stereo-data/%d_dpt.npy'%i))
        imgs.append(cv.imread('./stereo-data/%d_orig.jpg'%i))
    return dpts,imgs
def homo():

    dpts,imgs=load()

    cc = Stitch(imgs)  #中，右，左
    cc.getkeypoints()
    #cc.drawKeyPointsAndShow()
    #透视变换法
    new_img=cc.getStitch(radio=0.4)
    imshow(new_img, scale=0.2)
    cv.imwrite('out_homo.jpg',new_img)
    cv.destroyAllWindows()
def reproject():
    #重投影法
    dpts,imgs=load()
    cam=camera()
    cam.default_arguments()
    out_img=cam.getStitch(dpts,imgs)
    cv.imwrite('out_reproject.jpg',out_img)
    cv.destroyAllWindows()
if __name__=='__main__':
    homo()
    reproject()