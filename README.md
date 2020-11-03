# Third_Vision_Homework
### 透视变换法
- 透视变换法得到的效果如图
<img src="https://github.com/COMoER/Third_Vision_Homework/blob/main/out_homo.jpg" width = "700" height = "300" alt="" align=center />

- 透视变换中采用了比较简单的暴力匹配，筛选与距离最小点比例小于radio的特征点来生成单应性矩阵，这里使用了findHomography函数，采用RANSAC算法
- 关于findHomography函数，其返回两个参数，H是3\*3的变换矩阵，status存储了一个列表，代表RANSAC算法所排除的错误匹配的特征点，其中1为匹配成功，0为匹配失败
- 透视变换中遇到的问题是投影到像素区域外，后来了解到可以通过更改变换矩阵H的值来实现投影点的平移，即若令H=[a1,a2,a3],则只需使a1+=x\*a3,a2+=y\*a3
<img src="https://images2015.cnblogs.com/blog/893836/201603/893836-20160310181524100-995426777.png" width = "200" height = "100" alt="" align=left />
<img src="https://img-blog.csdn.net/20140521142820609" width = "200" height = "100" alt="" align=center />

-关于边界衔接不自然问题还没有想到解决方案
### 重投影法
- 重投影法的效果（重投影产生的图像类似摩尔纹的效果，而且感觉偏暗）
<img src="https://github.com/COMoER/Third_Vision_Homework/blob/main/out_reproject.jpg" width = "700" height = "300" alt="" align=center />

- 重投影法的基本思路是将欲投影的图片的相机坐标系设为世界坐标系，然后将用特征点匹配到的对应特征点中属于欲投影图片的点的像素坐标转换为相机坐标（3D），与对应特征点中属于被投影图片点的像素坐标（2D）解pnp，得到欲投影图片相机坐标系与被投影图片相机坐标系之间的变换矩阵，这样欲投影图片的每一个点都可以被转换为在被投影图片像素坐标系上的像素坐标，这样完成了投影
#### 这里遇到的问题有
- 第一个问题是比较棘手的，就是对应特征点匹配有误的问题，似乎用solvepnpRansac消除误匹配点的效果不是很好，直接用暴力方式匹配的特征点效果也不行，于是我换了knn的匹配方式，筛选最近点与次近点比例小于radio的点。然后在询问了txy学长后，用了findFundamentalMat函数来剔除误匹配的特征点，这个函数的返回值与findHomography基本类似，运用status来筛选，这样直接用solvepnp函数计算出的变换矩阵还是蛮精确的。
- 这里注明一些小坑，在python中np.float32矩阵对应c++中用列表存储的Point3f或Point2f，对于RGB（BGR）图片来说要采用np.uint8格式，在转换矩阵为int格式时，要用astype，直接用np.int格式转换会报错
- 第二个问题是用循环来构造图片坐标和投影太慢，这里是我不熟练使用np的原因。实际上坐标一般可以用np.meshgrid来构造
- xv,yv=np.meshgrid(你的横坐标取值，你的纵坐标取值，sparse=False，indexing='xy') 然后np.stack((xv,yv),axis=2)便可得到一组坐标，这个还可以用来画三维函数的点
- 投影的使用直接用np矩阵的对应功能就可以了，类似于ROI区域直接对应的方式
- 不要for循环，不要for循环
