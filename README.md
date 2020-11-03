# Third_Vision_Homework
### 透视变换法
- 透视变换法得到的效果如图
<img src="https://github.com/COMoER/Third_Vision_Homework/blob/main/out_homo.jpg" width = "700" height = "300" alt="" align=center />
- 透视变换中采用了比较简单的暴力匹配，筛选与距离最小点比例小于radio的特征点来生成单应性矩阵，这里使用了findHomography函数，采用RANSAC算法
- 关于findHomography函数，其返回两个参数，H是3\*3的变换矩阵，status存储了一个列表，代表RANSAC算法所排除的错误匹配的特征点，其中1为匹配成功，0为匹配失败
- 透视变换中遇到的问题是投影到像素区域外，后来了解到可以通过更改变换矩阵H的值来实现投影点的平移
<img src="https://images2015.cnblogs.com/blog/893836/201603/893836-20160310181524100-995426777.png" width = "200" height = "100" alt="" align=left />
<img src="https://img-blog.csdn.net/20140521142820609" width = "200" height = "100" alt="" align=center />
