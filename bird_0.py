#作者：九五要当学霸
#链接：https://zhuanlan.zhihu.com/p/24524583
#来源：知乎
#著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
-*-coding:utf-8-*-
# """
# 基于这个 tflearn 样例代码:
# 
# https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
# 
# """

from __future__ import division, print_function,
absolute_import


# 导入 tflearn 和一些辅助文件

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data,
dropout, fully_connected

from tflearn.layers.conv import conv_2d,
max_pool_2d

from tflearn.layers.estimator import regression

from tflearn.data_preprocessing import
ImagePreprocessing

from tflearn.data_augmentation import
ImageAugmentation

import pickle



# 加载数据集



X, Y, X_test, Y_test =
pickle.load(open("full_dataset.pkl", "rb"))



# 打乱数据

X, Y = shuffle(X, Y)

# 确定数据是规范的



img_prep = ImagePreprocessing()



img_prep.add_featurewise_zero_center()



img_prep.add_featurewise_stdnorm()



# 翻转、旋转和模糊效果数据集中的图片,



# 来创造一些合成训练数据.



img_aug = ImageAugmentation()



img_aug.add_random_flip_leftright()



img_aug.add_random_rotation(max_angle=25.)



img_aug.add_random_blur(sigma_max=3.)



# 定义我们的网络架构:



# 输入内容是一张 32x32 大小, 3 个颜色通道(红、绿、蓝）的图片



network = input_data(shape=[None, 32, 32, 3],



            
        data_preprocessing=img_prep,



            
        data_augmentation=img_aug)



# 第一步: 卷积



network = conv_2d(network, 32, 3,
activation='relu')



# 第二步: 最大池化



network = max_pool_2d(network, 2)



# 第三步: 再卷积



network = conv_2d(network, 64, 3,
activation='relu')



# 第四步: 再再卷积



network = conv_2d(network, 64, 3,
activation='relu')



# 第五步: 再最大池化



network = max_pool_2d(network, 2)



# 第六步: 拥有 512 个节点的全连接神经网络



network = fully_connected(network, 512,
activation='relu')



# 第七步: Dropout - 在训练过程中随机丢掉一些数据来防止过拟合



network = dropout(network, 0.5)



# 第八步: 拥有两个输出 (0=不是鸟, 1=是鸟) 的全连接神经网络，yong l做出最终预测



network = fully_connected(network, 2,
activation='softmax')



# 告诉 tflearn 我们想如何训练神经网络



network = regression(network, optimizer='adam',



            
        loss='categorical_crossentropy',



            
        learning_rate=0.001)



# 把网络打包为一个模型对象  



model = tflearn.DNN(network,
tensorboard_verbose=0, checkpoint_path='bird-classifier.tfl.ckpt')



# 开始训练！我们将进行 100 次训练, 并实时监视它.



model.fit(X, Y, n_epoch=100, shuffle=True,
validation_set=(X_test, Y_test),



        
 show_metric=True, batch_size=96,



        
 snapshot_epoch=True,



        
 run_id='bird-classifier')



# 当训练结束时保存模型



model.save("bird-classifier.tfl")



print("Network trained and saved as
bird-classifier.tfl!")