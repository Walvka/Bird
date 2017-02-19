#���ߣ�����Ҫ��ѧ��
#���ӣ�https://zhuanlan.zhihu.com/p/24524583
#��Դ��֪��
#����Ȩ���������С���ҵת������ϵ���߻����Ȩ������ҵת����ע��������
-*-coding:utf-8-*-
# """
# ������� tflearn ��������:
# 
# https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
# 
# """

from __future__ import division, print_function,
absolute_import


# ���� tflearn ��һЩ�����ļ�

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



# �������ݼ�



X, Y, X_test, Y_test =
pickle.load(open("full_dataset.pkl", "rb"))



# ��������

X, Y = shuffle(X, Y)

# ȷ�������ǹ淶��



img_prep = ImagePreprocessing()



img_prep.add_featurewise_zero_center()



img_prep.add_featurewise_stdnorm()



# ��ת����ת��ģ��Ч�����ݼ��е�ͼƬ,



# ������һЩ�ϳ�ѵ������.



img_aug = ImageAugmentation()



img_aug.add_random_flip_leftright()



img_aug.add_random_rotation(max_angle=25.)



img_aug.add_random_blur(sigma_max=3.)



# �������ǵ�����ܹ�:



# ����������һ�� 32x32 ��С, 3 ����ɫͨ��(�졢�̡�������ͼƬ



network = input_data(shape=[None, 32, 32, 3],



            
        data_preprocessing=img_prep,



            
        data_augmentation=img_aug)



# ��һ��: ���



network = conv_2d(network, 32, 3,
activation='relu')



# �ڶ���: ���ػ�



network = max_pool_2d(network, 2)



# ������: �پ��



network = conv_2d(network, 64, 3,
activation='relu')



# ���Ĳ�: ���پ��



network = conv_2d(network, 64, 3,
activation='relu')



# ���岽: �����ػ�



network = max_pool_2d(network, 2)



# ������: ӵ�� 512 ���ڵ��ȫ����������



network = fully_connected(network, 512,
activation='relu')



# ���߲�: Dropout - ��ѵ���������������һЩ��������ֹ�����



network = dropout(network, 0.5)



# �ڰ˲�: ӵ��������� (0=������, 1=����) ��ȫ���������磬yong l��������Ԥ��



network = fully_connected(network, 2,
activation='softmax')



# ���� tflearn ���������ѵ��������



network = regression(network, optimizer='adam',



            
        loss='categorical_crossentropy',



            
        learning_rate=0.001)



# ��������Ϊһ��ģ�Ͷ���  



model = tflearn.DNN(network,
tensorboard_verbose=0, checkpoint_path='bird-classifier.tfl.ckpt')



# ��ʼѵ�������ǽ����� 100 ��ѵ��, ��ʵʱ������.



model.fit(X, Y, n_epoch=100, shuffle=True,
validation_set=(X_test, Y_test),



        
 show_metric=True, batch_size=96,



        
 snapshot_epoch=True,



        
 run_id='bird-classifier')



# ��ѵ������ʱ����ģ��



model.save("bird-classifier.tfl")



print("Network trained and saved as
bird-classifier.tfl!")