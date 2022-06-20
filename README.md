# mini_facere_ognition
简单的人脸识别库，使用到了opencv、dlib等等。
## 1.环境
1. python 3.6.13
2. requirements.txt(建议使用conda)

## 2.准备数据
数据的格式是这样的
![image](https://user-images.githubusercontent.com/48003861/174622968-8d9dbd7f-aa65-4f19-9663-7c885bacd662.png)
根目录为faces是因为我在pre_build_model.py文件当中进行设置了，自己也可以进行替换
![image](https://user-images.githubusercontent.com/48003861/174623292-35ca6aa4-3644-4ab3-8bc8-033a8afd00ac.png)
之后就是人名文件夹，人名文件夹下面就是需要训练的照片数据
![image](https://user-images.githubusercontent.com/48003861/174623452-45b80697-b038-4961-8373-ca2c203befaf.png)

## 3.操作步骤
1. 安装环境
2. 训练模型，进行将照片数据集替换之后 执行python pre_build_model.py
3. 测试文件 test.py
4. 使用flask作为一个微形服务器 可以提供json格式的返回数据，便于整合其他web项目
## have fun







