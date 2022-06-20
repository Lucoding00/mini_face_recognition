import dlib
import os
import cv2
import numpy as np
print("开始结束录入")
# 加载人脸特征提取器
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# 加载人脸标志点检测器
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")
# 记录所有模型信息
with open('model.scp', 'w', encoding='utf-8') as f:  # 记录人脸id与人脸模型
    base_path = 'faces'  # 遍历faces文件夹
    for face_id in os.listdir(base_path):
        face_dir = os.path.join(base_path, face_id)
        if os.path.isdir(face_dir):
            file_face_model = os.path.join(face_dir, face_id + '.npy')  # 遍历base_path/face_id文件夹
            face_vectors = []  # 人脸特征list

        for face_img in os.listdir(face_dir):  # 遍历，查找所有文件
            if os.path.splitext(face_img)[-1] == '.jpg' or os.path.splitext(face_img)[-1] == '.JPG' or os.path.splitext(face_img)[-1] == '.png':  # 寻找所有.jpg文件
                # img = cv2.imread(os.path.join(face_dir, face_img))  # 读取图片并转换格式
                img = cv2.imdecode(np.fromfile(os.path.join(face_dir, face_img), dtype=np.uint8), -1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.array(img)  # 存放的是已经截取好的人脸图片，所以在图内检测标志点
                h, w, _ = np.shape(img)
                rect = dlib.rectangle(0, 0, w, h)  # 整个区域
                shape = sp(img, rect)  # 辅助人脸定位，获取关键位
                print("Generate face vector of", face_img)
                face_vector = facerec.compute_face_descriptor(img, shape)  # 获取128维人脸特征
                face_vectors.append(face_vector)  # 保存图像和人脸id
        if len(face_vectors) > 0:  # 人脸模型保存，并写入model.scp文件
            np.save(file_face_model, face_vectors)
            # f.write('%s %s\n' % (face_id, file_face_model))
            f.write('%s %s\n' % (face_id, file_face_model))
