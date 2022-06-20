import dlib
import cv2
import numpy as np
def load_model(file_scp):  # 加载训练好的模型
    with open(file_scp, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()  # 拆分
    face_ids = [line.split()[0] for line in lines]  # 该列表存放人脸id
    face_models = [np.load(line.split()[-1]) for line in lines]  # 该列表存放人脸模型
    return face_ids, face_models
def face_recognize(face_vec, face_models, face_ids):  # 计算欧氏距离，越相似数值越小
    scores = []
    for model in face_models:
        N = model.shape[0]
        diffMat = np.tile(face_vec, (N, 1))-model
        # 计算欧式距离
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        # 找到最小距离
        score = np.min(distances)
        scores.append(score)
    print(scores)
    index = np.argmin(scores)
    return face_ids[index], scores[index]  # 返回id和距离

face_ids, face_models = load_model(r"./model.scp")  # 加载训练好的人脸模型
detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")  # 人脸标志检测器
# 人脸特征提取器
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def test():
    img = r"C:\Users\Lei\Desktop\image\chenjieyi.jpg"
    img = cv2.imread(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换格式
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度
    print(detector)
    face_rects = detector(gray, 0)  # 检测人脸
    print(face_rects)
    for k, rect in enumerate(face_rects):  # 遍历检测的人脸
        shape = sp(img_rgb, rect)  # 标志点检测
        print(shape)
        face_vector = facerec.compute_face_descriptor(img_rgb, shape)  # 获取人脸特征
        print(face_vector)
        print("123123")
        print(face_models)
        # 计算人脸特征和人脸模型的距离
        face_id, score = face_recognize(np.array(face_vector), face_models, face_ids)  # 进行识别，返回id和距离
        if score < 0.35:  # 设定阈值来判定是否为同一人
            str_face = face_id
            str_confidence = " %.2f" % (score)
            st = "签到成功"
            print(str_face + st)
        else:
            str_face = "unknow"
            str_confidence = " %.2f" % (score)
            st = "未知"
            print(str_face + st)
if __name__ == "__main__":
    test()
