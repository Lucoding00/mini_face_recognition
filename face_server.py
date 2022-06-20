# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains a picture of Barack Obama.
# The result is returned as json. For example:
#
# $ curl -XPOST -F "file=@obama2.jpg" http://127.0.0.1:5001
#
# Returns:
#
# {
#  "face_found_in_image": true,
#  "is_picture_of_obama": true
# }
#
# This example is based on the Flask file upload example: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# NOTE: This example requires flask to be installed! You can install it with pip:
# $ pip3 install flask

import face_recognition
from flask import Flask, jsonify, request, redirect
import dlib
import os
import cv2
import numpy as np

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


UPLOAD_FOLDER = './upload'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



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
    index = np.argmin(scores)
    return face_ids[index], scores[index]  # 返回id和距离



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

face_ids, face_models = load_model("model.scp")  # 加载训练好的人脸模型
detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")  # 人脸标志检测器
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            print("保存完成，开始识别")
            return detect_faces_in_image(file.read(),os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Is this a picture of Obama?</title>
    <h1>Upload a picture and see if it's a picture of Obama!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''


def detect_faces_in_image(file,file_path):
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换格式
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度
    face_rects = detector(gray, 0)  # 检测人脸
    str_face=""
    st=""
    for k, rect in enumerate(face_rects):  # 遍历检测的人脸
        shape = sp(img_rgb, rect)  # 标志点检测
        print(shape)
        face_vector = facerec.compute_face_descriptor(img_rgb, shape)  # 获取人脸特征

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

    result = {
        "face_found_in_image": str_face,
        "is_picture_of_obama": st
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
