#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import time
from model import mini_Xception
from cvinference import cv_inference, emotions, emotion_to_color
import torch


"""
demo를 위한 변수 초기화
"""
class_num = 4    # 0: Normal, 1: Drowsy, 2: Call, 3: Tobacco, 4: Out of Camera
detect_sec = 10  # 몇 초 마다 저장할지?
detect_sec_list = []  
detect_list = []  
detect_count = [0] * (class_num + 1)  # class 개수 + 1 (error: frame 밖으로 나갔을때 or face detection 실패)
start_time = time.time()


def detection_count(result, emotion, time, face_detect=True):
    """
    frame classifier 정보들을 처리
    설정한 시간(10초)마다 결과값을 저장한다.
    """
    global detect_sec, detect_count

    if face_detect:
        detect_count[result] += 1

    # 매 10초(2~300 frame) 마다 한번, 가장 많이 classifier된 상태를 결과값으로 저장한다.
    if (
        int(time) != 0
        and int(time) % detect_sec == 0
        and int(time) not in detect_sec_list
    ):
        max_result = detect_count.index(max(detect_count))
        detect_list.append(emotions[max_result])

        # 중복 저장 회피
        detect_sec_list.append(int(time))

        # initialize
        print(
            f"Noraml: {detect_count[0]}, Drowsy: {detect_count[1]}",
            f"Call: {detect_count[2]}, Tobacco: {detect_count[3]}, Out of Camera: {detect_count[4]}"
        )
        detect_count = [0] * (class_num + 1)
        if len(detect_sec_list) > 5:
            detect_sec_list.pop(0)

        return int(time), detect_list
    return 0, None


def print_classifier_info():
    """
    오른쪽 상단에 보여줄 classifier 정보
    -> 가장 최근 정보
    """
    global detect_list
    
    if len(detect_list) == 0:
        return "Classifier Start..."
    
    result = detect_list[-1]
    if result in ["Call", "Tobacco"]:
        result = "Careless"
    return result
    
"""
Start Cam
"""
detector_path = "./detection_model/haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(detector_path)
checkpoint_path = "./classification_model/mini_xception_log_epoch_best1.pth"
checkpoint = torch.load(checkpoint_path)
classifier = mini_Xception(n_class=4)
classifier.load_state_dict(checkpoint["model_state_dict"])
classifier.to("cpu")
classifier.eval()

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 480)

if not cap.isOpened():
    print("Camera doesn't work")
    exit()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detector.detectMultiScale(gray_img, 1.3, 5)
    
    # classifier info
    w, h, c = color_img.shape
    cv2.putText(
        color_img,
        f"{print_classifier_info()}",
        color=[102,102,255],
        org=(20, 50),
        fontFace=2,
        fontScale=2,
        thickness=2,
    )

    # face detection 없음
    if len(faces) == 0:
        output_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("detection", output_img)
        cur_time, result_list = detection_count(
            class_num, "Out of Camera", (time.time() - start_time)
        )
        if result_list:
            pass
#             print(
#                 f"[{cur_time} sec]",
#                 f"Normal: {result_list.count('Normal')}", 
#                 f"Drowsy: {result_list.count('Drowsy')}", 
#                 f"Careless: {result_list.count('Careless')}",
#                 f"Out of Camera: {result_list.count('Out of Camera')}",
#             )

    # face detection 성공
    for face_coords in faces:
        x, y, w, h = face_coords
        x_min = x - 30
        x_max = x + w + 30
        y_max = y - 30
        y_min = y + h + 30

        input_img = gray_img[y_max:y_min, x_min:x_max]
        
        try:
            result, emotion = cv_inference(classifier, input_img)
            color = emotion_to_color[emotion]
            cv2.rectangle(color_img, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(
                color_img,
                f"{emotion}",
                color=color,
                org=(x_min, y_max - 10),
                fontFace=0,
                fontScale=1,
                thickness=2,
            )
            output_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("detection", output_img)
            
        # cam frame 에 걸쳐 있는 경우
        except Exception as e:
            result, emotion = class_num, "Out of Camera"
            color = [0, 0, 0]
            cv2.rectangle(color_img, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(
                color_img,
                f"{emotion}",
                color=color,
                org=(x_min, y_max - 10),
                fontFace=0,
                fontScale=1,
                thickness=2,
            )
            output_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("detection", output_img)
        finally:
            cur_time, result_list = detection_count(result, emotion, (time.time() - start_time))
            if result_list:
                pass
#                 print(
#                     f"[{cur_time} sec]",
#                     f"Normal: {result_list.count('Normal')}", 
#                     f"Drowsy: {result_list.count('Drowsy')}", 
#                     f"Careless: {result_list.count('Careless')}",
#                     f"Out of Camera: {result_list.count('Out of Camera')}",
#                 )

    if cv2.waitKey(1) == ord("q"):
        cap.release()
        cv2.destroyWindow("detection")
        break


# In[ ]:




