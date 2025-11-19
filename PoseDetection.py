import cv2
import mediapipe as mp

# 初始化 Pose 和 Hands 模块
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# 打开摄像头，可替换为视频或摄像机路径
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("无法读取视频帧。")
        break

    # 转换颜色格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 处理姿态和手部检测
    results_pose = pose.process(image_rgb)
    results_hands = hands.process(image_rgb)

    # 绘制可视化结果
    annotated_image = image.copy()
    if results_pose.pose_landmarks:
        (mp.solutions.drawing_utils.draw_landmarks
         (annotated_image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS))
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 显示结果
    cv2.imshow('MediaPipe Pose and Hands', annotated_image)

    # 按 'q' 键退出
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
pose.close()
hands.close()
cv2.destroyAllWindows()

