import cv2
import mediapipe as mp
import numpy as np

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

def detect_wrist_flip(hand_landmarks):
    """
    检测翻腕动作
    通过计算手掌法向量与重力方向的夹角来判断手心朝向
    """
    if not hand_landmarks:
        return False
    
    # 获取关键点坐标
    landmarks = hand_landmarks.landmark
    
    # 获取手掌关键点：手腕、中指根部、小指根部
    wrist = np.array([landmarks[mp_hands.HandLandmark.WRIST].x,
                      landmarks[mp_hands.HandLandmark.WRIST].y,
                      landmarks[mp_hands.HandLandmark.WRIST].z])
    
    middle_mcp = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                           landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                           landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z])
    
    pinky_mcp = np.array([landmarks[mp_hands.HandLandmark.PINKY_MCP].x,
                          landmarks[mp_hands.HandLandmark.PINKY_MCP].y,
                          landmarks[mp_hands.HandLandmark.PINKY_MCP].z])
    
    # 计算手掌平面的两个向量
    vector1 = middle_mcp - wrist
    vector2 = pinky_mcp - wrist
    
    # 计算手掌法向量（手掌平面的垂直向量）
    palm_normal = np.cross(vector1, vector2)
    
    # 标准化向量
    palm_normal = palm_normal / np.linalg.norm(palm_normal)
    
    # 定义重力方向向量（简化为Z轴方向）
    gravity_vector = np.array([0, 0, 1])
    
    # 计算手掌法向量与重力方向的夹角余弦值
    dot_product = np.dot(palm_normal, gravity_vector)
    
    # 手心朝下时，夹角接近0度，cos值接近1
    # 手心朝上时，夹角接近180度，cos值接近-1
    # 当cos值小于某个阈值时，认为是翻腕状态
    if dot_product < -0.5:  # 阈值可根据实际调整
        return True
    return False

# 用于跟踪翻腕状态的变量
wrist_flipped = False

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
    
    # 重置翻腕检测标志
    flip_detected = False
    
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 检测翻腕动作
            if detect_wrist_flip(hand_landmarks):
                flip_detected = True
    
    # 输出检测结果
    if flip_detected:
        print("检测到翻腕动作！")
    else:
        print("没检测到翻腕")

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