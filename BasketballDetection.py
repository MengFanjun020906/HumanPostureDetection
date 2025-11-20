import cv2
from ultralytics import YOLO
import numpy as np

# 加载预训练的YOLOv8模型
model = YOLO('yolov8n.pt')  # 或使用专门训练的篮球检测模型

# 视频文件路径
video_path = 'test_video.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频属性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # 默认30fps

# 创建VideoWriter对象用于保存视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码
out = cv2.VideoWriter('basketball_detected_video.mp4', fourcc, fps, (frame_width, frame_height))

# 创建跟踪器（用于多目标跟踪）
trackers = []
tracker_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
tracked_bboxes = []  # 存储跟踪的边界框

# 处理每一帧
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 每5帧重新检测一次（更频繁的检测），或当没有跟踪器时
    if frame_count % 5 == 0 or len(trackers) == 0:
        # 使用YOLO检测篮球，提高置信度阈值以提高精度
        results = model(frame, classes=[32], conf=0.6)  # 提高置信度阈值到0.6
        
        # 清除旧的跟踪器
        trackers = []
        tracked_bboxes = []
        
        # 为每个检测到的篮球创建跟踪器
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if box.conf[0] > 0.6:  # 提高置信度阈值
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 确保边界框在图像范围内
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_width, x2)
                    y2 = min(frame_height, y2)
                    
                    # 只有当边界框有效时才创建跟踪器
                    if x2 > x1 and y2 > y1:
                        tracker = cv2.TrackerCSRT_create()  # 使用CSRT跟踪器，精度更高
                        tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                        trackers.append(tracker)
                        tracked_bboxes.append((x1, y1, x2-x1, y2-y1))
    
    # 更新所有跟踪器（每一帧都更新）
    updated_bboxes = []
    failed_trackers = []
    
    for i, (tracker, bbox) in enumerate(zip(trackers, tracked_bboxes)):
        success, new_bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, new_bbox)
            # 检查边界框是否有效
            if x >= 0 and y >= 0 and x+w <= frame_width and y+h <= frame_height and w > 0 and h > 0:
                updated_bboxes.append((x, y, w, h))
                # 绘制跟踪框
                cv2.rectangle(frame, (x, y), (x+w, y+h), tracker_colors[i % len(tracker_colors)], 2)
                cv2.putText(frame, f'Basketball {i+1}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracker_colors[i % len(tracker_colors)], 2)
            else:
                failed_trackers.append(i)
        else:
            failed_trackers.append(i)
    
    # 移除失败的跟踪器
    for i in sorted(failed_trackers, reverse=True):
        if i < len(trackers):
            trackers.pop(i)
            tracked_bboxes.pop(i)
    
    # 更新跟踪的边界框
    tracked_bboxes = updated_bboxes
    
    # 如果跟踪器数量较少，增加重新检测的频率
    if len(trackers) < 2 and frame_count % 3 == 0:
        # 使用YOLO检测篮球
        results = model(frame, classes=[32], conf=0.5)  # 稍微降低置信度阈值
        
        # 为新检测到的篮球创建跟踪器
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if box.conf[0] > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 检查是否与现有跟踪器重叠，避免重复跟踪
                    new_bbox = (x1, y1, x2-x1, y2-y1)
                    overlap = False
                    for existing_bbox in tracked_bboxes:
                        # 计算IoU（交并比）
                        xA = max(x1, existing_bbox[0])
                        yA = max(y1, existing_bbox[1])
                        xB = min(x2, existing_bbox[0] + existing_bbox[2])
                        yB = min(y2, existing_bbox[1] + existing_bbox[3])
                        
                        if xB > xA and yB > yA:
                            interArea = (xB - xA) * (yB - yA)
                            boxAArea = (x2 - x1) * (y2 - y1)
                            boxBArea = existing_bbox[2] * existing_bbox[3]
                            iou = interArea / float(boxAArea + boxBArea - interArea)
                            if iou > 0.3:  # 如果IoU大于0.3，认为是同一个对象
                                overlap = True
                                break
                    
                    # 如果没有重叠，创建新的跟踪器
                    if not overlap:
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame_width, x2)
                        y2 = min(frame_height, y2)
                        
                        if x2 > x1 and y2 > y1:
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                            trackers.append(tracker)
                            tracked_bboxes.append((x1, y1, x2-x1, y2-y1))
    
    # 将处理后的帧写入输出视频文件
    out.write(frame)
    
    # 显示结果
    cv2.imshow('Basketball Tracking', frame)
    
    # 按q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()  # 释放视频写入器
cv2.destroyAllWindows()

print("检测后的视频已保存为 basketball_detected_video.mp4")