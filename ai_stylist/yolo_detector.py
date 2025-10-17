#!/usr/bin/env python3
"""
🎯 YOLO ДЕТЕКТОР - ОБНАРУЖЕНИЕ ОДЕЖДЫ
"""

from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

class YOLODetector:
    def __init__(self):
        """YOLO детектор одежды"""
        print("🚀 Инициализация YOLO...")
        
        # Загружаем YOLO (можно заменить на специализированную модель для одежды)
        self.model = YOLO('yolov8n.pt')
        
        # Классы одежды в YOLO
        self.clothing_classes = {
            0: 'person',        # Может содержать одежду
            26: 'tie', 27: 'suitcase', 28: 'frisbee', 
            29: 'skis', 30: 'snowboard', 31: 'sports ball',
            32: 'kite', 33: 'baseball bat', 34: 'baseball glove',
            35: 'skateboard', 36: 'surfboard', 37: 'tennis racket',
            39: 'bottle', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
            48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
            52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
            68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
            76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
        
        print("✅ YOLO загружен!")
    
    def detect_clothing(self, image: Image.Image) -> dict:
        """Обнаруживает одежду на изображении"""
        try:
            # Конвертируем PIL в numpy для YOLO
            img_np = np.array(image)
            
            # Детекция
            results = self.model(img_np)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    
                    # Фильтруем по уверенности и классам
                    if confidence > 0.5:
                        detections.append({
                            'class_id': class_id,
                            'class_name': self.clothing_classes.get(class_id, 'unknown'),
                            'confidence': confidence,
                            'bbox': bbox,
                            'center_x': (bbox[0] + bbox[2]) / 2,
                            'center_y': (bbox[1] + bbox[3]) / 2
                        })
            
            return {
                'detections': detections,
                'total_items': len(detections),
                'dominant_region': self._find_dominant_region(detections)
            }
            
        except Exception as e:
            return {"error": f"YOLO ошибка: {str(e)}"}
    
    def _find_dominant_region(self, detections):
        """Находит доминирующую область на изображении"""
        if not detections:
            return "center"
        
        # Анализируем распределение объектов
        centers_x = [d['center_x'] for d in detections]
        centers_y = [d['center_y'] for d in detections]
        
        avg_x = np.mean(centers_x)
        avg_y = np.mean(centers_y)
        
        if avg_x < 0.33: return "left"
        elif avg_x > 0.66: return "right"
        else: return "center"