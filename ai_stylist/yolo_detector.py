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
        
        # Используем YOLOv8 с лучшей детекцией
        self.model = YOLO('yolov8m.pt')  # Средняя модель для баланса скорости/точности
        
        print("✅ YOLO загружен!")
    
    def detect_clothing(self, image: Image.Image) -> dict:
        """Обнаруживает одежду на изображении"""
        try:
            img_np = np.array(image)
            results = self.model(img_np, conf=0.3)  # Более низкий порог для большего охвата
            
            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    
                    class_name = self.model.names[class_id]
                    
                    # Расширенный фильтр для одежды и связанных объектов
                    if self._is_clothing_related(class_name, confidence):
                        # Вычисляем дополнительные метрики
                        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        image_area = img_np.shape[1] * img_np.shape[0]
                        area_ratio = bbox_area / image_area
                        
                        detections.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': bbox,
                            'area_ratio': area_ratio,
                            'is_large': area_ratio > 0.3,
                            'center_x': (bbox[0] + bbox[2]) / 2 / img_np.shape[1],
                            'center_y': (bbox[1] + bbox[3]) / 2 / img_np.shape[0]
                        })
            
            # Анализ композиции
            composition = self._analyze_composition(detections, img_np.shape)
            
            return {
                'detections': detections,
                'total_items': len(detections),
                'composition': composition,
                'has_clothing': len([d for d in detections if self._is_direct_clothing(d['class_name'])]) > 0
            }
            
        except Exception as e:
            return {"error": f"YOLO ошибка: {str(e)}"}
    
    def _is_clothing_related(self, class_name: str, confidence: float) -> bool:
        """РАСШИРЕННЫЙ ФИЛЬТР ДЛЯ ВСЕЙ ОДЕЖДЫ"""
        clothing_categories = [
        # ОСНОВНАЯ ОДЕЖДА
        'person', 'tie', 
        
        # ОБУВЬ (если есть в YOLO классах)
        'shoe', 'sneaker', 'boot', 
        
        # СУМКИ И РЮКЗАКИ
        'handbag', 'backpack', 'suitcase', 'purse',
        
        # СПОРТИВНЫЙ ИНВЕНТАРЬ (может содержать одежду)
        'sports ball', 'baseball bat', 'baseball glove', 
        'tennis racket', 'skateboard', 'surfboard',
        
        # АКСЕССУАРЫ
        'umbrella', 'hat', 'cap'
    ]
    
    # Для person - более низкий порог, так как на человеке может быть одежда
        if class_name == 'person':
            return confidence > 0.3
        else:
            return class_name in clothing_categories and confidence > 0.4
    
    def _is_direct_clothing(self, class_name: str) -> bool:
        """Определяет, является ли объект непосредственно одеждой"""
        direct_clothing = ['tie', 'handbag', 'backpack']
        return class_name in direct_clothing
    
    def _analyze_composition(self, detections: list, image_shape: tuple) -> dict:
        """Анализирует композицию изображения"""
        if not detections:
            return {'dominant_region': 'center', 'layout': 'empty'}
        
        # Анализ распределения объектов
        centers_x = [d['center_x'] for d in detections]
        centers_y = [d['center_y'] for d in detections]
        
        avg_x = np.mean(centers_x)
        avg_y = np.mean(centers_y)
        
        # Определяем доминирующую область
        if avg_x < 0.33: 
            horizontal_pos = 'left'
        elif avg_x > 0.66: 
            horizontal_pos = 'right'
        else: 
            horizontal_pos = 'center'
            
        if avg_y < 0.33: 
            vertical_pos = 'top'
        elif avg_y > 0.66: 
            vertical_pos = 'bottom'
        else: 
            vertical_pos = 'middle'
        
        # Анализ размера объектов
        large_objects = len([d for d in detections if d['is_large']])
        layout = 'focused' if large_objects > 0 else 'scattered'
        
        return {
            'dominant_region': f"{horizontal_pos}-{vertical_pos}",
            'layout': layout,
            'object_density': len(detections) / (image_shape[0] * image_shape[1] / 10000),
            'has_large_objects': large_objects > 0
        }