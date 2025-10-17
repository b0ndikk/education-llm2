#!/usr/bin/env python3
"""
🎯 RESNET АНАЛИЗАТОР - ТЕКСТУРА И ЦВЕТ
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

class ResNetAnalyzer:
    def __init__(self):
        """ResNet анализатор текстур и цветов"""
        print("🚀 Инициализация ResNet...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загружаем предобученную ResNet
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        
        # Трансформы для ResNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Классы ImageNet для анализа
        self.texture_classes = {
            # Текстуры и материалы
            411: 'velvet', 412: 'wool', 413: 'silk', 414: 'cotton',
            415: 'linen', 416: 'denim', 417: 'leather', 418: 'fur',
            # Цвета и паттерны
            419: 'plaid', 420: 'striped', 421: 'polka dot', 422: 'floral'
        }
        
        print("✅ ResNet загружен!")
    
    def analyze_image(self, image: Image.Image) -> dict:
        """Анализирует текстуру и цвет изображения"""
        try:
            # Анализ через ResNet
            with torch.no_grad():
                inputs = self.transform(image).unsqueeze(0).to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs[0], dim=0)
            
            # Находим релевантные классы
            relevant_features = {}
            for class_id, class_name in self.texture_classes.items():
                if class_id < len(probabilities):
                    prob = probabilities[class_id].item()
                    if prob > 0.01:  # Порог значимости
                        relevant_features[class_name] = prob
            
            # Анализ цвета
            color_analysis = self._analyze_colors(image)
            
            # Анализ текстуры
            texture_analysis = self._analyze_texture(image)
            
            return {
                'texture_features': relevant_features,
                'color_analysis': color_analysis,
                'texture_complexity': texture_analysis,
                'deep_features': outputs[0][:100].tolist()  # Первые 100 признаков
            }
            
        except Exception as e:
            return {"error": f"ResNet ошибка: {str(e)}"}
    
    def _analyze_colors(self, image: Image.Image) -> dict:
        """Анализирует доминирующие цвета"""
        img_np = np.array(image)
        img_small = cv2.resize(img_np, (100, 100))
        pixels = img_small.reshape(-1, 3)
        
        # K-means для поиска доминирующих цветов
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels.astype(np.float32), 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Преобразуем в названия цветов
        color_names = []
        for color in centers:
            color_names.append(self._rgb_to_color_name(color))
        
        return {
            'dominant_colors': color_names[:3],  # Топ-3 цвета
            'color_variance': np.var(pixels, axis=0).tolist(),
            'brightness': np.mean(pixels) / 255.0
        }
    
    def _analyze_texture(self, image: Image.Image) -> str:
        """Анализирует сложность текстуры"""
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Вычисляем вариацию Лапласиана
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100: return "smooth"
        elif laplacian_var < 500: return "medium"
        else: return "complex"
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Преобразует RGB в название цвета"""
        color_map = {
            'black': [0, 0, 0], 'white': [255, 255, 255],
            'red': [255, 0, 0], 'blue': [0, 0, 255],
            'green': [0, 255, 0], 'yellow': [255, 255, 0],
            'purple': [128, 0, 128], 'pink': [255, 192, 203],
            'brown': [165, 42, 42], 'gray': [128, 128, 128],
            'orange': [255, 165, 0], 'beige': [245, 245, 220]
        }
        
        min_distance = float('inf')
        closest_color = 'unknown'
        
        for name, ref_rgb in color_map.items():
            distance = np.linalg.norm(rgb - ref_rgb)
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        
        return closest_color