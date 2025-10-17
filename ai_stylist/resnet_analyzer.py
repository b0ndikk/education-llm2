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
import os

class ResNetAnalyzer:
    def __init__(self):
        """ResNet анализатор текстур и цветов"""
        print("🚀 Инициализация ResNet...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Принудительно устанавливаем кэш в домашнюю директорию
        os.makedirs(os.path.expanduser('~/.cache/torch/hub/checkpoints'), exist_ok=True)
        torch.hub.set_dir(os.path.expanduser('~/.cache/torch/hub'))
        
        # Загружаем ResNet с явным указанием весов
        self.model = models.resnet50(weights='DEFAULT')
        self.model.eval()
        self.model.to(self.device)
        
        # Трансформы
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ ResNet загружен!")
    
    def analyze_image(self, image: Image.Image) -> dict:
        """Анализирует текстуру и цвет изображения"""
        try:
            # ResNet анализ
            with torch.no_grad():
                inputs = self.transform(image).unsqueeze(0).to(self.device)
                outputs = self.model(inputs)
                features = outputs[0].cpu().numpy()
            
            # Анализ цвета
            color_analysis = self._analyze_colors(image)
            
            # Анализ текстуры
            texture_analysis = self._analyze_texture(image)
            
            # Извлекаем текстовые признаки
            texture_features = self._extract_texture_features(features)
            
            return {
                'resnet_features': features[:512].tolist(),  # Полные фичи
                'texture_features': texture_features,
                'color_analysis': color_analysis,
                'texture_complexity': texture_analysis,
                'feature_dim': len(features)
            }
            
        except Exception as e:
            return {"error": f"ResNet ошибка: {str(e)}"}
    
    def _extract_texture_features(self, features):
        """Анализирует текстуры на основе ResNet фич"""
        # Анализируем статистики фич для определения текстуры
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        feature_range = np.max(features) - np.min(features)
        
        texture_type = "complex" if feature_std > 1.0 else "smooth"
        
        """ДОБАВИТЬ АНАЛИЗ МАТЕРИАЛОВ ОБУВИ"""
        texture_classes = {
        # МАТЕРИАЛЫ ОБУВИ
        411: 'velvet', 412: 'wool', 413: 'silk', 414: 'cotton',
        416: 'denim', 417: 'leather', 418: 'fur',
        419: 'plaid', 420: 'striped', 421: 'polka dot', 422: 'floral',
        # ДОБАВИТЬ ДЛЯ ОБУВИ:
        423: 'rubber', 424: 'suede', 425: 'canvas', 426: 'mesh'
    }

        return {
            'feature_mean': float(feature_mean),
            'feature_std': float(feature_std),
            'feature_range': float(feature_range),
            'texture_type': texture_type
        }
        
    
    def _analyze_colors(self, image: Image.Image) -> dict:
        """Анализирует доминирующие цвета с K-means"""
        img_np = np.array(image)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
        # K-means для точного анализа цветов
        img_small = cv2.resize(img_np, (100, 100))
        pixels = img_small.reshape(-1, 3).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Находим доминирующие цвета
        unique, counts = np.unique(labels, return_counts=True)
        dominant_colors = []
        
        for i in range(min(3, len(centers))):
            color = centers[unique[np.argsort(-counts)[i]]]
            color_name = self._rgb_to_color_name(color)
            dominant_colors.append({
                'color': color_name,
                'rgb': color.tolist(),
                'percentage': float(counts[unique[i]] / len(labels))
            })
        
        return {
            'dominant_colors': dominant_colors,
            'color_variance': float(np.var(pixels)),
            'is_colorful': np.var(pixels) > 1000
        }
    
    def _analyze_texture(self, image: Image.Image) -> str:
        """Сложный анализ текстуры"""
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Несколько метрик текстуры
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # GLCM-like features (упрощенно)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        texture_energy = np.mean(gradient_magnitude)
        
        if laplacian_var < 100 and texture_energy < 10:
            return "smooth"
        elif laplacian_var < 500 and texture_energy < 50:
            return "medium"
        else:
            return "complex"
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Точное преобразование RGB в название цвета"""
        color_map = {
            'black': [0, 0, 0], 
            'white': [255, 255, 255],
            'red': [255, 0, 0], 
            'blue': [0, 0, 255],
            'green': [0, 128, 0], 
            'yellow': [255, 255, 0],
            'purple': [128, 0, 128], 
            'pink': [255, 192, 203],
            'brown': [139, 69, 19], 
            'gray': [128, 128, 128],
            'orange': [255, 165, 0], 
            'beige': [245, 245, 220],
            'navy': [0, 0, 128],
            'teal': [0, 128, 128],
            'olive': [128, 128, 0],
            'maroon': [128, 0, 0],
            'cyan': [0, 255, 255],
            'magenta': [255, 0, 255]
        }
        
        min_distance = float('inf')
        closest_color = 'unknown'
        
        for name, ref_rgb in color_map.items():
            distance = np.sqrt(np.sum((rgb - ref_rgb) ** 2))
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        
        return closest_color