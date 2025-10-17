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
        
        try:
            # Загружаем ResNet с явным указанием весов
            self.model = models.resnet50(weights='DEFAULT')
            self.model.eval()
            self.model.to(self.device)
            print("✅ ResNet загружен!")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки ResNet: {e}")
            self.model = None
        
        # Трансформы
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def analyze_image(self, image: Image.Image) -> dict:
        """Анализирует текстуру и цвет изображения"""
        try:
            # Анализ цвета и текстуры (всегда работает)
            color_analysis = self._analyze_colors(image)
            texture_analysis = self._analyze_texture(image)
            
            result = {
                'color_analysis': color_analysis,
                'texture_complexity': texture_analysis,
            }
            
            # ResNet анализ (если модель загружена)
            if self.model is not None:
                with torch.no_grad():
                    inputs = self.transform(image).unsqueeze(0).to(self.device)
                    outputs = self.model(inputs)
                    features = outputs[0].cpu().numpy()
                
                # Извлекаем текстовые признаки
                texture_features = self._extract_texture_features(features)
                result.update({
                    'resnet_features': features[:512].tolist(),
                    'texture_features': texture_features,
                    'feature_dim': len(features)
                })
            else:
                result['resnet_error'] = "Модель не загружена"
            
            return result
            
        except Exception as e:
            return {"error": f"ResNet ошибка: {str(e)}"}
    
    def _extract_texture_features(self, features):
        """Анализирует текстуры на основе ResNet фич"""
        try:
            # Анализируем статистики фич для определения текстуры
            feature_mean = np.mean(features)
            feature_std = np.std(features)
            feature_range = np.max(features) - np.min(features)
            
            # Определяем тип текстуры
            if feature_std < 0.5:
                texture_type = "smooth"
            elif feature_std < 1.5:
                texture_type = "medium"
            else:
                texture_type = "complex"
            
            # Анализ материалов на основе фич
            material_analysis = self._analyze_materials(features)
            
            return {
                'feature_mean': float(feature_mean),
                'feature_std': float(feature_std),
                'feature_range': float(feature_range),
                'texture_type': texture_type,
                'material_analysis': material_analysis
            }
        except Exception as e:
            return {"error": f"Анализ текстур: {str(e)}"}
    
    def _analyze_materials(self, features):
        """Анализирует вероятные материалы на основе ResNet фич"""
        try:
            # Анализ фич для определения материалов
            feature_energy = np.sum(np.abs(features))
            feature_variance = np.var(features)
            
            materials = {}
            
            # Эвристический анализ материалов
            if feature_variance > 2.0:
                materials['complex_material'] = 0.7
            if feature_energy > 50.0:
                materials['textured_material'] = 0.6
            if np.mean(features[:100]) > 0.1:
                materials['smooth_material'] = 0.5
            
            # Специфичные для обуви материалы
            if feature_std > 1.0:
                materials['rubber'] = 0.4
                materials['leather'] = 0.6
            
            return materials
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_colors(self, image: Image.Image) -> dict:
        """Анализирует доминирующие цвета с K-means"""
        try:
            img_np = np.array(image)
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
            # K-means для точного анализа цветов
            img_small = cv2.resize(img_np, (100, 100))
            pixels = img_small.reshape(-1, 3).astype(np.float32)
            
            # Убедимся, что у нас достаточно пикселей
            if len(pixels) < 5:
                return {
                    'dominant_colors': [{'color': 'unknown', 'rgb': [0, 0, 0], 'percentage': 1.0}],
                    'color_variance': 0.0,
                    'is_colorful': False
                }
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixels, min(5, len(pixels)), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Находим доминирующие цвета
            unique, counts = np.unique(labels, return_counts=True)
            dominant_colors = []
            
            for i in range(min(3, len(centers))):
                if i < len(unique):
                    color = centers[unique[i]]
                    color_name = self._rgb_to_color_name(color)
                    percentage = float(counts[i] / len(labels))
                    
                    dominant_colors.append({
                        'color': color_name,
                        'rgb': color.tolist(),
                        'percentage': percentage
                    })
            
            return {
                'dominant_colors': dominant_colors,
                'color_variance': float(np.var(pixels)),
                'is_colorful': np.var(pixels) > 1000,
                'total_colors': len(dominant_colors)
            }
            
        except Exception as e:
            return {
                'dominant_colors': [{'color': 'error', 'rgb': [0, 0, 0], 'percentage': 1.0}],
                'color_variance': 0.0,
                'is_colorful': False,
                'error': str(e)
            }
    
    def _analyze_texture(self, image: Image.Image) -> str:
        """Сложный анализ текстуры"""
        try:
            img_np = np.array(image)
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            
            # Убедимся, что изображение не пустое
            if gray.size == 0:
                return "unknown"
            
            # Несколько метрик текстуры
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if np.isnan(laplacian_var):
                laplacian_var = 0
            
            # GLCM-like features (упрощенно)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            if gradient_magnitude.size > 0:
                texture_energy = np.mean(gradient_magnitude)
            else:
                texture_energy = 0
            
            # Определяем сложность текстуры
            if laplacian_var < 50 or texture_energy < 5:
                return "smooth"
            elif laplacian_var < 200 or texture_energy < 20:
                return "medium"
            else:
                return "complex"
                
        except Exception as e:
            return f"error: {str(e)}"
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Точное преобразование RGB в название цвета"""
        try:
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
                'magenta': [255, 0, 255],
                'silver': [192, 192, 192],
                'gold': [255, 215, 0]
            }
            
            min_distance = float('inf')
            closest_color = 'unknown'
            
            for name, ref_rgb in color_map.items():
                distance = np.sqrt(np.sum((rgb - ref_rgb) ** 2))
                if distance < min_distance:
                    min_distance = distance
                    closest_color = name
            
            return closest_color
            
        except Exception:
            return 'unknown'