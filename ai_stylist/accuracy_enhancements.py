import torch
import torch.nn as nn
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import cv2
from PIL import Image
import random
from typing import Dict, List, Tuple, Any
import warnings

class AccuracyEnhancer:
    """Класс для улучшения точности ансамбля моделей"""
    
    def __init__(self):
        self.calibrated_models = {}
        self.ensemble_weights = {}
        self.augmentation_pipeline = None
        self.hierarchical_classifier = None
        self._setup_enhancements()
    
    def _setup_enhancements(self):
        """Инициализация всех улучшений"""
        print("🔧 Настройка улучшений точности...")
        
        # 1. Настройка весов ансамбля
        self._setup_ensemble_weights()
        
        # 2. Настройка калибровки уверенности
        self._setup_confidence_calibration()
        
        # 3. Настройка аугментации данных
        self._setup_data_augmentation()
        
        # 4. Настройка многоуровневой классификации
        self._setup_hierarchical_classification()
        
        print("✅ Улучшения точности настроены!")
    
    def _setup_ensemble_weights(self):
        """Настройка весов ансамбля"""
        self.ensemble_weights = {
            'fashion_clip': 0.4,      # Стиль и мода
            'deepfashion_cnn': 0.3,   # Классификация одежды
            'yolo': 0.2,              # Детекция объектов
            'vit': 0.1                # Совместимость
        }
        
        # Адаптивные веса на основе производительности
        self.adaptive_weights = {
            'performance_history': {},
            'confidence_threshold': 0.7,
            'weight_adjustment_rate': 0.1
        }
    
    def _setup_confidence_calibration(self):
        """Настройка калибровки уверенности"""
        self.calibration_methods = {
            'platt_scaling': True,
            'isotonic_regression': True,
            'temperature_scaling': True
        }
        
        # Калибровочные параметры
        self.calibration_params = {
            'temperature': 1.0,
            'bias': 0.0,
            'scale': 1.0
        }
    
    def _setup_data_augmentation(self):
        """Настройка аугментации данных"""
        self.augmentation_transforms = {
            'rotation': {'angle_range': (-15, 15)},
            'brightness': {'factor_range': (0.8, 1.2)},
            'contrast': {'factor_range': (0.8, 1.2)},
            'saturation': {'factor_range': (0.8, 1.2)},
            'hue': {'shift_range': (-0.1, 0.1)},
            'flip': {'horizontal': True, 'vertical': False},
            'crop': {'scale_range': (0.8, 1.0)},
            'noise': {'std_range': (0, 0.1)}
        }
    
    def _setup_hierarchical_classification(self):
        """Настройка многоуровневой классификации"""
        self.hierarchy_levels = {
            'level_1': 'clothing_type',      # Одежда / Обувь / Аксессуары
            'level_2': 'garment_category',   # Верх / Низ / Обувь
            'level_3': 'specific_item',      # Футболка / Джинсы / Кроссовки
            'level_4': 'style_details'       # Стиль / Цвет / Материал
        }
        
        # Иерархические правила
        self.hierarchical_rules = {
            'clothing_type': ['одежда', 'обувь', 'аксессуары'],
            'garment_category': ['верх', 'низ', 'обувь', 'верхняя_одежда'],
            'specific_item': ['футболка', 'джинсы', 'кроссовки', 'куртка'],
            'style_details': ['стиль', 'цвет', 'материал', 'сезон']
        }
    
    def apply_ensemble_weights(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """Применяет веса ансамбля к предсказаниям"""
        weighted_predictions = {}
        
        for model_name, prediction in predictions.items():
            if model_name in self.ensemble_weights:
                weight = self.ensemble_weights[model_name]
                
                # Применяем вес к уверенности
                if 'confidence' in prediction:
                    weighted_confidence = prediction['confidence'] * weight
                    weighted_predictions[model_name] = {
                        **prediction,
                        'weighted_confidence': weighted_confidence,
                        'weight': weight
                    }
                else:
                    weighted_predictions[model_name] = prediction
        
        return weighted_predictions
    
    def calibrate_confidence(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Калибрует уверенность предсказаний"""
        calibrated_predictions = {}
        
        for model_name, prediction in predictions.items():
            if 'confidence' in prediction:
                # Применяем температурное масштабирование
                temperature = self.calibration_params['temperature']
                calibrated_conf = self._temperature_scaling(
                    prediction['confidence'], temperature
                )
                
                # Применяем Platt scaling
                calibrated_conf = self._platt_scaling(calibrated_conf)
                
                calibrated_predictions[model_name] = {
                    **prediction,
                    'calibrated_confidence': calibrated_conf,
                    'original_confidence': prediction['confidence']
                }
            else:
                calibrated_predictions[model_name] = prediction
        
        return calibrated_predictions
    
    def _temperature_scaling(self, confidence: float, temperature: float) -> float:
        """Температурное масштабирование уверенности"""
        return torch.softmax(torch.tensor([confidence, 1-confidence]) / temperature, dim=0)[0].item()
    
    def _platt_scaling(self, confidence: float) -> float:
        """Platt scaling для калибровки"""
        # Упрощенная версия Platt scaling
        return 1 / (1 + np.exp(-(confidence - 0.5) * 4))
    
    def augment_image(self, image: Image.Image, num_augmentations: int = 3) -> List[Image.Image]:
        """Аугментирует изображение для улучшения обучения"""
        augmented_images = [image]  # Оригинальное изображение
        
        for _ in range(num_augmentations):
            aug_image = image.copy()
            
            # Случайные трансформации
            if random.random() < 0.7:  # 70% вероятность
                aug_image = self._apply_rotation(aug_image)
            
            if random.random() < 0.5:  # 50% вероятность
                aug_image = self._apply_brightness_contrast(aug_image)
            
            if random.random() < 0.3:  # 30% вероятность
                aug_image = self._apply_noise(aug_image)
            
            if random.random() < 0.4:  # 40% вероятность
                aug_image = self._apply_crop(aug_image)
            
            augmented_images.append(aug_image)
        
        return augmented_images
    
    def _apply_rotation(self, image: Image.Image) -> Image.Image:
        """Применяет поворот к изображению"""
        angle = random.uniform(*self.augmentation_transforms['rotation']['angle_range'])
        return image.rotate(angle, fillcolor=(255, 255, 255))
    
    def _apply_brightness_contrast(self, image: Image.Image) -> Image.Image:
        """Применяет изменение яркости и контраста"""
        brightness = random.uniform(*self.augmentation_transforms['brightness']['factor_range'])
        contrast = random.uniform(*self.augmentation_transforms['contrast']['factor_range'])
        
        # Конвертируем в numpy для обработки
        img_array = np.array(image)
        img_array = np.clip(img_array * brightness, 0, 255)
        img_array = np.clip((img_array - 128) * contrast + 128, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _apply_noise(self, image: Image.Image) -> Image.Image:
        """Применяет шум к изображению"""
        std = random.uniform(*self.augmentation_transforms['noise']['std_range'])
        img_array = np.array(image)
        noise = np.random.normal(0, std * 255, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _apply_crop(self, image: Image.Image) -> Image.Image:
        """Применяет случайное обрезание"""
        scale = random.uniform(*self.augmentation_transforms['crop']['scale_range'])
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Центрированное обрезание
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        return image.crop((left, top, right, bottom)).resize((width, height))
    
    def hierarchical_classify(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Применяет многоуровневую классификацию"""
        hierarchical_results = {}
        
        for model_name, prediction in predictions.items():
            if 'category' in prediction:
                category = prediction['category']
                
                # Определяем уровень в иерархии
                hierarchy_level = self._determine_hierarchy_level(category)
                
                # Применяем иерархические правила
                refined_prediction = self._apply_hierarchical_rules(
                    category, hierarchy_level
                )
                
                hierarchical_results[model_name] = {
                    **prediction,
                    'hierarchy_level': hierarchy_level,
                    'refined_prediction': refined_prediction
                }
            else:
                hierarchical_results[model_name] = prediction
        
        return hierarchical_results
    
    def _determine_hierarchy_level(self, category: str) -> str:
        """Определяет уровень иерархии для категории"""
        category_lower = category.lower()
        
        # Уровень 1: Тип одежды
        if any(item in category_lower for item in ['футболка', 'рубашка', 'платье', 'джинсы']):
            return 'clothing_type'
        
        # Уровень 2: Категория одежды
        elif any(item in category_lower for item in ['верх', 'низ', 'обувь']):
            return 'garment_category'
        
        # Уровень 3: Конкретный предмет
        elif any(item in category_lower for item in ['футболка', 'джинсы', 'кроссовки']):
            return 'specific_item'
        
        # Уровень 4: Детали стиля
        else:
            return 'style_details'
    
    def _apply_hierarchical_rules(self, category: str, level: str) -> Dict[str, Any]:
        """Применяет иерархические правила"""
        rules = self.hierarchical_rules.get(level, [])
        
        # Находим наиболее подходящее правило
        best_match = None
        best_score = 0
        
        for rule in rules:
            if rule in category.lower():
                score = len(rule) / len(category)
                if score > best_score:
                    best_score = score
                    best_match = rule
        
        return {
            'original_category': category,
            'hierarchical_category': best_match or category,
            'confidence': best_score,
            'level': level
        }
    
    def enhance_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Применяет все улучшения к предсказаниям"""
        enhanced_predictions = predictions.copy()
        
        # 1. Применяем веса ансамбля
        enhanced_predictions = self.apply_ensemble_weights(enhanced_predictions)
        
        # 2. Калибруем уверенность
        enhanced_predictions = self.calibrate_confidence(enhanced_predictions)
        
        # 3. Применяем многоуровневую классификацию
        enhanced_predictions = self.hierarchical_classify(enhanced_predictions)
        
        # 4. Вычисляем финальную оценку
        enhanced_predictions['final_score'] = self._calculate_final_score(enhanced_predictions)
        
        return enhanced_predictions
    
    def _calculate_final_score(self, predictions: Dict[str, Any]) -> float:
        """Вычисляет финальную оценку с учетом всех улучшений"""
        scores = []
        weights = []
        
        for model_name, prediction in predictions.items():
            if model_name in self.ensemble_weights:
                weight = self.ensemble_weights[model_name]
                
                # Используем калиброванную уверенность если есть
                confidence = prediction.get('calibrated_confidence', 
                                          prediction.get('confidence', 0.5))
                
                scores.append(confidence)
                weights.append(weight)
        
        if scores and weights:
            return np.average(scores, weights=weights)
        else:
            return 0.5
    
    def get_enhancement_info(self) -> Dict[str, Any]:
        """Возвращает информацию об улучшениях"""
        return {
            'ensemble_weights': self.ensemble_weights,
            'calibration_methods': self.calibration_methods,
            'augmentation_transforms': list(self.augmentation_transforms.keys()),
            'hierarchy_levels': self.hierarchy_levels,
            'status': 'active'
        }
