#!/usr/bin/env python3
"""
🎯 АНСАМБЛЬ МОДЕЛЕЙ - FASHIONCLIP + YOLO + RESNET
"""

from fashion_clip import FashionCLIPAnalyzer
from yolo_detector import YOLODetector
from resnet_analyzer import ResNetAnalyzer
from PIL import Image
import numpy as np

class FashionEnsemble:
    def __init__(self):
        """Ансамбль из трех моделей"""
        print("🚀 Инициализация ансамбля моделей...")
        
        self.fashion_clip = FashionCLIPAnalyzer()
        self.yolo = YOLODetector()
        self.resnet = ResNetAnalyzer()
        
        print("✅ Ансамбль моделей готов!")
    
    def analyze_image(self, image: Image.Image) -> dict:
        """Полный анализ изображения всеми моделями"""
        try:
            # Параллельный анализ всеми моделями
            clip_results = self.fashion_clip.analyze_image(image)
            yolo_results = self.yolo.detect_clothing(image)
            resnet_results = self.resnet.analyze_image(image)
            
            # Объединяем результаты
            combined_analysis = self._combine_results(
                clip_results, yolo_results, resnet_results
            )
            
            return combined_analysis
            
        except Exception as e:
            return {"error": f"Ошибка ансамбля: {str(e)}"}
    
    def _combine_results(self, clip: dict, yolo: dict, resnet: dict) -> dict:
        """Объединяет результаты всех моделей"""
        
        # Базовая информация
        analysis = {
            'models_used': ['FashionCLIP', 'YOLO', 'ResNet'],
            'combined_confidence': 0.0
        }
        
        # Результаты от каждой модели
        if 'error' not in clip:
            analysis['fashion_clip'] = clip
        if 'error' not in yolo:
            analysis['yolo'] = yolo
        if 'error' not in resnet:
            analysis['resnet'] = resnet
        
        # Сводный анализ
        summary = self._create_summary(clip, yolo, resnet)
        analysis['summary'] = summary
        
        return analysis
    
    def _create_summary(self, clip: dict, yolo: dict, resnet: dict) -> dict:
        """Создает сводный отчет"""
        summary = {}
        
        # Информация от FashionCLIP
        if 'error' not in clip and 'garment_category' in clip:
            garment = clip['garment_category']['best_match']
            summary['garment_type'] = garment['item']
            summary['garment_confidence'] = garment['confidence']
        
        # Информация от YOLO
        if 'error' not in yolo:
            summary['detected_items'] = yolo.get('total_items', 0)
            summary['detection_region'] = yolo.get('dominant_region', 'unknown')
        
        # Информация от ResNet
        if 'error' not in resnet:
            if 'color_analysis' in resnet:
                summary['dominant_colors'] = resnet['color_analysis'].get('dominant_colors', [])
            summary['texture_complexity'] = resnet.get('texture_complexity', 'unknown')
        
        # Общая уверенность
        confidences = []
        if 'garment_confidence' in summary:
            confidences.append(summary['garment_confidence'])
        if 'detected_items' in summary and summary['detected_items'] > 0:
            confidences.append(0.8)  # YOLO обнаружил объекты
        
        if confidences:
            summary['overall_confidence'] = np.mean(confidences)
        else:
            summary['overall_confidence'] = 0.5
        
        return summary
    
    def get_detailed_description(self, analysis: dict) -> str:
        """Генерирует детальное описание на основе анализа"""
        if 'error' in analysis:
            return f"❌ Ошибка анализа: {analysis['error']}"
        
        summary = analysis.get('summary', {})
        
        description_parts = []
        
        # Основная информация
        if 'garment_type' in summary:
            description_parts.append(f"👕 **Тип одежды:** {summary['garment_type']}")
            description_parts.append(f"🎯 **Уверенность:** {summary.get('garment_confidence', 0):.1%}")
        
        # Детекция
        if 'detected_items' in summary:
            description_parts.append(f"🔍 **Обнаружено объектов:** {summary['detected_items']}")
        
        # Цвета
        if 'dominant_colors' in summary:
            colors = summary['dominant_colors']
            description_parts.append(f"🎨 **Доминирующие цвета:** {', '.join(colors)}")
        
        # Текстура
        if 'texture_complexity' in summary:
            texture = summary['texture_complexity']
            texture_map = {'smooth': 'гладкая', 'medium': 'средняя', 'complex': 'сложная'}
            description_parts.append(f"📐 **Текстура:** {texture_map.get(texture, texture)}")
        
        # Общая уверенность
        description_parts.append(f"📊 **Общая уверенность анализа:** {summary.get('overall_confidence', 0):.1%}")
        
        return "\n".join(description_parts)