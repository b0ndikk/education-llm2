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
            # Параллельный анализ (можно распараллелить при необходимости)
            clip_results = self.fashion_clip.analyze_image(image)
            yolo_results = self.yolo.detect_clothing(image)
            resnet_results = self.resnet.analyze_image(image)
            
            # Объединяем результаты
            combined_analysis = self._combine_results(
                clip_results, yolo_results, resnet_results, image
            )
            
            return combined_analysis
            
        except Exception as e:
            return {"error": f"Ошибка ансамбля: {str(e)}"}
    
    def _combine_results(self, clip: dict, yolo: dict, resnet: dict, image: Image.Image) -> dict:
        """Объединяет результаты всех моделей"""
        
        analysis = {
            'models_used': ['FashionCLIP', 'YOLO', 'ResNet50'],
            'timestamp': np.datetime64('now'),
        }
        
        # Добавляем результаты каждой модели
        analysis['fashion_clip'] = clip
        analysis['yolo_detection'] = yolo
        analysis['resnet_analysis'] = resnet
        
        # Создаем интегрированный анализ
        integrated = self._create_integrated_analysis(clip, yolo, resnet, image)
        analysis['integrated_analysis'] = integrated
        
        # Оценка качества анализа
        analysis['quality_metrics'] = self._calculate_quality_metrics(clip, yolo, resnet)
        
        return analysis
    
    def _create_integrated_analysis(self, clip: dict, yolo: dict, resnet: dict, image: Image.Image) -> dict:
        """Создает интегрированный анализ из всех моделей"""
        integrated = {}
        
        # Информация о предмете одежды
        if 'error' not in clip and 'garment_category' in clip:
            garment_data = clip['garment_category']['best_match']
            integrated['primary_garment'] = {
                'type': garment_data['item'],
                'confidence': garment_data['confidence'],
                'style': clip.get('style_occasion', {}).get('best_match', {}).get('item', 'unknown'),
                'material': clip.get('material_fabric', {}).get('best_match', {}).get('item', 'unknown'),
                'color': clip.get('color_pattern', {}).get('best_match', {}).get('item', 'unknown')
            }
        
        # Информация о композиции
        if 'error' not in yolo:
            integrated['composition'] = {
                'total_objects': yolo.get('total_items', 0),
                'layout': yolo.get('composition', {}).get('layout', 'unknown'),
                'dominant_region': yolo.get('composition', {}).get('dominant_region', 'unknown'),
                'has_clothing': yolo.get('has_clothing', False)
            }
        
        # Визуальные характеристики
        if 'error' not in resnet:
            integrated['visual_characteristics'] = {
                'texture_complexity': resnet.get('texture_complexity', 'unknown'),
                'color_palette': resnet.get('color_analysis', {}).get('dominant_colors', []),
                'feature_richness': resnet.get('texture_features', {}).get('feature_std', 0)
            }
        
        # Сводная оценка
        integrated['summary'] = self._generate_summary(integrated)
        
        return integrated
    
    def _calculate_quality_metrics(self, clip: dict, yolo: dict, resnet: dict) -> dict:
        """Рассчитывает метрики качества анализа"""
        metrics = {}
        
        # Уверенность FashionCLIP
        if 'error' not in clip and 'garment_category' in clip:
            metrics['clip_confidence'] = clip['garment_category']['best_match']['confidence']
        else:
            metrics['clip_confidence'] = 0.0
        
        # Качество детекции
        if 'error' not in yolo:
            metrics['detection_score'] = min(1.0, yolo.get('total_items', 0) * 0.3)
            metrics['has_clothing'] = yolo.get('has_clothing', False)
        else:
            metrics['detection_score'] = 0.0
            metrics['has_clothing'] = False
        
        # Качество визуального анализа
        if 'error' not in resnet:
            metrics['visual_analysis_score'] = 0.8
        else:
            metrics['visual_analysis_score'] = 0.2
        
        # Общая оценка
        scores = [v for k, v in metrics.items() if 'score' in k or 'confidence' in k]
        metrics['overall_quality'] = np.mean(scores) if scores else 0.5
        
        return metrics
    
    def _generate_summary(self, integrated: dict) -> dict:
        """Генерирует сводку анализа"""
        summary = {}
        
        garment = integrated.get('primary_garment', {})
        composition = integrated.get('composition', {})
        visual = integrated.get('visual_characteristics', {})
        
        summary['description'] = self._create_text_description(garment, composition, visual)
        summary['confidence_level'] = garment.get('confidence', 0.5)
        summary['analysis_depth'] = 'deep' if composition.get('total_objects', 0) > 0 else 'basic'
        
        return summary
    
    def _create_text_description(self, garment: dict, composition: dict, visual: dict) -> str:
        """Создает текстовое описание на основе анализа"""
        parts = []
        
        if garment.get('type'):
            parts.append(f"Основной предмет: {garment['type']}")
        
        if garment.get('style') and garment['style'] != 'unknown':
            parts.append(f"Стиль: {garment['style']}")
        
        if garment.get('material') and garment['material'] != 'unknown':
            parts.append(f"Материал: {garment['material']}")
        
        if composition.get('total_objects', 0) > 0:
            parts.append(f"Обнаружено объектов: {composition['total_objects']}")
        
        if visual.get('texture_complexity'):
            parts.append(f"Текстура: {visual['texture_complexity']}")
        
        return " | ".join(parts) if parts else "Анализ не дал четких результатов"
    
    def get_detailed_description(self, analysis: dict) -> str:
        """Генерирует детальное описание для интерфейса"""
        if 'error' in analysis:
            return f"❌ Ошибка: {analysis['error']}"
        
        integrated = analysis.get('integrated_analysis', {})
        summary = integrated.get('summary', {})
        metrics = analysis.get('quality_metrics', {})
        
        description = []
        
        # Основное описание
        if 'description' in summary:
            description.append(f"📋 **ОПИСАНИЕ:** {summary['description']}")
        
        # Детали от моделей
        description.append("\n🔍 **ДЕТАЛИ АНАЛИЗА:**")
        
        if 'fashion_clip' in analysis and 'error' not in analysis['fashion_clip']:
            clip = analysis['fashion_clip']
            for category, data in clip.items():
                if category != 'error' and 'best_match' in data:
                    best = data['best_match']
                    description.append(f"  • {category}: {best['item']} ({best['confidence']:.1%})")
        
        # Метрики качества
        description.append(f"\n📊 **КАЧЕСТВО АНАЛИЗА:** {metrics.get('overall_quality', 0):.1%}")
        
        return "\n".join(description)