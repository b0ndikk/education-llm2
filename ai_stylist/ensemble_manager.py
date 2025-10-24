#!/usr/bin/env python3
"""
🎯 СПЕЦИАЛИЗИРОВАННЫЙ АНСАМБЛЬ МОДЕЛЕЙ
- FashionCLIP: стиль, цвет, материал, сезон
- DeepFashion CNN: классификация одежды, категории
- YOLO: детекция объектов, композиция
- ViT: совместимость, сборка образов
"""

from fashion_clip import FashionCLIPAnalyzer
from yolo_detector import YOLODetector
from deepfashion_cnn import DeepFashionCNN
from accuracy_enhancements import AccuracyEnhancer
from PIL import Image
import numpy as np

class FashionEnsemble:
    def __init__(self):
        """Специализированный ансамбль с улучшениями точности"""
        print("🚀 Инициализация специализированного ансамбля...")
        
        # Основные модели
        self.fashion_clip = FashionCLIPAnalyzer()
        self.yolo = YOLODetector()
        self.deepfashion = DeepFashionCNN()
        
        # Система улучшения точности
        self.accuracy_enhancer = AccuracyEnhancer()
        
        print("✅ Специализированный ансамбль с улучшениями готов!")
    
    def analyze_image(self, image: Image.Image) -> dict:
        """Полный анализ изображения с улучшениями точности"""
        try:
            # Параллельный анализ
            clip_results = self.fashion_clip.analyze_image(image)
            yolo_results = self.yolo.detect_clothing(image)
            deepfashion_results = self.deepfashion.analyze_image(image)
            
            # Применяем улучшения точности
            enhanced_predictions = self._apply_accuracy_enhancements({
                'fashion_clip': clip_results,
                'yolo': yolo_results,
                'deepfashion': deepfashion_results
            })
            
            # Объединяем результаты с улучшениями
            combined_analysis = self._combine_enhanced_results(
                enhanced_predictions, image
            )
            
            return combined_analysis
            
        except Exception as e:
            return {"error": f"Ошибка ансамбля: {str(e)}"}
    
    def _apply_accuracy_enhancements(self, predictions: dict) -> dict:
        """Применяет все улучшения точности к предсказаниям"""
        try:
            # Применяем все улучшения
            enhanced_predictions = self.accuracy_enhancer.enhance_predictions(predictions)
            
            # Добавляем информацию об улучшениях
            enhanced_predictions['enhancement_info'] = self.accuracy_enhancer.get_enhancement_info()
            
            return enhanced_predictions
            
        except Exception as e:
            print(f"⚠️ Ошибка применения улучшений: {e}")
            return predictions  # Возвращаем оригинальные предсказания
    
    def _combine_enhanced_results(self, enhanced_predictions: dict, image: Image.Image) -> dict:
        """Объединяет результаты с улучшениями"""
        
        analysis = {
            'models_used': ['FashionCLIP', 'YOLO', 'DeepFashion CNN'],
            'enhancements_applied': ['Ensemble Weights', 'Confidence Calibration', 'Data Augmentation', 'Hierarchical Classification'],
            'timestamp': np.datetime64('now'),
        }
        
        # Добавляем улучшенные результаты
        analysis['enhanced_predictions'] = enhanced_predictions
        
        # Создаем интегрированный анализ
        integrated = self._create_enhanced_integrated_analysis(enhanced_predictions, image)
        analysis['integrated_analysis'] = integrated
        
        # Оценка качества с улучшениями
        analysis['quality_metrics'] = self._calculate_enhanced_quality_metrics(enhanced_predictions)
        
        return analysis
    
    def _create_enhanced_integrated_analysis(self, enhanced_predictions: dict, image: Image.Image) -> dict:
        """Создает интегрированный анализ с улучшениями"""
        integrated = {}
        
        # Получаем улучшенные предсказания
        clip_results = enhanced_predictions.get('fashion_clip', {})
        yolo_results = enhanced_predictions.get('yolo', {})
        deepfashion_results = enhanced_predictions.get('deepfashion', {})
        
        # СПЕЦИАЛИЗАЦИЯ: FashionCLIP - стиль, цвет, материал, сезон
        if 'error' not in clip_results:
            integrated['style_analysis'] = {
                'style': clip_results.get('style_occasion', {}).get('best_match', {}).get('item', 'unknown'),
                'style_confidence': clip_results.get('style_occasion', {}).get('best_match', {}).get('confidence', 0.0),
                'material': clip_results.get('material_fabric', {}).get('best_match', {}).get('item', 'unknown'),
                'material_confidence': clip_results.get('material_fabric', {}).get('best_match', {}).get('confidence', 0.0),
                'color': clip_results.get('color_pattern', {}).get('best_match', {}).get('item', 'unknown'),
                'color_confidence': clip_results.get('color_pattern', {}).get('best_match', {}).get('confidence', 0.0),
                'season': clip_results.get('season_weather', {}).get('best_match', {}).get('item', 'unknown'),
                'season_confidence': clip_results.get('season_weather', {}).get('best_match', {}).get('confidence', 0.0),
                'enhanced_confidence': clip_results.get('calibrated_confidence', clip_results.get('confidence', 0.0))
            }
        
        # СПЕЦИАЛИЗАЦИЯ: DeepFashion CNN - тип одежды, категория, форма
        if 'error' not in deepfashion_results:
            integrated['garment_classification'] = {
                'category': deepfashion_results.get('category', 'unknown'),
                'confidence': deepfashion_results.get('confidence', 0.0),
                'enhanced_confidence': deepfashion_results.get('calibrated_confidence', deepfashion_results.get('confidence', 0.0)),
                'hierarchical_prediction': deepfashion_results.get('refined_prediction', {}),
                'top_predictions': deepfashion_results.get('top_predictions', []),
                'model': 'DeepFashion CNN'
            }
        
        # СПЕЦИАЛИЗАЦИЯ: YOLO - детекция, композиция, расположение
        if 'error' not in yolo_results:
            integrated['object_detection'] = {
                'total_objects': yolo_results.get('total_items', 0),
                'layout': yolo_results.get('composition', {}).get('layout', 'unknown'),
                'dominant_region': yolo_results.get('composition', {}).get('dominant_region', 'unknown'),
                'has_clothing': yolo_results.get('has_clothing', False),
                'enhanced_confidence': yolo_results.get('calibrated_confidence', yolo_results.get('confidence', 0.0))
            }
        
        # Финальная оценка с улучшениями
        integrated['final_score'] = enhanced_predictions.get('final_score', 0.5)
        integrated['enhancement_impact'] = self._calculate_enhancement_impact(enhanced_predictions)
        
        # Сводная оценка
        integrated['summary'] = self._generate_enhanced_summary(integrated)
        
        return integrated
    
    def _calculate_enhancement_impact(self, enhanced_predictions: dict) -> dict:
        """Вычисляет влияние улучшений"""
        impact = {}
        
        for model_name, prediction in enhanced_predictions.items():
            if model_name in ['fashion_clip', 'yolo', 'deepfashion']:
                original_conf = prediction.get('confidence', 0.0)
                enhanced_conf = prediction.get('calibrated_confidence', original_conf)
                
                impact[model_name] = {
                    'original_confidence': original_conf,
                    'enhanced_confidence': enhanced_conf,
                    'improvement': enhanced_conf - original_conf,
                    'improvement_percent': ((enhanced_conf - original_conf) / original_conf * 100) if original_conf > 0 else 0
                }
        
        return impact
    
    def _calculate_enhanced_quality_metrics(self, enhanced_predictions: dict) -> dict:
        """Рассчитывает улучшенные метрики качества"""
        metrics = {}
        
        # Получаем финальную оценку
        final_score = enhanced_predictions.get('final_score', 0.5)
        metrics['enhanced_overall_quality'] = final_score
        
        # Специализированные метрики с улучшениями
        for model_name, prediction in enhanced_predictions.items():
            if model_name in ['fashion_clip', 'yolo', 'deepfashion']:
                enhanced_conf = prediction.get('calibrated_confidence', prediction.get('confidence', 0.0))
                metrics[f'{model_name}_enhanced_score'] = enhanced_conf
        
        # Влияние улучшений
        metrics['enhancement_impact'] = self._calculate_enhancement_impact(enhanced_predictions)
        
        return metrics
    
    def _generate_enhanced_summary(self, integrated: dict) -> dict:
        """Генерирует улучшенную сводку анализа"""
        summary = {}
        
        # Получаем специализированные результаты
        style_analysis = integrated.get('style_analysis', {})
        garment_classification = integrated.get('garment_classification', {})
        object_detection = integrated.get('object_detection', {})
        
        # Создаем описание с улучшениями
        summary['description'] = self._create_enhanced_description(
            style_analysis, garment_classification, object_detection
        )
        
        # Уровень уверенности с улучшениями
        style_conf = style_analysis.get('enhanced_confidence', style_analysis.get('style_confidence', 0.0))
        garment_conf = garment_classification.get('enhanced_confidence', garment_classification.get('confidence', 0.0))
        detection_conf = object_detection.get('enhanced_confidence', 0.0)
        
        summary['enhanced_confidence_level'] = np.mean([style_conf, garment_conf, detection_conf])
        summary['analysis_depth'] = 'enhanced_specialized'  # Улучшенный специализированный анализ
        
        return summary
    
    def _create_enhanced_description(self, style_analysis: dict, garment_classification: dict, object_detection: dict) -> str:
        """Создает улучшенное описание на основе анализа"""
        parts = []
        
        # Стиль и модные характеристики (FashionCLIP) с улучшениями
        if style_analysis.get('style') and style_analysis['style'] != 'unknown':
            enhanced_conf = style_analysis.get('enhanced_confidence', style_analysis.get('style_confidence', 0.0))
            parts.append(f"🎨 Стиль: {style_analysis['style']} ({enhanced_conf:.1%})")
        
        if style_analysis.get('material') and style_analysis['material'] != 'unknown':
            parts.append(f"🧵 Материал: {style_analysis['material']}")
        
        if style_analysis.get('color') and style_analysis['color'] != 'unknown':
            parts.append(f"🎨 Цвет: {style_analysis['color']}")
        
        if style_analysis.get('season') and style_analysis['season'] != 'unknown':
            parts.append(f"🌤️ Сезон: {style_analysis['season']}")
        
        # Классификация одежды (DeepFashion CNN) с улучшениями
        if garment_classification.get('category') and garment_classification['category'] != 'unknown':
            enhanced_conf = garment_classification.get('enhanced_confidence', garment_classification.get('confidence', 0.0))
            parts.append(f"👕 Тип: {garment_classification['category']} ({enhanced_conf:.1%})")
        
        # Детекция объектов (YOLO) с улучшениями
        if object_detection.get('total_objects', 0) > 0:
            parts.append(f"🔍 Объектов: {object_detection['total_objects']}")
        
        if object_detection.get('layout') and object_detection['layout'] != 'unknown':
            parts.append(f"📐 Композиция: {object_detection['layout']}")
        
        return " | ".join(parts) if parts else "Улучшенный специализированный анализ не дал четких результатов"
    
    def _combine_results(self, clip: dict, yolo: dict, deepfashion: dict, image: Image.Image) -> dict:
        """Объединяет результаты всех моделей"""
        
        analysis = {
            'models_used': ['FashionCLIP', 'YOLO', 'DeepFashion CNN'],
            'timestamp': np.datetime64('now'),
        }
        
        # Добавляем результаты каждой модели
        analysis['fashion_clip'] = clip
        analysis['yolo_detection'] = yolo
        analysis['deepfashion_analysis'] = deepfashion
        
        # Создаем интегрированный анализ
        integrated = self._create_integrated_analysis(clip, yolo, deepfashion, image)
        analysis['integrated_analysis'] = integrated
        
        # Оценка качества анализа
        analysis['quality_metrics'] = self._calculate_quality_metrics(clip, yolo, deepfashion)
        
        return analysis
    
    def _create_integrated_analysis(self, clip: dict, yolo: dict, deepfashion: dict, image: Image.Image) -> dict:
        """Создает интегрированный анализ из всех моделей с специализацией"""
        integrated = {}
        
        # СПЕЦИАЛИЗАЦИЯ: FashionCLIP - стиль, цвет, материал, сезон
        if 'error' not in clip:
            integrated['style_analysis'] = {
                'style': clip.get('style_occasion', {}).get('best_match', {}).get('item', 'unknown'),
                'style_confidence': clip.get('style_occasion', {}).get('best_match', {}).get('confidence', 0.0),
                'material': clip.get('material_fabric', {}).get('best_match', {}).get('item', 'unknown'),
                'material_confidence': clip.get('material_fabric', {}).get('best_match', {}).get('confidence', 0.0),
                'color': clip.get('color_pattern', {}).get('best_match', {}).get('item', 'unknown'),
                'color_confidence': clip.get('color_pattern', {}).get('best_match', {}).get('confidence', 0.0),
                'season': clip.get('season_weather', {}).get('best_match', {}).get('item', 'unknown'),
                'season_confidence': clip.get('season_weather', {}).get('best_match', {}).get('confidence', 0.0),
                'price_range': clip.get('price_range', {}).get('best_match', {}).get('item', 'unknown'),
                'price_confidence': clip.get('price_range', {}).get('best_match', {}).get('confidence', 0.0)
            }
        
        # СПЕЦИАЛИЗАЦИЯ: DeepFashion CNN - тип одежды, категория, форма
        if 'error' not in deepfashion:
            integrated['garment_classification'] = {
                'category': deepfashion.get('category', 'unknown'),
                'confidence': deepfashion.get('confidence', 0.0),
                'top_predictions': deepfashion.get('top_predictions', []),
                'model': 'DeepFashion CNN'
            }
        
        # СПЕЦИАЛИЗАЦИЯ: YOLO - детекция, композиция, расположение
        if 'error' not in yolo:
            integrated['object_detection'] = {
                'total_objects': yolo.get('total_items', 0),
                'layout': yolo.get('composition', {}).get('layout', 'unknown'),
                'dominant_region': yolo.get('composition', {}).get('dominant_region', 'unknown'),
                'has_clothing': yolo.get('has_clothing', False),
                'detection_confidence': yolo.get('confidence', 0.0)
            }
        
        # Сводная оценка
        integrated['summary'] = self._generate_summary(integrated)
        
        return integrated
    
    def _calculate_quality_metrics(self, clip: dict, yolo: dict, deepfashion: dict) -> dict:
        """Рассчитывает метрики качества анализа с учетом специализации"""
        metrics = {}
        
        # СПЕЦИАЛИЗИРОВАННЫЕ МЕТРИКИ
        
        # FashionCLIP - стиль и модные характеристики
        if 'error' not in clip:
            style_confidence = clip.get('style_occasion', {}).get('best_match', {}).get('confidence', 0.0)
            material_confidence = clip.get('material_fabric', {}).get('best_match', {}).get('confidence', 0.0)
            color_confidence = clip.get('color_pattern', {}).get('best_match', {}).get('confidence', 0.0)
            metrics['style_analysis_score'] = np.mean([style_confidence, material_confidence, color_confidence])
        else:
            metrics['style_analysis_score'] = 0.0
        
        # DeepFashion CNN - классификация одежды
        if 'error' not in deepfashion:
            metrics['garment_classification_score'] = deepfashion.get('confidence', 0.0)
        else:
            metrics['garment_classification_score'] = 0.0
        
        # YOLO - детекция объектов
        if 'error' not in yolo:
            metrics['object_detection_score'] = min(1.0, yolo.get('total_items', 0) * 0.3)
            metrics['has_clothing'] = yolo.get('has_clothing', False)
        else:
            metrics['object_detection_score'] = 0.0
            metrics['has_clothing'] = False
        
        # СПЕЦИАЛИЗИРОВАННЫЕ ВЕСА
        weights = {
            'style_analysis_score': 0.4,      # FashionCLIP - стиль
            'garment_classification_score': 0.3,  # DeepFashion - категория
            'object_detection_score': 0.3     # YOLO - детекция
        }
        
        # Взвешенная общая оценка
        weighted_scores = []
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_scores.append(metrics[metric] * weight)
        
        metrics['overall_quality'] = np.sum(weighted_scores) if weighted_scores else 0.5
        metrics['specialization_weights'] = weights
        
        return metrics
    
    def _generate_summary(self, integrated: dict) -> dict:
        """Генерирует сводку анализа с учетом специализации"""
        summary = {}
        
        # Получаем специализированные результаты
        style_analysis = integrated.get('style_analysis', {})
        garment_classification = integrated.get('garment_classification', {})
        object_detection = integrated.get('object_detection', {})
        
        # Создаем описание на основе специализации
        summary['description'] = self._create_specialized_description(
            style_analysis, garment_classification, object_detection
        )
        
        # Уровень уверенности на основе специализированных метрик
        style_conf = style_analysis.get('style_confidence', 0.0)
        garment_conf = garment_classification.get('confidence', 0.0)
        detection_conf = object_detection.get('detection_confidence', 0.0)
        
        summary['confidence_level'] = np.mean([style_conf, garment_conf, detection_conf])
        summary['analysis_depth'] = 'specialized'  # Специализированный анализ
        
        return summary
    
    def _create_specialized_description(self, style_analysis: dict, garment_classification: dict, object_detection: dict) -> str:
        """Создает специализированное описание на основе анализа"""
        parts = []
        
        # Стиль и модные характеристики (FashionCLIP)
        if style_analysis.get('style') and style_analysis['style'] != 'unknown':
            parts.append(f"🎨 Стиль: {style_analysis['style']}")
        
        if style_analysis.get('material') and style_analysis['material'] != 'unknown':
            parts.append(f"🧵 Материал: {style_analysis['material']}")
        
        if style_analysis.get('color') and style_analysis['color'] != 'unknown':
            parts.append(f"🎨 Цвет: {style_analysis['color']}")
        
        if style_analysis.get('season') and style_analysis['season'] != 'unknown':
            parts.append(f"🌤️ Сезон: {style_analysis['season']}")
        
        # Классификация одежды (DeepFashion CNN)
        if garment_classification.get('category') and garment_classification['category'] != 'unknown':
            parts.append(f"👕 Тип: {garment_classification['category']}")
        
        # Детекция объектов (YOLO)
        if object_detection.get('total_objects', 0) > 0:
            parts.append(f"🔍 Объектов: {object_detection['total_objects']}")
        
        if object_detection.get('layout') and object_detection['layout'] != 'unknown':
            parts.append(f"📐 Композиция: {object_detection['layout']}")
        
        return " | ".join(parts) if parts else "Специализированный анализ не дал четких результатов"
    
    def get_detailed_description(self, analysis: dict) -> str:
        """Генерирует детальное описание для интерфейса с учетом специализации"""
        if 'error' in analysis:
            return f"❌ Ошибка: {analysis['error']}"
        
        integrated = analysis.get('integrated_analysis', {})
        summary = integrated.get('summary', {})
        metrics = analysis.get('quality_metrics', {})
        
        description = []
        
        # Основное описание
        if 'description' in summary:
            description.append(f"📋 **ОПИСАНИЕ:** {summary['description']}")
        
        # СПЕЦИАЛИЗИРОВАННЫЕ ДЕТАЛИ
        description.append("\n🔍 **СПЕЦИАЛИЗИРОВАННЫЙ АНАЛИЗ:**")
        
        # FashionCLIP - стиль и модные характеристики
        if 'style_analysis' in integrated:
            style = integrated['style_analysis']
            description.append("\n🎨 **СТИЛЬ И МОДА (FashionCLIP):**")
            if style.get('style') != 'unknown':
                description.append(f"  • Стиль: {style['style']} ({style.get('style_confidence', 0):.1%})")
            if style.get('material') != 'unknown':
                description.append(f"  • Материал: {style['material']} ({style.get('material_confidence', 0):.1%})")
            if style.get('color') != 'unknown':
                description.append(f"  • Цвет: {style['color']} ({style.get('color_confidence', 0):.1%})")
            if style.get('season') != 'unknown':
                description.append(f"  • Сезон: {style['season']} ({style.get('season_confidence', 0):.1%})")
        
        # DeepFashion CNN - классификация одежды
        if 'garment_classification' in integrated:
            garment = integrated['garment_classification']
            description.append(f"\n👕 **КЛАССИФИКАЦИЯ (DeepFashion CNN):**")
            description.append(f"  • Тип: {garment.get('category', 'unknown')} ({garment.get('confidence', 0):.1%})")
            if garment.get('top_predictions'):
                description.append("  • Топ-3 предсказания:")
                for i, pred in enumerate(garment['top_predictions'][:3]):
                    description.append(f"    {i+1}. {pred['category']} ({pred['confidence']:.1%})")
        
        # YOLO - детекция объектов
        if 'object_detection' in integrated:
            detection = integrated['object_detection']
            description.append(f"\n🔍 **ДЕТЕКЦИЯ (YOLO):**")
            description.append(f"  • Объектов: {detection.get('total_objects', 0)}")
            description.append(f"  • Композиция: {detection.get('layout', 'unknown')}")
            description.append(f"  • Область: {detection.get('dominant_region', 'unknown')}")
        
        # Специализированные метрики
        description.append(f"\n📊 **СПЕЦИАЛИЗИРОВАННОЕ КАЧЕСТВО:**")
        if 'specialization_weights' in metrics:
            weights = metrics['specialization_weights']
            description.append(f"  • Стиль (FashionCLIP): {metrics.get('style_analysis_score', 0):.1%} (вес: {weights['style_analysis_score']:.1%})")
            description.append(f"  • Классификация (DeepFashion): {metrics.get('garment_classification_score', 0):.1%} (вес: {weights['garment_classification_score']:.1%})")
            description.append(f"  • Детекция (YOLO): {metrics.get('object_detection_score', 0):.1%} (вес: {weights['object_detection_score']:.1%})")
        
        description.append(f"\n🎯 **ОБЩЕЕ КАЧЕСТВО:** {metrics.get('overall_quality', 0):.1%}")
        
        return "\n".join(description)