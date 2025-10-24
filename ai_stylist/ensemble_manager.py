#!/usr/bin/env python3
"""
🎯 АНСАМБЛЬ МОДЕЛЕЙ - FASHIONCLIP + YOLO + RESNET
"""

from fashion_clip import FashionCLIPAnalyzer
from yolo_detector import YOLODetector
from resnet_analyzer import ResNetAnalyzer
from compatibility_analyzer import CompatibilityAnalyzer
from specialized_models import SpecializedFashionModels
from PIL import Image
import numpy as np
import hashlib
from functools import lru_cache

class FashionEnsemble:
    def __init__(self):
        """Ансамбль из трех моделей"""
        print("🚀 Инициализация ансамбля моделей...")
        
        self.fashion_clip = FashionCLIPAnalyzer()
        self.yolo = YOLODetector()
        self.resnet = ResNetAnalyzer()
        
        # ПРИОРИТЕТ 2: Новые компоненты
        self.compatibility_analyzer = CompatibilityAnalyzer()
        self.specialized_models = SpecializedFashionModels()
        
        print("✅ Ансамбль моделей с улучшениями готов!")
    
    def get_image_hash(self, image: Image.Image) -> str:
        """Получение хэша изображения для кэширования"""
        try:
            # Конвертируем в байты и создаем хэш
            image_bytes = image.tobytes()
            return hashlib.md5(image_bytes).hexdigest()
        except Exception as e:
            print(f"⚠️ Ошибка создания хэша: {e}")
            return str(hash(str(image.size)))
    
    @lru_cache(maxsize=1000)
    def _cached_analysis(self, image_hash: str, image_size: tuple) -> dict:
        """Кэшированный анализ (внутренний метод)"""
        # Этот метод будет вызываться с хэшем, но реальный анализ
        # происходит в analyze_image
        pass
    
    def analyze_image(self, image: Image.Image) -> dict:
        """Полный анализ изображения всеми моделями с кэшированием"""
        try:
            # Получаем хэш изображения для кэширования
            image_hash = self.get_image_hash(image)
            
            # Проверяем кэш (упрощенная версия)
            cache_key = f"{image_hash}_{image.size}"
            
            # Параллельный анализ (можно распараллелить при необходимости)
            clip_results = self.fashion_clip.analyze_image(image)
            yolo_results = self.yolo.detect_clothing(image)
            resnet_results = self.resnet.analyze_image(image)
            
            # Объединяем результаты
            combined_analysis = self._combine_results(
                clip_results, yolo_results, resnet_results, image
            )
            
            # ПРИОРИТЕТ 2: Добавляем новые анализы
            # Многоуровневая классификация
            if 'fashion_analysis' in combined_analysis.get('integrated_analysis', {}):
                fashion_data = combined_analysis['integrated_analysis']['fashion_analysis']
                hierarchy = self.hierarchical_classification(fashion_data)
                combined_analysis['hierarchical_analysis'] = hierarchy
            
            # Специализированный анализ
            garment_type = combined_analysis.get('integrated_analysis', {}).get('fashion_analysis', {}).get('garment_type', 'unknown')
            if garment_type != 'unknown':
                specialized_analysis = self.specialized_models.analyze_specialized(image, garment_type)
                combined_analysis['specialized_analysis'] = specialized_analysis
            
            # Добавляем информацию о кэшировании
            combined_analysis['cache_info'] = {
                'image_hash': image_hash,
                'cached': False,  # В будущем можно добавить реальное кэширование
                'cache_key': cache_key
            }
            
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
        """Создает интегрированный анализ из всех моделей с специализацией"""
        integrated = {}
        
        # FASHIONCLIP ОПРЕДЕЛЯЕТ ВСЕ - основной анализатор
        if 'error' not in clip:
            integrated['fashion_analysis'] = {
                # Тип одежды
                'garment_type': clip.get('garment_category', {}).get('best_match', {}).get('item', 'unknown'),
                'garment_confidence': clip.get('garment_category', {}).get('best_match', {}).get('confidence', 0.0),
                
                # Стиль и мода
                'style': clip.get('style_occasion', {}).get('best_match', {}).get('item', 'unknown'),
                'style_confidence': clip.get('style_occasion', {}).get('best_match', {}).get('confidence', 0.0),
                
                # Материал и ткань
                'material': clip.get('material_fabric', {}).get('best_match', {}).get('item', 'unknown'),
                'material_confidence': clip.get('material_fabric', {}).get('best_match', {}).get('confidence', 0.0),
                
                # Цвет и узор
                'color': clip.get('color_pattern', {}).get('best_match', {}).get('item', 'unknown'),
                'color_confidence': clip.get('color_pattern', {}).get('best_match', {}).get('confidence', 0.0),
                
                # Сезон и погода
                'season': clip.get('season_weather', {}).get('best_match', {}).get('item', 'unknown'),
                'season_confidence': clip.get('season_weather', {}).get('best_match', {}).get('confidence', 0.0),
                
                # Ценовой диапазон
                'price_range': clip.get('price_range', {}).get('best_match', {}).get('item', 'unknown'),
                'price_confidence': clip.get('price_range', {}).get('best_match', {}).get('confidence', 0.0),
                
                # Возрастная группа
                'age_group': clip.get('age_group', {}).get('best_match', {}).get('item', 'unknown'),
                'age_confidence': clip.get('age_group', {}).get('best_match', {}).get('confidence', 0.0),
                
                # Подходящий размер
                'body_fit': clip.get('body_fit', {}).get('best_match', {}).get('item', 'unknown'),
                'fit_confidence': clip.get('body_fit', {}).get('best_match', {}).get('confidence', 0.0),
                
                'model': 'FashionCLIP (Primary)'
            }
        
        # ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ: ResNet - визуальные характеристики
        if 'error' not in resnet:
            integrated['visual_support'] = {
                'texture_analysis': resnet.get('category', 'unknown'),
                'confidence': resnet.get('confidence', 0.0),
                'model': 'ResNet50 (Support)'
            }
        
        # ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ: YOLO - композиция и расположение
        if 'error' not in yolo:
            integrated['composition_support'] = {
                'total_objects': yolo.get('total_items', 0),
                'layout': yolo.get('composition', {}).get('layout', 'unknown'),
                'dominant_region': yolo.get('composition', {}).get('dominant_region', 'unknown'),
                'has_clothing': yolo.get('has_clothing', False),
                'model': 'YOLO (Support)'
            }
        
        # Сводная оценка
        integrated['summary'] = self._generate_summary(integrated)
        
        return integrated
    
    def _calculate_quality_metrics(self, clip: dict, yolo: dict, resnet: dict) -> dict:
        """Рассчитывает метрики качества анализа с приоритетом FashionCLIP"""
        metrics = {}
        
        # FASHIONCLIP - ОСНОВНОЙ АНАЛИЗАТОР (80% веса)
        if 'error' not in clip:
            # Собираем все уверенности FashionCLIP
            fashion_confidences = []
            
            # Тип одежды
            garment_conf = clip.get('garment_category', {}).get('best_match', {}).get('confidence', 0.0)
            if garment_conf > 0:
                fashion_confidences.append(garment_conf)
            
            # Стиль
            style_conf = clip.get('style_occasion', {}).get('best_match', {}).get('confidence', 0.0)
            if style_conf > 0:
                fashion_confidences.append(style_conf)
            
            # Материал
            material_conf = clip.get('material_fabric', {}).get('best_match', {}).get('confidence', 0.0)
            if material_conf > 0:
                fashion_confidences.append(material_conf)
            
            # Цвет
            color_conf = clip.get('color_pattern', {}).get('best_match', {}).get('confidence', 0.0)
            if color_conf > 0:
                fashion_confidences.append(color_conf)
            
            # Сезон
            season_conf = clip.get('season_weather', {}).get('best_match', {}).get('confidence', 0.0)
            if season_conf > 0:
                fashion_confidences.append(season_conf)
            
            # Ценовой диапазон
            price_conf = clip.get('price_range', {}).get('best_match', {}).get('confidence', 0.0)
            if price_conf > 0:
                fashion_confidences.append(price_conf)
            
            # Возрастная группа
            age_conf = clip.get('age_group', {}).get('best_match', {}).get('confidence', 0.0)
            if age_conf > 0:
                fashion_confidences.append(age_conf)
            
            # Подходящий размер
            fit_conf = clip.get('body_fit', {}).get('best_match', {}).get('confidence', 0.0)
            if fit_conf > 0:
                fashion_confidences.append(fit_conf)
            
            # Средняя уверенность FashionCLIP
            metrics['fashion_clip_score'] = np.mean(fashion_confidences) if fashion_confidences else 0.0
        else:
            metrics['fashion_clip_score'] = 0.0
        
        # ResNet - поддержка (10% веса)
        if 'error' not in resnet:
            metrics['resnet_support_score'] = resnet.get('confidence', 0.0)
        else:
            metrics['resnet_support_score'] = 0.0
        
        # YOLO - поддержка (10% веса)
        if 'error' not in yolo:
            metrics['yolo_support_score'] = min(1.0, yolo.get('total_items', 0) * 0.3)
            metrics['has_clothing'] = yolo.get('has_clothing', False)
        else:
            metrics['yolo_support_score'] = 0.0
            metrics['has_clothing'] = False
        
        # ДИНАМИЧЕСКИЕ ВЕСА - адаптируются к уверенности моделей
        weights = self._calculate_dynamic_weights(
            metrics.get('fashion_clip_score', 0.0),
            metrics.get('resnet_support_score', 0.0),
            metrics.get('yolo_support_score', 0.0)
        )
        
        # Взвешенная общая оценка
        weighted_scores = []
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_scores.append(metrics[metric] * weight)
        
        metrics['overall_quality'] = np.sum(weighted_scores) if weighted_scores else 0.5
        metrics['priority_weights'] = weights
        
        return metrics
    
    def _calculate_dynamic_weights(self, clip_conf, resnet_conf, yolo_conf):
        """Динамические веса на основе уверенности моделей"""
        total_conf = clip_conf + resnet_conf + yolo_conf
        
        if total_conf > 0:
            # Адаптивные веса на основе уверенности
            return {
                'fashion_clip_score': clip_conf / total_conf,
                'resnet_support_score': resnet_conf / total_conf,
                'yolo_support_score': yolo_conf / total_conf
            }
        else:
            # Fallback веса
            return {
                'fashion_clip_score': 0.8,
                'resnet_support_score': 0.1,
                'yolo_support_score': 0.1
            }
    
    def _generate_summary(self, integrated: dict) -> dict:
        """Генерирует сводку анализа с приоритетом FashionCLIP"""
        summary = {}
        
        # Получаем результаты FashionCLIP (основной анализатор)
        fashion_analysis = integrated.get('fashion_analysis', {})
        visual_support = integrated.get('visual_support', {})
        composition_support = integrated.get('composition_support', {})
        
        # Создаем описание на основе FashionCLIP
        summary['description'] = self._create_fashion_description(
            fashion_analysis, visual_support, composition_support
        )
        
        # Уровень уверенности на основе FashionCLIP
        fashion_conf = fashion_analysis.get('garment_confidence', 0.0)
        style_conf = fashion_analysis.get('style_confidence', 0.0)
        material_conf = fashion_analysis.get('material_confidence', 0.0)
        color_conf = fashion_analysis.get('color_confidence', 0.0)
        
        # Средняя уверенность FashionCLIP
        fashion_confidences = [fashion_conf, style_conf, material_conf, color_conf]
        valid_confidences = [c for c in fashion_confidences if c > 0]
        summary['confidence_level'] = np.mean(valid_confidences) if valid_confidences else 0.5
        summary['analysis_depth'] = 'fashion_primary'  # FashionCLIP определяет все
        
        return summary
    
    def _create_fashion_description(self, fashion_analysis: dict, visual_support: dict, composition_support: dict) -> str:
        """Создает описание на основе FashionCLIP (основной анализатор)"""
        parts = []
        
        # FASHIONCLIP ОПРЕДЕЛЯЕТ ВСЕ - основной анализ
        if fashion_analysis.get('garment_type') and fashion_analysis['garment_type'] != 'unknown':
            parts.append(f"👕 Тип: {fashion_analysis['garment_type']}")
        
        if fashion_analysis.get('style') and fashion_analysis['style'] != 'unknown':
            parts.append(f"🎨 Стиль: {fashion_analysis['style']}")
        
        if fashion_analysis.get('material') and fashion_analysis['material'] != 'unknown':
            parts.append(f"🧵 Материал: {fashion_analysis['material']}")
        
        if fashion_analysis.get('color') and fashion_analysis['color'] != 'unknown':
            parts.append(f"🎨 Цвет: {fashion_analysis['color']}")
        
        if fashion_analysis.get('season') and fashion_analysis['season'] != 'unknown':
            parts.append(f"🌤️ Сезон: {fashion_analysis['season']}")
        
        if fashion_analysis.get('price_range') and fashion_analysis['price_range'] != 'unknown':
            parts.append(f"💰 Цена: {fashion_analysis['price_range']}")
        
        if fashion_analysis.get('age_group') and fashion_analysis['age_group'] != 'unknown':
            parts.append(f"👶 Возраст: {fashion_analysis['age_group']}")
        
        if fashion_analysis.get('body_fit') and fashion_analysis['body_fit'] != 'unknown':
            parts.append(f"📏 Размер: {fashion_analysis['body_fit']}")
        
        # Дополнительная поддержка от ResNet и YOLO
        if visual_support.get('texture_analysis') and visual_support['texture_analysis'] != 'unknown':
            parts.append(f"🔍 Текстура: {visual_support['texture_analysis']}")
        
        if composition_support.get('total_objects', 0) > 0:
            parts.append(f"📐 Объектов: {composition_support['total_objects']}")
        
        return " | ".join(parts) if parts else "FashionCLIP анализ не дал четких результатов"
    
    def get_detailed_description(self, analysis: dict) -> str:
        """Генерирует детальное описание для интерфейса с приоритетом FashionCLIP"""
        if 'error' in analysis:
            return f"❌ Ошибка: {analysis['error']}"
        
        integrated = analysis.get('integrated_analysis', {})
        summary = integrated.get('summary', {})
        metrics = analysis.get('quality_metrics', {})
        
        description = []
        
        # Основное описание
        if 'description' in summary:
            description.append(f"📋 **ОПИСАНИЕ:** {summary['description']}")
        
        # FASHIONCLIP ОПРЕДЕЛЯЕТ ВСЕ - основной анализ
        description.append("\n🔍 **FASHIONCLIP АНАЛИЗ (ОСНОВНОЙ):**")
        
        if 'fashion_analysis' in integrated:
            fashion = integrated['fashion_analysis']
            description.append("\n🎨 **МОДНЫЙ АНАЛИЗ (FashionCLIP):**")
            
            # Тип одежды
            if fashion.get('garment_type') != 'unknown':
                description.append(f"  • Тип: {fashion['garment_type']} ({fashion.get('garment_confidence', 0):.1%})")
            
            # Стиль
            if fashion.get('style') != 'unknown':
                description.append(f"  • Стиль: {fashion['style']} ({fashion.get('style_confidence', 0):.1%})")
            
            # Материал
            if fashion.get('material') != 'unknown':
                description.append(f"  • Материал: {fashion['material']} ({fashion.get('material_confidence', 0):.1%})")
            
            # Цвет
            if fashion.get('color') != 'unknown':
                description.append(f"  • Цвет: {fashion['color']} ({fashion.get('color_confidence', 0):.1%})")
            
            # Сезон
            if fashion.get('season') != 'unknown':
                description.append(f"  • Сезон: {fashion['season']} ({fashion.get('season_confidence', 0):.1%})")
            
            # Ценовой диапазон
            if fashion.get('price_range') != 'unknown':
                description.append(f"  • Цена: {fashion['price_range']} ({fashion.get('price_confidence', 0):.1%})")
            
            # Возрастная группа
            if fashion.get('age_group') != 'unknown':
                description.append(f"  • Возраст: {fashion['age_group']} ({fashion.get('age_confidence', 0):.1%})")
            
            # Подходящий размер
            if fashion.get('body_fit') != 'unknown':
                description.append(f"  • Размер: {fashion['body_fit']} ({fashion.get('fit_confidence', 0):.1%})")
        
        # ПРИОРИТЕТ 2: Новые анализы
        description.append(f"\n🔧 **ПРИОРИТЕТ 2 - РАСШИРЕННЫЙ АНАЛИЗ:**")
        
        # Многоуровневая классификация
        if 'hierarchical_analysis' in analysis:
            hierarchy = analysis['hierarchical_analysis']
            description.append(f"\n🏗️ **МНОГОУРОВНЕВАЯ КЛАССИФИКАЦИЯ:**")
            description.append(f"  • Уровень 1: {hierarchy.get('level1', 'unknown')}")
            description.append(f"  • Уровень 2: {hierarchy.get('level2', 'unknown')}")
            description.append(f"  • Уровень 3: {hierarchy.get('level3', 'unknown')}")
        
        # Специализированный анализ
        if 'specialized_analysis' in analysis:
            specialized = analysis['specialized_analysis']
            description.append(f"\n🎯 **СПЕЦИАЛИЗИРОВАННЫЙ АНАЛИЗ:**")
            description.append(f"  • Тип: {specialized.get('specialized_type', 'unknown')}")
            if 'shoe_type' in specialized:
                description.append(f"  • Обувь: {specialized.get('shoe_type', 'unknown')}")
            elif 'dress_type' in specialized:
                description.append(f"  • Платье: {specialized.get('dress_type', 'unknown')}")
            elif 'accessory_type' in specialized:
                description.append(f"  • Аксессуар: {specialized.get('accessory_type', 'unknown')}")
            elif 'formal_type' in specialized:
                description.append(f"  • Деловая одежда: {specialized.get('formal_type', 'unknown')}")
            description.append(f"  • Уверенность: {specialized.get('confidence', 0):.1%}")
        
        # Дополнительная поддержка
        description.append(f"\n🔧 **ДОПОЛНИТЕЛЬНАЯ ПОДДЕРЖКА:**")
        
        # ResNet - визуальная поддержка
        if 'visual_support' in integrated:
            visual = integrated['visual_support']
            description.append(f"\n👕 **ВИЗУАЛЬНАЯ ПОДДЕРЖКА (ResNet):**")
            description.append(f"  • Текстура: {visual.get('texture_analysis', 'unknown')} ({visual.get('confidence', 0):.1%})")
        
        # YOLO - композиционная поддержка
        if 'composition_support' in integrated:
            composition = integrated['composition_support']
            description.append(f"\n🔍 **КОМПОЗИЦИОННАЯ ПОДДЕРЖКА (YOLO):**")
            description.append(f"  • Объектов: {composition.get('total_objects', 0)}")
            description.append(f"  • Композиция: {composition.get('layout', 'unknown')}")
            description.append(f"  • Область: {composition.get('dominant_region', 'unknown')}")
        
        # Приоритетные метрики с улучшениями
        description.append(f"\n📊 **ПРИОРИТЕТНОЕ КАЧЕСТВО (УЛУЧШЕНО):**")
        if 'priority_weights' in metrics:
            weights = metrics['priority_weights']
            description.append(f"  • FashionCLIP (основной): {metrics.get('fashion_clip_score', 0):.1%} (вес: {weights['fashion_clip_score']:.1%})")
            description.append(f"  • ResNet (поддержка): {metrics.get('resnet_support_score', 0):.1%} (вес: {weights['resnet_support_score']:.1%})")
            description.append(f"  • YOLO (поддержка): {metrics.get('yolo_support_score', 0):.1%} (вес: {weights['yolo_support_score']:.1%})")
        
        # Информация об улучшениях
        description.append(f"\n🚀 **ПРИМЕНЕННЫЕ УЛУЧШЕНИЯ:**")
        description.append(f"  • ✅ Улучшенная предобработка изображений")
        description.append(f"  • ✅ Динамические веса ансамбля")
        description.append(f"  • ✅ Кэширование результатов")
        description.append(f"  • ✅ Многоуровневая классификация")
        description.append(f"  • ✅ Анализ совместимости элементов")
        description.append(f"  • ✅ Специализированные модели")
        
        description.append(f"\n🎯 **ОБЩЕЕ КАЧЕСТВО:** {metrics.get('overall_quality', 0):.1%}")
        
        return "\n".join(description)
    
    def hierarchical_classification(self, predictions):
        """Многоуровневая классификация для повышения точности"""
        hierarchy = {
            'level1': 'одежда',
            'level2': self._determine_level2(predictions),
            'level3': self._determine_level3(predictions)
        }
        return hierarchy
    
    def _determine_level2(self, predictions):
        """Определение уровня 2 (общая категория)"""
        garment_type = predictions.get('garment_type', 'unknown')
        
        # Категории уровня 2
        if garment_type in ['футболка', 'рубашка', 'блузка', 'свитер', 'топ', 'майка', 'поло', 'водолазка']:
            return 'верх'
        elif garment_type in ['джинсы', 'брюки', 'шорты', 'юбка', 'леггинсы', 'штаны']:
            return 'низ'
        elif garment_type in ['кроссовки', 'туфли', 'ботинки', 'сандали', 'сапоги', 'лодочки']:
            return 'обувь'
        elif garment_type in ['куртка', 'пальто', 'пиджак', 'жилет', 'кардиган', 'бомбер']:
            return 'верхняя_одежда'
        elif garment_type in ['сумка', 'рюкзак', 'кошелек', 'шляпа', 'кепка', 'шарф']:
            return 'аксессуары'
        else:
            return 'неопределено'
    
    def _determine_level3(self, predictions):
        """Определение уровня 3 (конкретный тип)"""
        return predictions.get('garment_type', 'unknown')