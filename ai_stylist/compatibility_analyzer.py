#!/usr/bin/env python3
"""
🔍 АНАЛИЗ СОВМЕСТИМОСТИ ЭЛЕМЕНТОВ ОДЕЖДЫ
"""

import numpy as np
from typing import Dict, List, Tuple

class CompatibilityAnalyzer:
    """Анализатор совместимости элементов одежды"""
    
    def __init__(self):
        """Инициализация анализатора совместимости"""
        print("🔍 Инициализация анализатора совместимости...")
        
        # Цветовые гармонии
        self.color_harmonies = {
            'monochromatic': ['черный', 'белый', 'серый'],
            'complementary': [('красный', 'зеленый'), ('синий', 'оранжевый'), ('желтый', 'фиолетовый')],
            'analogous': [('красный', 'оранжевый'), ('синий', 'фиолетовый'), ('зеленый', 'желтый')],
            'triadic': [('красный', 'синий', 'желтый'), ('оранжевый', 'зеленый', 'фиолетовый')]
        }
        
        # Стилевые совместимости
        self.style_compatibilities = {
            'casual': ['повседневный', 'спортивный', 'минималистичный'],
            'formal': ['деловой', 'элегантный', 'классический'],
            'sport': ['спортивный', 'активный', 'функциональный'],
            'romantic': ['романтический', 'женственный', 'нежный']
        }
        
        # Сезонные совместимости
        self.season_compatibilities = {
            'зима': ['зимний', 'теплый', 'уютный'],
            'лето': ['летний', 'легкий', 'свежий'],
            'весна': ['весенний', 'яркий', 'обновляющийся'],
            'осень': ['осенний', 'теплый', 'уютный']
        }
        
        print("✅ Анализатор совместимости готов!")
    
    def check_color_compatibility(self, color1: str, color2: str) -> float:
        """Проверка совместимости цветов"""
        try:
            # Нормализуем цвета
            color1 = color1.lower().strip()
            color2 = color2.lower().strip()
            
            # Если цвета одинаковые - высокая совместимость
            if color1 == color2:
                return 0.9
            
            # Проверяем монохромную гармонию
            if self._check_monochromatic(color1, color2):
                return 0.8
            
            # Проверяем дополнительные цвета
            if self._check_complementary(color1, color2):
                return 0.7
            
            # Проверяем аналогичные цвета
            if self._check_analogous(color1, color2):
                return 0.6
            
            # Проверяем триадические цвета
            if self._check_triadic(color1, color2):
                return 0.5
            
            # Нейтральные цвета совместимы с большинством
            if color1 in ['черный', 'белый', 'серый', 'бежевый'] or color2 in ['черный', 'белый', 'серый', 'бежевый']:
                return 0.4
            
            # Базовая совместимость
            return 0.3
            
        except Exception as e:
            print(f"⚠️ Ошибка анализа цветов: {e}")
            return 0.5
    
    def _check_monochromatic(self, color1: str, color2: str) -> bool:
        """Проверка монохромной гармонии"""
        return color1 in self.color_harmonies['monochromatic'] and color2 in self.color_harmonies['monochromatic']
    
    def _check_complementary(self, color1: str, color2: str) -> bool:
        """Проверка дополнительных цветов"""
        for pair in self.color_harmonies['complementary']:
            if (color1 in pair and color2 in pair):
                return True
        return False
    
    def _check_analogous(self, color1: str, color2: str) -> bool:
        """Проверка аналогичных цветов"""
        for pair in self.color_harmonies['analogous']:
            if (color1 in pair and color2 in pair):
                return True
        return False
    
    def _check_triadic(self, color1: str, color2: str) -> bool:
        """Проверка триадических цветов"""
        for triad in self.color_harmonies['triadic']:
            if (color1 in triad and color2 in triad):
                return True
        return False
    
    def check_style_compatibility(self, style1: str, style2: str) -> float:
        """Проверка совместимости стилей"""
        try:
            style1 = style1.lower().strip()
            style2 = style2.lower().strip()
            
            # Если стили одинаковые - высокая совместимость
            if style1 == style2:
                return 0.9
            
            # Проверяем совместимые стили
            for main_style, compatible_styles in self.style_compatibilities.items():
                if style1 == main_style and style2 in compatible_styles:
                    return 0.8
                if style2 == main_style and style1 in compatible_styles:
                    return 0.8
            
            # Проверяем обратную совместимость
            for main_style, compatible_styles in self.style_compatibilities.items():
                if style1 in compatible_styles and style2 in compatible_styles:
                    return 0.7
            
            # Нейтральные стили совместимы с большинством
            if style1 in ['повседневный', 'универсальный'] or style2 in ['повседневный', 'универсальный']:
                return 0.6
            
            # Базовая совместимость
            return 0.4
            
        except Exception as e:
            print(f"⚠️ Ошибка анализа стилей: {e}")
            return 0.5
    
    def check_season_compatibility(self, season1: str, season2: str) -> float:
        """Проверка совместимости сезонов"""
        try:
            season1 = season1.lower().strip()
            season2 = season2.lower().strip()
            
            # Если сезоны одинаковые - высокая совместимость
            if season1 == season2:
                return 0.9
            
            # Проверяем совместимые сезоны
            for main_season, compatible_seasons in self.season_compatibilities.items():
                if season1 == main_season and season2 in compatible_seasons:
                    return 0.8
                if season2 == main_season and season1 in compatible_seasons:
                    return 0.8
            
            # Смежные сезоны имеют среднюю совместимость
            adjacent_seasons = {
                'зима': ['осень', 'весна'],
                'весна': ['зима', 'лето'],
                'лето': ['весна', 'осень'],
                'осень': ['лето', 'зима']
            }
            
            if season1 in adjacent_seasons.get(season2, []) or season2 in adjacent_seasons.get(season1, []):
                return 0.6
            
            # Универсальные сезоны
            if season1 in ['универсальный', 'всесезонный'] or season2 in ['универсальный', 'всесезонный']:
                return 0.7
            
            # Базовая совместимость
            return 0.3
            
        except Exception as e:
            print(f"⚠️ Ошибка анализа сезонов: {e}")
            return 0.5
    
    def analyze_outfit_compatibility(self, items: List[Dict]) -> Dict:
        """Анализ совместимости всего образа"""
        try:
            if len(items) < 2:
                return {'compatibility_score': 1.0, 'analysis': 'Недостаточно элементов для анализа'}
            
            compatibility_scores = []
            analysis_details = []
            
            # Анализируем все пары элементов
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    item1 = items[i]
                    item2 = items[j]
                    
                    # Анализ цветовой совместимости
                    color_score = self.check_color_compatibility(
                        item1.get('color', 'unknown'),
                        item2.get('color', 'unknown')
                    )
                    
                    # Анализ стилевой совместимости
                    style_score = self.check_style_compatibility(
                        item1.get('style', 'unknown'),
                        item2.get('style', 'unknown')
                    )
                    
                    # Анализ сезонной совместимости
                    season_score = self.check_season_compatibility(
                        item1.get('season', 'unknown'),
                        item2.get('season', 'unknown')
                    )
                    
                    # Общий скор совместимости пары
                    pair_score = (color_score * 0.4 + style_score * 0.4 + season_score * 0.2)
                    compatibility_scores.append(pair_score)
                    
                    analysis_details.append({
                        'items': [item1.get('garment_type', 'unknown'), item2.get('garment_type', 'unknown')],
                        'color_score': color_score,
                        'style_score': style_score,
                        'season_score': season_score,
                        'overall_score': pair_score
                    })
            
            # Общий скор совместимости образа
            overall_score = np.mean(compatibility_scores) if compatibility_scores else 0.5
            
            return {
                'compatibility_score': overall_score,
                'analysis': analysis_details,
                'recommendations': self._generate_compatibility_recommendations(overall_score, analysis_details)
            }
            
        except Exception as e:
            print(f"⚠️ Ошибка анализа совместимости: {e}")
            return {'compatibility_score': 0.5, 'analysis': 'Ошибка анализа'}
    
    def _generate_compatibility_recommendations(self, score: float, analysis: List[Dict]) -> List[str]:
        """Генерация рекомендаций по совместимости"""
        recommendations = []
        
        if score < 0.3:
            recommendations.append("❌ Низкая совместимость элементов")
            recommendations.append("💡 Попробуйте изменить цвета или стили")
        elif score < 0.6:
            recommendations.append("⚠️ Средняя совместимость элементов")
            recommendations.append("💡 Добавьте нейтральные цвета для баланса")
        elif score < 0.8:
            recommendations.append("✅ Хорошая совместимость элементов")
            recommendations.append("💡 Образ выглядит гармонично")
        else:
            recommendations.append("🎯 Отличная совместимость элементов")
            recommendations.append("💡 Идеально подобранный образ!")
        
        return recommendations
