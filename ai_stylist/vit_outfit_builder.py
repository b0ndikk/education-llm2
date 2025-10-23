#!/usr/bin/env python3
"""
🎯 VIT OUTFIT BUILDER - Сборка образов с помощью Vision Transformer
"""

import os
import sys
import math
import uuid
import json
from typing import List, Dict, Tuple, Optional
from PIL import Image
import numpy as np

# Проверяем наличие необходимых библиотек
try:
    import torch
    import torch.nn as nn
    import timm
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PyTorch/Timm не установлены: {e}")
    print("Установите зависимости: pip install torch timm")
    TORCH_AVAILABLE = False

try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Torchvision не установлен: {e}")
    print("Установите зависимости: pip install torchvision")
    TORCHVISION_AVAILABLE = False

class FashionViT:
    """Vision Transformer для анализа совместимости одежды"""
    
    def __init__(self, num_items: int = 10, embedding_dim: int = 768):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch не установлен. Установите: pip install torch timm")
        
        # Базовый ViT
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, embedding_dim)
        
        # Замораживаем предобученные слои
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Размораживаем только голову
        for param in self.vit.head.parameters():
            param.requires_grad = True
        
        # Transformer для анализа совместимости
        self.compatibility_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=12,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=6
        )
        
        # Выходные головы
        self.compatibility_head = nn.Linear(embedding_dim, 1)
        self.outfit_head = nn.Linear(embedding_dim, num_items)
        
        # Позиционное кодирование
        self.pos_encoding = self._create_positional_encoding(embedding_dim)
        
    def _create_positional_encoding(self, d_model: int, max_len: int = 100):
        """Создает позиционное кодирование"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, item_images: List[Image.Image], occasion: str = "casual") -> Tuple[torch.Tensor, torch.Tensor]:
        """Анализирует совместимость предметов одежды"""
        
        # Извлекаем эмбеддинги для каждого предмета
        item_embeddings = []
        for image in item_images:
            # Преобразуем PIL в тензор
            image_tensor = self._preprocess_image(image)
            
            # Получаем эмбеддинг от ViT
            with torch.no_grad():
                embedding = self.vit(image_tensor)  # [embedding_dim]
            
            item_embeddings.append(embedding)
        
        # Объединяем все эмбеддинги
        sequence = torch.stack(item_embeddings)  # [num_items, embedding_dim]
        
        # Добавляем позиционное кодирование
        seq_len = sequence.size(0)
        sequence = sequence + self.pos_encoding[:, :seq_len, :]
        
        # Пропускаем через Transformer
        transformer_output = self.compatibility_transformer(sequence)
        
        # Получаем оценки совместимости
        compatibility_scores = self.compatibility_head(transformer_output)
        outfit_scores = self.outfit_head(transformer_output)
        
        return compatibility_scores, outfit_scores
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Предобработка изображения для ViT"""
        if not TORCHVISION_AVAILABLE:
            print("❌ torchvision не установлен. Установите: pip install torchvision")
            return torch.zeros(1, 3, 224, 224)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)  # Добавляем batch dimension


class OccasionRules:
    """Правила для разных случаев"""
    
    def __init__(self):
        self.rules = {
            "свидание": {
                "colors": ["черный", "белый", "красный", "темно-синий", "бордовый", "розовый", "золотой", "серебряный"],
                "styles": ["элегантный", "романтичный", "классический", "женственный", "утонченный"],
                "garments": ["платье", "блузка", "юбка", "брюки", "туфли", "сумка", "украшения"],
                "avoid": ["спортивный", "повседневный", "яркий", "кричащий", "мешковатый"],
                "season": ["весна", "лето", "осень", "зима"],
                "temperature": ["комфортная", "прохладная"],
                "price_range": ["средняя", "премиум", "люкс"],
                "weight": 1.0
            },
            "спорт": {
                "colors": ["любые", "яркие", "контрастные"],
                "styles": ["спортивный", "активный", "функциональный", "технический"],
                "garments": ["футболка", "шорты", "спортивные штаны", "кроссовки", "спортивная сумка"],
                "avoid": ["формальный", "элегантный", "деловой"],
                "season": ["любой"],
                "temperature": ["любая"],
                "price_range": ["бюджетная", "средняя", "премиум"],
                "weight": 1.0
            },
            "прогулка": {
                "colors": ["любые", "нейтральные", "природные"],
                "styles": ["повседневный", "комфортный", "расслабленный", "casual"],
                "garments": ["футболка", "джинсы", "кроссовки", "куртка", "рюкзак"],
                "avoid": ["формальный", "слишком нарядный"],
                "season": ["любой"],
                "temperature": ["любая"],
                "price_range": ["бюджетная", "средняя"],
                "weight": 0.8
            },
            "работа": {
                "colors": ["черный", "белый", "серый", "темно-синий", "коричневый", "бежевый"],
                "styles": ["деловой", "формальный", "профессиональный", "строгий"],
                "garments": ["блузка", "брюки", "юбка", "туфли", "пиджак", "рубашка"],
                "avoid": ["спортивный", "яркий", "кричащий", "повседневный"],
                "season": ["любой"],
                "temperature": ["комфортная"],
                "price_range": ["средняя", "премиум"],
                "weight": 1.0
            },
            "путешествия": {
                "colors": ["любые", "нейтральные", "темные"],
                "styles": ["комфортный", "практичный", "функциональный", "универсальный"],
                "garments": ["футболка", "джинсы", "кроссовки", "куртка", "рюкзак", "сумка"],
                "avoid": ["формальный", "хрупкий", "неудобный"],
                "season": ["любой"],
                "temperature": ["любая"],
                "price_range": ["бюджетная", "средняя"],
                "weight": 0.9
            },
            "вечеринка": {
                "colors": ["черный", "белый", "яркие", "металлик", "блестящие"],
                "styles": ["вечерний", "элегантный", "гламурный", "стильный"],
                "garments": ["платье", "туфли", "сумка", "украшения", "аксессуары"],
                "avoid": ["повседневный", "спортивный", "скромный"],
                "season": ["любой"],
                "temperature": ["комфортная"],
                "price_range": ["средняя", "премиум", "люкс"],
                "weight": 1.0
            },
            "отпуск": {
                "colors": ["яркие", "тропические", "светлые", "пастельные"],
                "styles": ["расслабленный", "курортный", "тропический", "пляжный"],
                "garments": ["купальник", "шорты", "футболка", "сандали", "сумка"],
                "avoid": ["формальный", "зимний", "темный"],
                "season": ["лето", "весна"],
                "temperature": ["жаркая", "теплая"],
                "price_range": ["бюджетная", "средняя"],
                "weight": 0.9
            },
            "шопинг": {
                "colors": ["любые", "удобные для примерки"],
                "styles": ["повседневный", "комфортный", "удобный"],
                "garments": ["футболка", "джинсы", "кроссовки", "сумка"],
                "avoid": ["формальный", "неудобный"],
                "season": ["любой"],
                "temperature": ["любая"],
                "price_range": ["любая"],
                "weight": 0.7
            }
        }
    
    def get_rules(self, occasion: str) -> Dict:
        """Получает правила для конкретного случая"""
        return self.rules.get(occasion, self.rules["casual"])
    
    def score_compatibility(self, item_features: Dict, occasion: str) -> float:
        """Оценивает совместимость предмета с случаем"""
        rules = self.get_rules(occasion)
        score = 0.0
        
        # Проверяем стиль (40% веса)
        if item_features.get("style") in rules["styles"]:
            score += 0.4
        elif any(avoid in str(item_features.get("style", "")).lower() for avoid in rules["avoid"]):
            score -= 0.2
        
        # Проверяем цвет (25% веса)
        if item_features.get("color") in rules["colors"] or "любые" in rules["colors"]:
            score += 0.25
        elif item_features.get("color_pattern") and any(color in str(item_features.get("color_pattern", "")).lower() for color in rules["colors"]):
            score += 0.2
        
        # Проверяем тип одежды (20% веса)
        if item_features.get("garment_type") in rules["garments"]:
            score += 0.2
        elif item_features.get("garment_category") and any(garment in str(item_features.get("garment_category", "")).lower() for garment in rules["garments"]):
            score += 0.15
        
        # Проверяем сезон (10% веса)
        if item_features.get("season") in rules.get("season", ["любой"]):
            score += 0.1
        elif item_features.get("season_weather") and any(season in str(item_features.get("season_weather", "")).lower() for season in rules.get("season", ["любой"])):
            score += 0.05
        
        # Проверяем ценовую категорию (5% веса)
        if item_features.get("price_range") in rules.get("price_range", ["любая"]):
            score += 0.05
        
        # Применяем вес случая
        final_score = max(0.0, min(1.0, score * rules["weight"]))
        
        return final_score


class OutfitBuilder:
    """Сборщик образов из гардероба"""
    
    def __init__(self):
        self.vit_model = FashionViT()
        self.occasion_rules = OccasionRules()
        self.wardrobe = {}  # Хранилище гардероба пользователя
        
    def add_item_to_wardrobe(self, item_id: str, image: Image.Image, 
                           features: Dict = None) -> None:
        """Добавляет предмет в гардероб"""
        self.wardrobe[item_id] = {
            "image": image,
            "features": features or {},
            "embedding": None
        }
    
    def generate_outfit(self, occasion: str = "casual", 
                       max_items: int = 5) -> Dict:
        """Генерирует образ для указанного случая"""
        
        if len(self.wardrobe) < 2:
            return {
                "error": "Недостаточно предметов в гардеробе (минимум 2)",
                "outfit": [],
                "confidence": 0.0
            }
        
        # Получаем все предметы из гардероба
        items = list(self.wardrobe.values())
        item_images = [item["image"] for item in items]
        
        # Анализируем совместимость с помощью ViT
        compatibility_scores, outfit_scores = self.vit_model(item_images, occasion)
        
        # Применяем правила случая
        occasion_scores = []
        for i, item in enumerate(items):
            if item["features"]:
                score = self.occasion_rules.score_compatibility(item["features"], occasion)
                occasion_scores.append(score)
            else:
                occasion_scores.append(0.5)  # Нейтральная оценка
        
        # Объединяем оценки ViT и правил
        final_scores = []
        for i in range(len(items)):
            vit_score = torch.sigmoid(compatibility_scores[i]).item()
            occasion_score = occasion_scores[i]
            final_score = (vit_score * 0.7) + (occasion_score * 0.3)
            final_scores.append(final_score)
        
        # Выбираем лучшие предметы
        item_ids = list(self.wardrobe.keys())
        sorted_indices = sorted(range(len(final_scores)), 
                               key=lambda i: final_scores[i], reverse=True)
        
        selected_items = []
        for i in sorted_indices[:max_items]:
            if final_scores[i] > 0.3:  # Минимальный порог
                selected_items.append({
                    "item_id": item_ids[i],
                    "image": items[i]["image"],
                    "features": items[i]["features"],
                    "score": final_scores[i]
                })
        
        # Вычисляем общую уверенность
        if selected_items:
            confidence = np.mean([item["score"] for item in selected_items])
        else:
            confidence = 0.0
        
        return {
            "outfit": selected_items,
            "confidence": confidence,
            "occasion": occasion,
            "total_items": len(selected_items),
            "explanation": self._generate_explanation(selected_items, occasion)
        }
    
    def _generate_explanation(self, outfit: List[Dict], occasion: str) -> str:
        """Генерирует объяснение выбранного образа"""
        if not outfit:
            return "Не удалось подобрать подходящий образ"
        
        explanation_parts = [f"Образ для случая '{occasion}':"]
        
        for item in outfit:
            features = item["features"]
            if features:
                parts = []
                if features.get("garment_type"):
                    parts.append(features["garment_type"])
                if features.get("color"):
                    parts.append(features["color"])
                if features.get("style"):
                    parts.append(features["style"])
                
                if parts:
                    explanation_parts.append(f"• {' '.join(parts)} (уверенность: {item['score']:.1%})")
        
        return "\n".join(explanation_parts)
    
    def get_wardrobe_stats(self) -> Dict:
        """Получает статистику гардероба"""
        return {
            "total_items": len(self.wardrobe),
            "item_types": list(set(item["features"].get("garment_type", "unknown") 
                                 for item in self.wardrobe.values())),
            "colors": list(set(item["features"].get("color", "unknown") 
                            for item in self.wardrobe.values())),
            "styles": list(set(item["features"].get("style", "unknown") 
                             for item in self.wardrobe.values()))
        }


class ViTOutfitManager:
    """Менеджер для интеграции ViT с существующим ансамблем"""
    
    def __init__(self):
        self.outfit_builder = OutfitBuilder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Переносим модель на устройство
        self.outfit_builder.vit_model.to(self.device)
        
    def add_item_from_analysis(self, image: Image.Image, analysis: Dict) -> str:
        """Добавляет предмет в гардероб на основе анализа ансамбля"""
        import uuid
        
        # Извлекаем признаки из анализа
        features = {}
        
        # Основные категории из FashionCLIP
        if "fashion_clip" in analysis:
            clip = analysis["fashion_clip"]
            
            # Тип одежды
            if "garment_category" in clip:
                features["garment_type"] = clip["garment_category"]["best_match"]["item"]
                features["garment_category"] = clip["garment_category"]["best_match"]["item"]
            
            # Материал и ткань
            if "material_fabric" in clip:
                features["material"] = clip["material_fabric"]["best_match"]["item"]
                features["material_fabric"] = clip["material_fabric"]["best_match"]["item"]
            
            # Цвет и паттерн
            if "color_pattern" in clip:
                features["color"] = clip["color_pattern"]["best_match"]["item"]
                features["color_pattern"] = clip["color_pattern"]["best_match"]["item"]
            
            # Стиль и случай
            if "style_occasion" in clip:
                features["style"] = clip["style_occasion"]["best_match"]["item"]
                features["style_occasion"] = clip["style_occasion"]["best_match"]["item"]
            
            # Новые категории
            if "season_weather" in clip:
                features["season"] = clip["season_weather"]["best_match"]["item"]
                features["season_weather"] = clip["season_weather"]["best_match"]["item"]
            
            if "body_type_fit" in clip:
                features["fit"] = clip["body_type_fit"]["best_match"]["item"]
                features["body_type_fit"] = clip["body_type_fit"]["best_match"]["item"]
            
            if "age_group" in clip:
                features["age_group"] = clip["age_group"]["best_match"]["item"]
            
            if "price_range" in clip:
                features["price_range"] = clip["price_range"]["best_match"]["item"]
        
        # Интегрированный анализ
        if "integrated_analysis" in analysis:
            integrated = analysis["integrated_analysis"]
            
            if "primary_garment" in integrated:
                garment = integrated["primary_garment"]
                features.update({
                    "garment_type": garment.get("type", features.get("garment_type", "unknown")),
                    "style": garment.get("style", features.get("style", "unknown")),
                    "material": garment.get("material", features.get("material", "unknown")),
                    "color": garment.get("color", features.get("color", "unknown"))
                })
        
        # Генерируем уникальный ID
        item_id = str(uuid.uuid4())
        
        # Добавляем в гардероб
        self.outfit_builder.add_item_to_wardrobe(item_id, image, features)
        
        return item_id
    
    def generate_outfit_for_occasion(self, occasion: str) -> Dict:
        """Генерирует образ для указанного случая"""
        return self.outfit_builder.generate_outfit(occasion)
    
    def get_wardrobe_info(self) -> Dict:
        """Получает информацию о гардеробе"""
        return self.outfit_builder.get_wardrobe_stats()
    
    def clear_wardrobe(self) -> None:
        """Очищает гардероб"""
        self.outfit_builder.wardrobe.clear()


if __name__ == "__main__":
    print("🎯 ViT Outfit Builder - Готов к работе!")
    print("Запустите main.py для использования интерфейса")
