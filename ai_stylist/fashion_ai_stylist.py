#!/usr/bin/env python3
"""
🎯 AI-СТИЛИСТ НА ОСНОВЕ FASHIONCLIP
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional
import gradio as gr
from datetime import datetime

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fashion_clip import FashionCLIP
    FASHION_CLIP_AVAILABLE = True
except ImportError:
    print("⚠️ FashionCLIP не установлен. Устанавливаем...")
    os.system("pip install fashion-clip")
    try:
        from fashion_clip import FashionCLIP
        FASHION_CLIP_AVAILABLE = True
    except ImportError:
        FASHION_CLIP_AVAILABLE = False

class FashionAIStylist:
    def __init__(self):
        """Инициализация AI-стилиста"""
        print("🚀 Инициализация AI-стилиста...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Устройство: {self.device}")
        
        # Загружаем FashionCLIP
        if FASHION_CLIP_AVAILABLE:
            try:
                self.fclip = FashionCLIP('fashion-clip')
                print("✅ FashionCLIP загружен успешно!")
            except Exception as e:
                print(f"❌ Ошибка загрузки FashionCLIP: {e}")
                self.fclip = None
        else:
            print("❌ FashionCLIP недоступен")
            self.fclip = None
        
        # Категории для анализа
        self.categories = {
            'garment_type': [
                "t-shirt", "shirt", "blouse", "dress", "skirt", "pants", 
                "jeans", "shorts", "jacket", "coat", "blazer", "sweater", 
                "hoodie", "cardigan", "tank top", "polo shirt", "jumpsuit", 
                "romper", "vest", "windbreaker", "parka", "sneakers", "boots", 
                "heels", "sandals", "flats", "loafers", "oxfords"
            ],
            'color': [
                "black", "white", "gray", "beige", "brown", "navy", "blue", 
                "light blue", "sky blue", "red", "burgundy", "maroon", "pink", 
                "rose", "green", "forest green", "olive", "emerald", "yellow", 
                "gold", "mustard", "cream", "purple", "violet", "lavender", 
                "orange", "coral", "peach", "apricot"
            ],
            'style': [
                "casual", "formal", "business", "sporty", "athletic", "vintage", 
                "retro", "modern", "contemporary", "minimalist", "bohemian", 
                "street", "urban", "hip-hop", "preppy", "classic", "elegant", 
                "sophisticated", "romantic", "edgy", "punk", "gothic", "chic"
            ],
            'material': [
                "cotton", "denim", "leather", "silk", "wool", "cashmere", 
                "polyester", "linen", "suede", "canvas", "mesh", "knit", 
                "fleece", "satin", "velvet", "lace", "chiffon", "organza"
            ],
            'occasion': [
                "casual", "work", "office", "business", "formal", "evening", 
                "party", "cocktail", "wedding", "date", "gym", "sport", 
                "beach", "vacation", "travel", "weekend", "everyday"
            ],
            'brand_style': [
                "luxury", "designer", "high-end", "premium", "fast fashion", 
                "affordable", "budget", "mass market", "streetwear", "urban", 
                "hip-hop", "vintage", "retro", "classic", "traditional", 
                "sportswear", "athletic", "fitness"
            ]
        }
        
        # Стилистические правила
        self.style_rules = self._load_style_rules()
        
        print("✅ AI-стилист инициализирован!")
    
    def _load_style_rules(self) -> Dict:
        """Загружает правила стилистики"""
        return {
            'color_harmony': {
                'complementary': ['red-green', 'blue-orange', 'yellow-purple'],
                'analogous': ['red-orange-yellow', 'blue-green-teal', 'purple-pink-red'],
                'triadic': ['red-yellow-blue', 'orange-green-purple'],
                'monochromatic': ['black-gray-white', 'navy-blue-sky']
            },
            'style_matching': {
                'formal': ['business', 'elegant', 'sophisticated'],
                'casual': ['everyday', 'weekend', 'relaxed'],
                'sporty': ['athletic', 'gym', 'active'],
                'vintage': ['retro', 'classic', 'traditional']
            },
            'occasion_rules': {
                'work': ['formal', 'business', 'professional'],
                'party': ['evening', 'cocktail', 'dressy'],
                'gym': ['sporty', 'athletic', 'active'],
                'date': ['romantic', 'elegant', 'chic']
            }
        }
    
    def analyze_outfit(self, image: Image.Image) -> Dict:
        """Анализирует образ"""
        if not self.fclip:
            return {"error": "FashionCLIP не загружен"}
        
        try:
            # Анализируем каждую категорию
            analysis = {}
            
            for category, options in self.categories.items():
                # Получаем эмбеддинги изображения
                image_embedding = self.fclip.encode_image(image)
                
                # Получаем эмбеддинги текста
                text_embeddings = self.fclip.encode_text(options)
                
                # Вычисляем сходство
                similarities = torch.cosine_similarity(
                    image_embedding, text_embeddings, dim=1
                )
                
                # Находим лучшие совпадения
                top_indices = torch.topk(similarities, 3).indices
                top_results = [
                    {
                        'item': options[i],
                        'confidence': similarities[i].item()
                    }
                    for i in top_indices
                ]
                
                analysis[category] = {
                    'best_match': top_results[0],
                    'alternatives': top_results[1:],
                    'all_scores': {options[i]: similarities[i].item() for i in top_indices}
                }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Ошибка анализа: {str(e)}"}
    
    def get_styling_recommendations(self, analysis: Dict) -> List[str]:
        """Получает рекомендации по стилю"""
        if "error" in analysis:
            return [f"❌ {analysis['error']}"]
        
        recommendations = []
        
        # Анализируем цвет
        color = analysis['color']['best_match']['item']
        style = analysis['style']['best_match']['item']
        occasion = analysis['occasion']['best_match']['item']
        
        # Рекомендации по цвету
        if color in ['black', 'white', 'gray']:
            recommendations.append(f"✅ {color.title()} - универсальный цвет, подходит ко всему")
        elif color in ['red', 'blue', 'green']:
            recommendations.append(f"🎨 {color.title()} - яркий цвет, сочетайте с нейтральными тонами")
        
        # Рекомендации по стилю
        if style == 'casual':
            recommendations.append("👕 Кэжуал стиль - добавьте джинсы или кроссовки")
        elif style == 'formal':
            recommendations.append("👔 Формальный стиль - подходит для работы и важных встреч")
        elif style == 'sporty':
            recommendations.append("🏃 Спортивный стиль - идеален для активного отдыха")
        
        # Рекомендации по поводу
        if occasion == 'work':
            recommendations.append("💼 Для работы - выберите классические цвета и силуэты")
        elif occasion == 'party':
            recommendations.append("🎉 Для вечеринки - добавьте аксессуары и яркие детали")
        
        return recommendations
    
    def find_similar_items(self, analysis: Dict, category: str = 'garment_type') -> List[str]:
        """Находит похожие предметы"""
        if "error" in analysis or category not in analysis:
            return []
        
        alternatives = analysis[category]['alternatives']
        return [item['item'] for item in alternatives]
    
    def create_outfit_combination(self, base_item: str) -> Dict:
        """Создает комбинацию образа"""
        combinations = {
            't-shirt': ['jeans', 'shorts', 'skirt', 'blazer'],
            'dress': ['jacket', 'cardigan', 'heels', 'flats'],
            'jeans': ['t-shirt', 'shirt', 'sweater', 'sneakers'],
            'blazer': ['shirt', 'blouse', 'pants', 'heels'],
            'sneakers': ['jeans', 'shorts', 't-shirt', 'hoodie']
        }
        
        return {
            'base_item': base_item,
            'suggested_combinations': combinations.get(base_item, []),
            'style_tips': self._get_style_tips(base_item)
        }
    
    def _get_style_tips(self, item: str) -> List[str]:
        """Получает советы по стилю для предмета"""
        tips = {
            't-shirt': [
                "Заправьте в джинсы для более структурированного вида",
                "Добавьте аксессуары для завершения образа"
            ],
            'dress': [
                "Подберите подходящую обувь по случаю",
                "Добавьте пояс для подчеркивания талии"
            ],
            'jeans': [
                "Выберите правильную посадку для вашей фигуры",
                "Сочетайте с разными топами для разнообразия"
            ]
        }
        
        return tips.get(item, ["Экспериментируйте с аксессуарами"])

def create_gradio_interface():
    """Создает Gradio интерфейс"""
    
    # Создаем стилиста
    stylist = FashionAIStylist()
    
    def analyze_and_recommend(image):
        if image is None:
            return "❌ Пожалуйста, загрузите изображение одежды", []
        
        # Анализируем образ
        analysis = stylist.analyze_outfit(image)
        
        if "error" in analysis:
            return f"❌ {analysis['error']}", []
        
        # Форматируем результаты
        result_text = "🎯 **АНАЛИЗ ОБРАЗА**\n\n"
        
        for category, data in analysis.items():
            best = data['best_match']
            result_text += f"**{category.replace('_', ' ').title()}:** {best['item']} ({best['confidence']:.1%})\n"
        
        # Получаем рекомендации
        recommendations = stylist.get_styling_recommendations(analysis)
        
        return result_text, recommendations
    
    # Создаем интерфейс
    with gr.Blocks(title="AI-Стилист") as interface:
        gr.Markdown("# 🎯 AI-Стилист на основе FashionCLIP")
        gr.Markdown("Загрузите изображение одежды для анализа и получения стилистических рекомендаций")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Загрузите изображение одежды")
                analyze_btn = gr.Button("🔍 Анализировать образ", variant="primary")
            
            with gr.Column():
                analysis_output = gr.Textbox(label="Результат анализа", lines=10)
                recommendations_output = gr.Textbox(label="Рекомендации", lines=5)
        
        # Обработчик
        analyze_btn.click(
            fn=analyze_and_recommend,
            inputs=[image_input],
            outputs=[analysis_output, recommendations_output]
        )
        
        # Примеры
        gr.Examples(
            examples=[
                ["data/fashion_mnist/train/0/00000.png"],
                ["data/fashion_mnist/train/1/00000.png"],
                ["data/fashion_mnist/train/2/00000.png"]
            ],
            inputs=[image_input]
        )
    
    return interface

def main():
    """Главная функция"""
    print("🎯 AI-СТИЛИСТ НА ОСНОВЕ FASHIONCLIP")
    print("=" * 50)
    
    # Создаем интерфейс
    interface = create_gradio_interface()
    
    # Запускаем
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

if __name__ == "__main__":
    main()
