#!/usr/bin/env python3
"""
🎯 FASHIONCLIP МОДЕЛЬ - ОСНОВНОЙ КЛАССИФИКАТОР
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface')

from transformers import CLIPModel, CLIPProcessor

class FashionCLIPAnalyzer:
    def __init__(self):
        """FashionCLIP анализатор одежды"""
        print("🚀 Инициализация FashionCLIP...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загружаем FashionCLIP
        self.model = CLIPModel.from_pretrained('patrickjohncyh/fashion-clip').to(self.device)
        self.processor = CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')
        
        # База промтов
        self.categories = self._create_fashion_prompts()
        print("✅ FashionCLIP загружен!")
    
    def _create_fashion_prompts(self):
        """Создает промты для классификации"""
        return {
            'garment_category': [
                "winter coat", "trench coat", "peacoat", "parka jacket", "puffer jacket",
                "bomber jacket", "leather jacket", "denim jacket", "blazer", "cardigan",
                "hoodie", "sweater", "dress shirt", "casual shirt", "flannel shirt",
                "polo shirt", "t-shirt", "blouse", "silk blouse", "cocktail dress",
                "maxi dress", "mini dress", "a-line skirt", "pencil skirt", "dress pants",
                "chino pants", "cargo pants", "skinny jeans", "straight jeans", "leggings",
                "shorts", "swimsuit", "bikini", "kimono", "sari"
            ],
            'material_fabric': [
                "cotton fabric", "linen fabric", "silk fabric", "wool fabric", "denim fabric",
                "leather material", "suede material", "polyester fabric", "nylon fabric",
                "velvet fabric", "satin fabric", "chiffon fabric"
            ],
            'color_pattern': [
                "black color", "white color", "navy blue", "gray", "beige color",
                "brown color", "burgundy color", "green color", "purple color", "pink color",
                "red color", "blue color", "striped pattern", "floral print", "animal print",
                "geometric pattern", "solid color"
            ],
            'style_occasion': [
                "casual style", "formal style", "business style", "sporty style", "vintage style",
                "bohemian style", "streetwear style", "evening occasion", "office appropriate"
            ]
        }
    
    def analyze_image(self, image: Image.Image) -> dict:
        """Анализирует изображение одежды"""
        try:
            analysis = {}
            
            for category, options in self.categories.items():
                # Подготавливаем текстовые входы
                text_inputs = self.processor(text=options, return_tensors="pt", padding=True)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                
                # Подготавливаем изображение
                image_inputs = self.processor(images=image, return_tensors="pt")
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                
                with torch.no_grad():
                    # Получаем признаки
                    image_features = self.model.get_image_features(**image_inputs)
                    text_features = self.model.get_text_features(**text_inputs)
                    
                    # Нормализуем
                    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
                    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
                    
                    # Вычисляем сходство
                    similarities = torch.matmul(image_features, text_features.T)[0]
                    similarities = similarities / 2.0  # Температурная шкала
                    probabilities = torch.softmax(similarities, dim=0)
                
                # Находим лучшие совпадения
                top_indices = torch.topk(probabilities, 3).indices
                top_results = [
                    {'item': options[i], 'confidence': probabilities[i].item()}
                    for i in top_indices
                ]
                
                analysis[category] = {
                    'best_match': top_results[0],
                    'alternatives': top_results[1:]
                }
            
            return analysis
            
        except Exception as e:
            return {"error": f"FashionCLIP ошибка: {str(e)}"}