#!/usr/bin/env python3
"""
🎯 ULTIMATE AI-СТИЛИСТ - МИРОВОЙ УРОВЕНЬ ОПИСАНИЯ ОДЕЖДЫ
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import gradio as gr
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import cv2
from collections import Counter
import re

# Устанавливаем переменные окружения
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface')

from transformers import CLIPModel, CLIPProcessor

class GlobalFashionAIStylist:
    def __init__(self):
        """AI-стилист мирового уровня для описания ВСЕЙ одежды"""
        print("🚀 Инициализация GLOBAL AI-стилиста...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Устройство: {self.device}")
        
        # Загружаем FashionCLIP
        print("📥 Загружаем FashionCLIP...")
        self.model = CLIPModel.from_pretrained('patrickjohncyh/fashion-clip').to(self.device)
        self.processor = CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')
        print("✅ FashionCLIP загружен!")
        
        # МИРОВАЯ БАЗА ЗНАНИЙ ОБ ОДЕЖДЕ - СТРУКТУРИРОВАННЫЕ ПРОМТЫ
        self.categories = self._create_global_fashion_prompts()
        
        print("✅ GLOBAL AI-стилист готов к работе!")
    
    def _create_global_fashion_prompts(self) -> Dict[str, List[str]]:
        """Создает МИРОВУЮ базу промтов для всей одежды"""
        
        return {
            # ОСНОВНЫЕ КАТЕГОРИИ ОДЕЖДЫ - СТРУКТУРИРОВАННЫЙ ПОДХОД
            'garment_category': [
                # ВЕРХНЯЯ ОДЕЖДА
                "winter coat", "trench coat", "peacoat", "duffle coat", "parka jacket", 
                "puffer jacket", "bomber jacket", "leather jacket", "denim jacket", "blazer",
                "cardigan", "hoodie", "sweater", "pullover", "windbreaker",
                
                # РУБАШКИ И БЛУЗКИ
                "dress shirt", "casual shirt", "flannel shirt", "hawaiian shirt", "polo shirt",
                "t-shirt", "blouse", "silk blouse", "chiffon blouse", "button-down shirt",
                
                # ПЛАТЬЯ И ЮБКИ
                "cocktail dress", "maxi dress", "mini dress", "shift dress", "bodycon dress",
                "a-line skirt", "pencil skirt", "pleated skirt", "wrap skirt", "midi skirt",
                
                # БРЮКИ И ШОРТЫ
                "dress pants", "chino pants", "cargo pants", "wide-leg pants", "skinny jeans",
                "straight jeans", "bootcut jeans", "leggings", "joggers", "bermuda shorts",
                
                # НИЖНЕЕ БЕЛЬЕ
                "bra", "panties", "lingerie set", "bodysuit", "camisole",
                "boxer shorts", "briefs", "long underwear", "shapewear",
                
                # СПОРТИВНАЯ ОДЕЖДА
                "tracksuit", "yoga pants", "gym shorts", "sports bra", "swimsuit",
                "bikini", "trunks", "rash guard", "cycling shorts",
                
                # ТРАДИЦИОННАЯ ОДЕЖДА
                "kimono", "sari", "hanbok", "cheongsam", "ao dai",
                "kilt", "dirndl", "poncho", "sarong", "thobe"
            ],
            
            'garment_details': [
                # ДЕТАЛИ ПОКРОЯ И СИЛУЭТА
                "fitted silhouette", "loose fit", "oversized cut", "tailored fit", "relaxed fit",
                "asymmetric hem", "high waist", "low rise", "empire waist", "drop shoulder",
                
                # ТИПЫ РУКАВОВ
                "long sleeves", "short sleeves", "sleeveless", "three-quarter sleeves", 
                "raglan sleeves", "puffed sleeves", "bell sleeves", "batwing sleeves",
                
                # ВОРОТНИКИ И ГОРЛОВИНА
                "round neckline", "v-neckline", "crew neck", "turtle neck", "boat neck",
                "scoop neck", "square neck", "off-shoulder", "halter neck", "cowl neck",
                
                # ЗАСТЕЖКИ И ЗАМКИ
                "button front", "zipper closure", "pullover style", "wrap style", "tie closure",
                "hook and eye", "snap buttons", "frog closures", "elastic waist", "drawstring",
                
                # КАРМАНЫ
                "patch pockets", "slit pockets", "flap pockets", "zippered pockets", "no pockets",
                "chest pocket", "side pockets", "back pockets", "cargo pockets",
                
                # ДЕКОРАТИВНЫЕ ЭЛЕМЕНТЫ
                "embroidery details", "lace trim", "beaded work", "sequin accents", "print pattern",
                "contrast stitching", "fringe details", "ruffle trim", "pleated details", "quilted pattern"
            ],
            
            'material_fabric': [
                # НАТУРАЛЬНЫЕ ТКАНИ
                "cotton fabric", "linen fabric", "silk fabric", "wool fabric", "cashmere fabric",
                "denim fabric", "leather material", "suede material", "fur material", "felt fabric",
                
                # СИНТЕТИЧЕСКИЕ ТКАНИ
                "polyester fabric", "nylon fabric", "rayon fabric", "spandex fabric", "acrylic fabric",
                "velvet fabric", "satin fabric", "chiffon fabric", "organza fabric", "tulle fabric",
                
                # ТЕХНИЧЕСКИЕ ТКАНИ
                "waterproof fabric", "breathable fabric", "stretch fabric", "technical fabric", 
                "performance fabric", "moisture-wicking", "thermal insulation", "ripstop fabric",
                
                # СМЕСОВЫЕ ТКАНИ
                "cotton-polyester blend", "wool-synthetic blend", "linen-cotton blend", "silk-wool blend"
            ],
            
            'color_pattern': [
                # БАЗОВЫЕ ЦВЕТА
                "black color", "white color", "navy blue", "charcoal gray", "beige color",
                "brown color", "burgundy color", "forest green", "purple color", "pink color",
                
                # ПАТТЕРНЫ И ПРИНТЫ
                "solid color", "striped pattern", "floral print", "animal print", "geometric pattern",
                "paisley pattern", "polka dot", "checkered pattern", "houndstooth", "plaid pattern",
                "camouflage print", "tie-dye pattern", "abstract print", "ethnic pattern",
                
                # ЦВЕТОВЫЕ КОМБИНАЦИИ
                "color block", "ombre effect", "gradient colors", "contrast trim", "monochromatic"
            ],
            
            'style_occasion': [
                # СТИЛИ
                "casual style", "formal style", "business style", "sporty style", "vintage style",
                "bohemian style", "minimalist style", "streetwear style", "preppy style", "romantic style",
                
                # ПОВОДЫ
                "everyday wear", "office appropriate", "evening occasion", "wedding guest", 
                "cocktail party", "business meeting", "date night", "vacation wear",
                "beach appropriate", "winter season", "summer season"
            ],
            
            'brand_origin': [
                # МИРОВЫЕ БРЕНДЫ
                "luxury designer brand", "premium fashion brand", "fast fashion brand", 
                "sportswear brand", "streetwear brand", "vintage clothing", 
                "local designer", "independent brand", "mass market brand"
            ]
        }
    
    def _create_dynamic_prompts(self, detected_category: str) -> List[str]:
        """Создает динамические промты на основе обнаруженной категории"""
        
        dynamic_prompts = []
        
        # СПЕЦИАЛИЗИРОВАННЫЕ ПРОМТЫ ДЛЯ КАЖДОЙ КАТЕГОРИИ
        category_specific = {
            'dress': [
                "a-line dress with {details} in {color}",
                "{length} dress made of {material} for {occasion}",
                "{style} dress with {sleeve_type} sleeves and {neckline}",
                "women's {type} dress featuring {pattern} pattern"
            ],
            'shirt': [
                "{sleeve_length} shirt with {collar_type} collar in {color}",
                "{style} shirt made from {material} with {details}",
                "{fit_type} fit shirt with {pattern} for {occasion}"
            ],
            'pants': [
                "{fit_type} {style} pants in {color} {material}",
                "{length} pants with {waist_type} waist and {details}",
                "{occasion} appropriate trousers with {pattern}"
            ],
            'shoes': [
                "{type} shoes in {color} {material} for {occasion}",
                "{style} footwear with {heel_type} and {details}",
                "{brand} {category} shoes with {closure_type}"
            ],
            'jacket': [
                "{type} jacket in {color} {material} with {details}",
                "{style} outerwear with {lining_type} lining for {season}",
                "{length} coat with {closure_type} and {pocket_type} pockets"
            ]
        }
        
        # Добавляем общие промты
        general_prompts = [
            "fashion photography of {category} in {color} with {details}",
            "product shot of {material} {category} for {occasion}",
            "{style} clothing item: {category} with {pattern}",
            "garment details: {category} featuring {details}"
        ]
        
        # Выбираем соответствующие промты
        for cat, prompts in category_specific.items():
            if cat in detected_category.lower():
                dynamic_prompts.extend(prompts)
        
        dynamic_prompts.extend(general_prompts)
        return dynamic_prompts
    
    def analyze_garment_structure(self, image: Image.Image) -> Dict:
        """Анализирует структуру одежды для лучшего описания"""
        try:
            # Конвертируем в numpy для OpenCV
            img_array = np.array(image)
            
            analysis = {
                'dominant_colors': self._extract_dominant_colors(img_array),
                'contour_shape': self._analyze_garment_shape(img_array),
                'texture_complexity': self._analyze_texture_complexity(img_array),
                'symmetry_score': self._analyze_symmetry(img_array)
            }
            
            return analysis
        except Exception as e:
            return {'error': f'Structure analysis failed: {str(e)}'}
    
    def _extract_dominant_colors(self, img_array: np.ndarray) -> List[str]:
        """Извлекает доминирующие цвета из изображения"""
        try:
            # Уменьшаем размер для быстрого анализа
            img_small = cv2.resize(img_array, (100, 100))
            pixels = img_small.reshape(-1, 3)
            
            # K-means для поиска доминирующих цветов
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixels.astype(np.float32), 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Преобразуем в названия цветов
            color_names = []
            for color in centers:
                color_names.append(self._rgb_to_color_name(color))
            
            return color_names
        except:
            return ['unknown color']
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Преобразует RGB в название цвета"""
        color_map = {
            'black': [0, 0, 0],
            'white': [255, 255, 255],
            'red': [255, 0, 0],
            'blue': [0, 0, 255],
            'green': [0, 255, 0],
            'yellow': [255, 255, 0],
            'purple': [128, 0, 128],
            'pink': [255, 192, 203],
            'brown': [165, 42, 42],
            'gray': [128, 128, 128],
            'navy': [0, 0, 128],
            'beige': [245, 245, 220]
        }
        
        min_distance = float('inf')
        closest_color = 'unknown'
        
        for name, ref_rgb in color_map.items():
            distance = np.linalg.norm(rgb - ref_rgb)
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        
        return closest_color
    
    def _analyze_garment_shape(self, img_array: np.ndarray) -> str:
        """Анализирует форму одежды"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Анализ контуров
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return "unknown shape"
            
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            aspect_ratio = w / h
            
            if aspect_ratio > 1.5:
                return "elongated silhouette"
            elif aspect_ratio < 0.7:
                return "vertical silhouette"
            else:
                return "balanced silhouette"
                
        except:
            return "standard silhouette"
    
    def _analyze_texture_complexity(self, img_array: np.ndarray) -> str:
        """Анализирует сложность текстуры"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Вычисляем вариацию Лапласиана как меру текстуры
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < 100:
                return "smooth texture"
            elif laplacian_var < 500:
                return "medium texture"
            else:
                return "complex texture"
        except:
            return "standard texture"
    
    def _analyze_symmetry(self, img_array: np.ndarray) -> float:
        """Анализирует симметрию изображения"""
        try:
            height, width = img_array.shape[:2]
            mid = width // 2
            
            left_half = img_array[:, :mid]
            right_half = img_array[:, mid:]
            
            # Зеркально отражаем правую половину для сравнения
            right_flipped = cv2.flip(right_half, 1)
            
            # Выравниваем размеры
            min_height = min(left_half.shape[0], right_flipped.shape[0])
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            
            left_cropped = left_half[:min_height, :min_width]
            right_cropped = right_flipped[:min_height, :min_width]
            
            # Сравниваем гистограммы
            correlation = cv2.compareHist(
                cv2.calcHist([left_cropped], [0], None, [256], [0, 256]),
                cv2.calcHist([right_cropped], [0], None, [256], [0, 256]),
                cv2.HISTCMP_CORREL
            )
            
            return float(correlation)
        except:
            return 0.5
    
    def analyze_outfit(self, image: Image.Image) -> Dict:
        """Максимально точный анализ одежды с мировым охватом"""
        try:
            # Первичный анализ структуры
            structure_analysis = self.analyze_garment_structure(image)
            
            # Основной анализ через FashionCLIP
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
                    
                    # Вычисляем сходство с температурной шкалой
                    similarities = torch.matmul(image_features, text_features.T)[0]
                    temperature = 2.0
                    similarities = similarities / temperature
                    
                    # Softmax для вероятностей
                    probabilities = torch.softmax(similarities, dim=0)
                
                # Находим лучшие совпадения
                top_indices = torch.topk(probabilities, 3).indices
                top_results = [
                    {
                        'item': options[i],
                        'confidence': probabilities[i].item()
                    }
                    for i in top_indices
                ]
                
                analysis[category] = {
                    'best_match': top_results[0],
                    'alternatives': top_results[1:],
                    'all_scores': {options[i]: probabilities[i].item() for i in top_indices}
                }
            
            # Добавляем структурный анализ
            analysis['structure'] = structure_analysis
            
            return analysis
            
        except Exception as e:
            return {"error": f"Ошибка анализа: {str(e)}"}
    
    def generate_comprehensive_description(self, analysis: Dict) -> str:
        """Генерирует комплексное описание одежды"""
        if "error" in analysis:
            return f"❌ {analysis['error']}"
        
        try:
            # Извлекаем ключевые характеристики
            category = analysis['garment_category']['best_match']['item']
            details = analysis['garment_details']['best_match']['item']
            material = analysis['material_fabric']['best_match']['item']
            color = analysis['color_pattern']['best_match']['item']
            style = analysis['style_occasion']['best_match']['item']
            
            # Строим описание
            description_parts = []
            
            # Основное описание
            description_parts.append(f"**{category.upper()}**")
            
            # Детали
            description_parts.append(f"• **Материал:** {material}")
            description_parts.append(f"• **Цвет/Принт:** {color}")
            description_parts.append(f"• **Стиль:** {style}")
            description_parts.append(f"• **Детали:** {details}")
            
            # Добавляем структурные наблюдения
            if 'structure' in analysis:
                structure = analysis['structure']
                if 'dominant_colors' in structure:
                    colors = [c for c in structure['dominant_colors'] if c != 'unknown color']
                    if colors:
                        description_parts.append(f"• **Доминирующие цвета:** {', '.join(colors[:2])}")
                
                if 'contour_shape' in structure:
                    description_parts.append(f"• **Силуэт:** {structure['contour_shape']}")
                
                if 'texture_complexity' in structure:
                    description_parts.append(f"• **Текстура:** {structure['texture_complexity']}")
            
            return "\n".join(description_parts)
            
        except Exception as e:
            return f"❌ Ошибка генерации описания: {str(e)}"
    
    def get_global_recommendations(self, analysis: Dict) -> List[str]:
        """Получает рекомендации мирового уровня"""
        if "error" in analysis:
            return [f"❌ {analysis['error']}"]
        
        recommendations = []
        
        try:
            category = analysis['garment_category']['best_match']['item']
            style = analysis['style_occasion']['best_match']['item']
            color = analysis['color_pattern']['best_match']['item']
            material = analysis['material_fabric']['best_match']['item']
            
            # УНИВЕРСАЛЬНЫЕ РЕКОМЕНДАЦИИ
            recommendations.append("🌍 **МЕЖДУНАРОДНЫЕ СТИЛЕВЫЕ РЕКОМЕНДАЦИИ:**")
            
            # Рекомендации по категориям
            if any(word in category.lower() for word in ['dress', 'skirt']):
                recommendations.append("👗 **Для платьев и юбок:**")
                recommendations.append("   • Сочетайте с классическими туфлями или модными кроссовками")
                recommendations.append("   • Добавьте аксессуары: пояс, сумку, бижутерию")
                recommendations.append("   • Для офиса - добавьте блейзер")
                
            elif any(word in category.lower() for word in ['shirt', 'blouse', 'top']):
                recommendations.append("👕 **Для верхов:**")
                recommendations.append("   • Заправляйте в брюки/юбку для элегантного вида")
                recommendations.append("   • Носите навыпуск для кэжуал стиля")
                recommendations.append("   • Комбинируйте с джинсами, брюками, юбками")
                
            elif any(word in category.lower() for word in ['pants', 'jeans']):
                recommendations.append("👖 **Для брюк:**")
                recommendations.append("   • Выбирайте обувь по стилю: кеды/кроссовки/туфли")
                recommendations.append("   • Сочетайте с разными верхами для разнообразия")
                recommendations.append("   • Учитывайте длину - может потребоваться подгонка")
                
            elif any(word in category.lower() for word in ['jacket', 'coat', 'blazer']):
                recommendations.append("🧥 **Для верхней одежды:**")
                recommendations.append("   • Носите поверх базовых вещей")
                recommendations.append("   • Подходит для наслоения (layering)")
                recommendations.append("   • Выбирайте по сезону и погоде")
            
            # Рекомендации по стилю
            if 'casual' in style.lower():
                recommendations.append("🎯 **Кэжуал стиль:**")
                recommendations.append("   • Идеален для повседневной носки")
                recommendations.append("   • Сочетайте с джинсами и удобной обувью")
                recommendations.append("   • Подходит для встреч с друзьями, прогулок")
                
            elif 'formal' in style.lower() or 'business' in style.lower():
                recommendations.append("💼 **Формальный/Деловой стиль:**")
                recommendations.append("   • Подходит для офиса и деловых встреч")
                recommendations.append("   • Сочетайте с классической обувью")
                recommendations.append("   • Добавьте минималистичные аксессуары")
                
            elif 'sport' in style.lower():
                recommendations.append("🏃 **Спортивный стиль:**")
                recommendations.append("   • Для активного образа жизни")
                recommendations.append("   • Сочетайте со спортивной обувью")
                recommendations.append("   • Подходит для тренировок и отдыха")
            
            # Сезонные рекомендации
            if any(word in material.lower() for word in ['wool', 'cashmere', 'fur']):
                recommendations.append("❄️ **Зимний сезон:**")
                recommendations.append("   • Теплые материалы для холодной погоды")
                recommendations.append("   • Сочетайте с зимними аксессуарами")
                
            elif any(word in material.lower() for word in ['linen', 'cotton', 'light']):
                recommendations.append("☀️ **Летний сезон:**")
                recommendations.append("   • Легкие дышащие материалы")
                recommendations.append("   • Идеальны для теплой погоды")
            
            # Уход за одеждой
            recommendations.append("🧼 **РЕКОМЕНДАЦИИ ПО УХОДУ:**")
            if 'leather' in material.lower() or 'suede' in material.lower():
                recommendations.append("   • Профессиональная химчистка для кожи/замши")
                recommendations.append("   • Защита от влаги и прямого солнца")
            elif 'silk' in material.lower() or 'wool' in material.lower():
                recommendations.append("   • Деликатная стирка или химчистка")
                recommendations.append("   • Сушить в расправленном виде")
            else:
                recommendations.append("   • Следуйте инструкциям на ярлыке")
                recommendations.append("   • Стандартная стирка при средней температуре")
            
        except Exception as e:
            recommendations.append(f"❌ Ошибка рекомендаций: {str(e)}")
        
        return recommendations

def create_global_fashion_interface():
    """Создает интерфейс мирового уровня"""
    
    stylist = GlobalFashionAIStylist()
    
    def analyze_global_outfit(image):
        if image is None:
            return "❌ Пожалуйста, загрузите изображение одежды", [], []
        
        # Анализируем образ
        analysis = stylist.analyze_outfit(image)
        
        if "error" in analysis:
            return f"❌ {analysis['error']}", [], []
        
        # Генерируем описание
        description = stylist.generate_comprehensive_description(analysis)
        
        # Форматируем детальный анализ
        analysis_text = "🔍 **ДЕТАЛЬНЫЙ АНАЛИЗ:**\n\n"
        for category, data in analysis.items():
            if category != 'structure':
                best = data['best_match']
                analysis_text += f"**{category.replace('_', ' ').title()}:** {best['item']} ({best['confidence']:.1%})\n"
        
        # Добавляем структурный анализ
        if 'structure' in analysis:
            structure = analysis['structure']
            analysis_text += "\n**📐 СТРУКТУРНЫЙ АНАЛИЗ:**\n"
            for key, value in structure.items():
                if key != 'error':
                    analysis_text += f"• {key.replace('_', ' ').title()}: {value}\n"
        
        # Получаем рекомендации
        recommendations = stylist.get_global_recommendations(analysis)
        
        return description, analysis_text, recommendations
    
    # Создаем интерфейс
    with gr.Blocks(title="🌍 GLOBAL AI-Стилист") as interface:
        gr.Markdown("# 🌍 GLOBAL AI-СТИЛИСТ")
        gr.Markdown("**МИРОВОЙ УРОВЕНЬ:** Описывает ВСЮ одежду мира с детальным анализом")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="📷 Загрузите изображение одежды")
                analyze_btn = gr.Button("🌍 АНАЛИЗИРОВАТЬ ОДЕЖДУ", variant="primary")
            
            with gr.Column():
                description_output = gr.Textbox(label="🧬 ОПИСАНИЕ ОДЕЖДЫ", lines=8)
                analysis_output = gr.Textbox(label="🔍 ДЕТАЛЬНЫЙ АНАЛИЗ", lines=10)
                recommendations_output = gr.Textbox(label="💡 МИРОВЫЕ РЕКОМЕНДАЦИИ", lines=12)
        
        # Обработчик
        analyze_btn.click(
            fn=analyze_global_outfit,
            inputs=[image_input],
            outputs=[description_output, analysis_output, recommendations_output]
        )
        
        gr.Markdown("""
        ### 🌟 ВОЗМОЖНОСТИ СИСТЕМЫ:
        - **Мировая база знаний** о всех типах одежды
        - **Структурный анализ** силуэта и текстуры
        - **Цветовой анализ** с определением доминирующих цветов
        - **Рекомендации по уходу** и стилизации
        - **Поддержка традиционной** и современной одежды
        """)
    
    return interface

def main():
    """Главная функция"""
    print("🌍 GLOBAL AI-СТИЛИСТ - МИРОВОЙ УРОВЕНЬ")
    print("=" * 70)
    print("✨ МИРОВЫЕ ВОЗМОЖНОСТИ:")
    print("• Описание ВСЕЙ одежды мира")
    print("• Структурный анализ силуэта и текстуры")
    print("• Цветовой анализ и определение паттернов")
    print("• Рекомендации по уходу и стилизации")
    print("• Поддержка традиционной и этнической одежды")
    print("=" * 70)
    
    # Создаем интерфейс
    interface = create_global_fashion_interface()
    
    # Запускаем
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

if __name__ == "__main__":
    main()