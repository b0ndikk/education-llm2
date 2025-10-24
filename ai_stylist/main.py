#!/usr/bin/env python3
"""
🎯 ГЛАВНЫЙ ФАЙЛ С ИНТЕРФЕЙСОМ - ОБНОВЛЕННАЯ ВЕРСИЯ
"""

import gradio as gr
from ensemble_manager import FashionEnsemble
from vit_outfit_builder import ViTOutfitManager
from PIL import Image
import json

class FashionAIStylist:
    def __init__(self):
        self.ensemble = FashionEnsemble()
        self.vit_manager = ViTOutfitManager()
    
    def analyze_outfit(self, image):
        if image is None:
            return "❌ Пожалуйста, загрузите изображение одежды", "Нет данных"
        
        # Анализ ансамблем
        analysis = self.ensemble.analyze_image(image)
        
        # Детальное описание
        description = self.ensemble.get_detailed_description(analysis)
        
        # Детальный анализ для разработчика
        detailed_analysis = "🔍 **ДЕТАЛЬНЫЙ АНАЛИЗ:**\n\n"
        
        if 'fashion_clip' in analysis:
            clip = analysis['fashion_clip']
            detailed_analysis += "**FashionCLIP:**\n"
            for category, data in clip.items():
                if category != 'error':
                    best = data['best_match']
                    detailed_analysis += f"  {category}: {best['item']} ({best['confidence']:.1%})\n"
        
        if 'yolo' in analysis:
            yolo = analysis['yolo']
            detailed_analysis += f"\n**YOLO:** {yolo.get('total_items', 0)} объектов\n"
        
        if 'resnet' in analysis:
            resnet = analysis['resnet']
            detailed_analysis += f"\n**ResNet:** текстура {resnet.get('texture_complexity', 'unknown')}\n"
        
        return description, detailed_analysis
    
    def add_to_wardrobe(self, image):
        """Добавляет предмет в гардероб"""
        if image is None:
            return "❌ Пожалуйста, загрузите изображение одежды", "Нет данных"
        
        # Анализируем предмет
        analysis = self.ensemble.analyze_image(image)
        
        # Добавляем в гардероб
        item_id = self.vit_manager.add_item_from_analysis(image, analysis)
        
        # Получаем информацию о гардеробе
        wardrobe_info = self.vit_manager.get_wardrobe_info()
        
        return f"✅ Предмет добавлен в гардероб! ID: {item_id}", f"📊 Гардероб: {wardrobe_info['total_items']} предметов"
    
    def add_multiple_to_wardrobe(self, files):
        """Добавляет несколько предметов в гардероб из файлов"""
        if not files:
            return "❌ Пожалуйста, загрузите изображения", "Нет данных"
        
        added_count = 0
        errors = []
        
        for i, file in enumerate(files):
            if file is None:
                continue
                
            try:
                # Открываем изображение
                image = Image.open(file.name)
                
                # Анализируем предмет
                analysis = self.ensemble.analyze_image(image)
                
                # Добавляем в гардероб
                item_id = self.vit_manager.add_item_from_analysis(image, analysis)
                added_count += 1
                
            except Exception as e:
                errors.append(f"Ошибка с файлом {i+1}: {str(e)}")
        
        # Получаем информацию о гардеробе
        wardrobe_info = self.vit_manager.get_wardrobe_info()
        
        result_text = f"✅ Добавлено {added_count} предметов в гардероб!\n"
        if errors:
            result_text += f"\n⚠️ Ошибки:\n" + "\n".join(errors)
        
        return result_text, f"📊 Гардероб: {wardrobe_info['total_items']} предметов"
    
    def generate_outfit(self, occasion):
        """Генерирует образ для указанного случая"""
        if not occasion:
            return "❌ Пожалуйста, выберите случай", "Нет данных"
        
        # Генерируем образ
        outfit_result = self.vit_manager.generate_outfit_for_occasion(occasion)
        
        if "error" in outfit_result:
            return f"❌ {outfit_result['error']}", "Нет данных"
        
        # Форматируем результат
        outfit_text = f"🎯 **ОБРАЗ ДЛЯ СЛУЧАЯ: {occasion.upper()}**\n\n"
        outfit_text += f"📊 Уверенность: {outfit_result['confidence']:.1%}\n"
        outfit_text += f"👕 Количество предметов: {outfit_result['total_items']}\n\n"
        
        if outfit_result['outfit']:
            outfit_text += "**ВЫБРАННЫЕ ПРЕДМЕТЫ:**\n"
            for i, item in enumerate(outfit_result['outfit'], 1):
                features = item['features']
                item_desc = []
                if features.get('garment_type'):
                    item_desc.append(features['garment_type'])
                if features.get('color'):
                    item_desc.append(features['color'])
                if features.get('style'):
                    item_desc.append(features['style'])
                
                outfit_text += f"{i}. {' '.join(item_desc)} (уверенность: {item['score']:.1%})\n"
        
        outfit_text += f"\n**ОБЪЯСНЕНИЕ:**\n{outfit_result['explanation']}"
        
        # Получаем статистику гардероба для отображения
        wardrobe_info = self.vit_manager.get_wardrobe_info()
        
        return outfit_text, f"📈 Статистика гардероба: {wardrobe_info['total_items']} предметов"
    
    def get_wardrobe_info(self):
        """Получает информацию о гардеробе"""
        wardrobe_info = self.vit_manager.get_wardrobe_info()
        
        info_text = f"📊 **СТАТИСТИКА ГАРДЕРОБА**\n\n"
        info_text += f"👕 Всего предметов: {wardrobe_info['total_items']}\n"
        
        if wardrobe_info['item_types']:
            info_text += f"🏷️ Типы одежды: {', '.join(wardrobe_info['item_types'])}\n"
        
        if wardrobe_info['colors']:
            info_text += f"🎨 Цвета: {', '.join(wardrobe_info['colors'])}\n"
        
        if wardrobe_info['styles']:
            info_text += f"✨ Стили: {', '.join(wardrobe_info['styles'])}\n"
        
        return info_text, "📊 Информация о гардеробе"


def create_interface():
    """Создает Gradio интерфейс"""
    stylist = FashionAIStylist()
    
    def analyze_image(image):
        return stylist.analyze_outfit(image)
    
    def add_to_wardrobe(image):
        return stylist.add_to_wardrobe(image)
    
    def add_multiple_to_wardrobe(files):
        return stylist.add_multiple_to_wardrobe(files)
    
    def generate_outfit(occasion):
        return stylist.generate_outfit(occasion)
    
    def get_wardrobe_info():
        return stylist.get_wardrobe_info()
    
    with gr.Blocks(title="🎯 AI Стилист - Ансамбль Моделей + ViT") as interface:
        gr.Markdown("# 🎯 AI СТИЛИСТ - АНСАМБЛЬ МОДЕЛЕЙ + ViT")
        gr.Markdown("**FashionCLIP + YOLO + ResNet + Vision Transformer** - анализ и сборка образов")
        
        # Вкладки для разных функций
        with gr.Tabs():
            # Вкладка анализа
            with gr.Tab("🔍 Анализ одежды"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="📷 Загрузите изображение одежды")
                        analyze_btn = gr.Button("🎯 АНАЛИЗИРОВАТЬ", variant="primary")
                    
                    with gr.Column():
                        description_output = gr.Textbox(label="🧬 ОПИСАНИЕ ОДЕЖДЫ", lines=8)
                        analysis_output = gr.Textbox(label="🔍 ДЕТАЛЬНЫЙ АНАЛИЗ", lines=10)
                
                analyze_btn.click(
                    fn=analyze_image,
                    inputs=[image_input],
                    outputs=[description_output, analysis_output]
                )
            
            # Вкладка гардероба
            with gr.Tab("👕 Гардероб"):
                with gr.Row():
                    with gr.Column():
                        wardrobe_image_input = gr.Image(type="pil", label="📷 Добавить предмет в гардероб")
                        add_to_wardrobe_btn = gr.Button("➕ ДОБАВИТЬ В ГАРДЕРОБ", variant="primary")
                        wardrobe_status = gr.Textbox(label="📊 Статус", lines=2)
                    
                    with gr.Column():
                        wardrobe_info_btn = gr.Button("📊 ПОКАЗАТЬ СТАТИСТИКУ", variant="secondary")
                        wardrobe_info_output = gr.Textbox(label="📈 Информация о гардеробе", lines=8)
                
                add_to_wardrobe_btn.click(
                    fn=add_to_wardrobe,
                    inputs=[wardrobe_image_input],
                    outputs=[wardrobe_status, wardrobe_info_output]
                )
                
                wardrobe_info_btn.click(
                    fn=get_wardrobe_info,
                    outputs=[wardrobe_info_output, gr.Textbox()]
                )
                
                # Массовая загрузка
                gr.Markdown("---")
                gr.Markdown("### 📦 Массовая загрузка гардероба")
                
                with gr.Row():
                    with gr.Column():
                        multiple_images_input = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="📷 Загрузите до 20 изображений одежды",
                            height=200
                        )
                        add_multiple_btn = gr.Button("📦 ДОБАВИТЬ ВСЕ В ГАРДЕРОБ", variant="primary", size="lg")
                        multiple_status = gr.Textbox(label="📊 Статус массовой загрузки", lines=3)
                    
                    with gr.Column():
                        gr.Markdown("""
                        **Инструкция по массовой загрузке:**
                        1. Выберите до 20 изображений одежды
                        2. Нажмите "ДОБАВИТЬ ВСЕ В ГАРДЕРОБ"
                        3. Дождитесь обработки всех изображений
                        4. Проверьте статистику гардероба
                        
                        **Поддерживаемые форматы:** JPG, PNG, WEBP
                        **Максимальный размер:** 10 MB на изображение
                        """)
                
                add_multiple_btn.click(
                    fn=add_multiple_to_wardrobe,
                    inputs=[multiple_images_input],
                    outputs=[multiple_status, wardrobe_info_output]
                )
            
            # Вкладка генерации образов
            with gr.Tab("🎯 Генерация образов"):
                with gr.Row():
                    with gr.Column():
                        occasion_dropdown = gr.Dropdown(
                            choices=["свидание", "спорт", "прогулка", "работа", "путешествия", "вечеринка", "отпуск", "шопинг"],
                            label="🎯 Выберите случай",
                            value="прогулка"
                        )
                        generate_outfit_btn = gr.Button("✨ СОБРАТЬ ОБРАЗ", variant="primary")
                    
                    with gr.Column():
                        outfit_output = gr.Textbox(label="👗 Сгенерированный образ", lines=12)
                        outfit_stats = gr.Textbox(label="📊 Статистика", lines=3)
                
                generate_outfit_btn.click(
                    fn=generate_outfit,
                    inputs=[occasion_dropdown],
                    outputs=[outfit_output, outfit_stats]
                )
        
        gr.Markdown("""
        ### 🚀 ИСПОЛЬЗУЕМЫЕ МОДЕЛИ:
        - **FashionCLIP** - классификация типа и стиля одежды
        - **YOLO** - обнаружение объектов и композиции
        - **ResNet** - анализ текстур и цветов
        - **Vision Transformer (ViT)** - анализ совместимости и сборка образов
        
        ### 📋 ИНСТРУКЦИЯ:
        1. **Анализ**: Загрузите фото одежды для анализа
        2. **Гардероб**: Добавьте предметы в свой гардероб (по одному или массово)
        3. **Образы**: Выберите случай и получите готовый образ
        """)
    
    return interface


def main():
    print("🎯 AI СТИЛИСТ - АНСАМБЛЬ МОДЕЛЕЙ + ViT")
    print("=" * 60)
    print("✨ ИСПОЛЬЗУЕМЫЕ МОДЕЛИ: FashionCLIP + YOLO + ResNet + Vision Transformer")
    print("=" * 60)
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )


if __name__ == "__main__":
    main()
