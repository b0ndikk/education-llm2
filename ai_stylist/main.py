#!/usr/bin/env python3
"""
🎯 ГЛАВНЫЙ ФАЙЛ С ИНТЕРФЕЙСОМ
"""

import gradio as gr
from ensemble_manager import FashionEnsemble
from PIL import Image

class FashionAIStylist:
    def __init__(self):
        self.ensemble = FashionEnsemble()
    
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

def create_interface():
    """Создает Gradio интерфейс"""
    stylist = FashionAIStylist()
    
    def analyze_image(image):
        return stylist.analyze_outfit(image)
    
    with gr.Blocks(title="🎯 AI Стилист - Ансамбль Моделей") as interface:
        gr.Markdown("# 🎯 AI СТИЛИСТ - АНСАМБЛЬ МОДЕЛЕЙ")
        gr.Markdown("**FashionCLIP + YOLO + ResNet** - максимальная точность анализа одежды")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="📷 Загрузите изображение одежды")
                analyze_btn = gr.Button("🎯 АНАЛИЗИРОВАТЬ", variant="primary")
            
            with gr.Column():
                description_output = gr.Textbox(label="🧬 ОПИСАНИЕ ОДЕЖДЫ", lines=8)
                analysis_output = gr.Textbox(label="🔍 ДЕТАЛЬНЫЙ АНАЛИЗ", lines=10)
        
        gr.Markdown("""
        ### 🚀 ИСПОЛЬЗУЕМЫЕ МОДЕЛИ:
        - **FashionCLIP** - классификация типа и стиля одежды
        - **YOLO** - обнаружение объектов и композиции
        - **ResNet** - анализ текстур и цветов
        """)
        
        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_input],
            outputs=[description_output, analysis_output]
        )
    
    return interface

def main():
    print("🎯 AI СТИЛИСТ - АНСАМБЛЬ МОДЕЛЕЙ")
    print("=" * 60)
    print("✨ ИСПОЛЬЗУЕМЫЕ МОДЕЛИ: FashionCLIP + YOLO + ResNet")
    print("=" * 60)
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

if __name__ == "__main__":
    main()