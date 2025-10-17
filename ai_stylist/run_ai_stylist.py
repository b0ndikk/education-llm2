#!/usr/bin/env python3
"""
🚀 ЗАПУСК AI-СТИЛИСТА
"""

import os
import sys
import subprocess

def main():
    print("🎯 ЗАПУСК AI-СТИЛИСТА")
    print("=" * 40)
    
    # Проверяем зависимости
    try:
        import fashion_clip
        print("✅ FashionCLIP установлен")
    except ImportError:
        print("📦 Устанавливаем FashionCLIP...")
        subprocess.run([sys.executable, "-m", "pip", "install", "fashion-clip"])
    
    try:
        import gradio
        print("✅ Gradio установлен")
    except ImportError:
        print("📦 Устанавливаем Gradio...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gradio"])
    
    # Запускаем AI-стилиста
    os.chdir("ai_stylist")
    subprocess.run([sys.executable, "fashion_ai_stylist.py"])

if __name__ == "__main__":
    main()
