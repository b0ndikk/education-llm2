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
    """МАКСИМАЛЬНЫЕ ПРОМТЫ ДЛЯ ВСЕЙ ОДЕЖДЫ МИРА"""
    return {
        'garment_category': [
            # === ОБУВЬ (200+ ТИПОВ) ===
            "sneakers", "running shoes", "athletic shoes", "training shoes", "basketball shoes",
            "tennis shoes", "football cleats", "soccer cleats", "golf shoes", "baseball cleats",
            "hiking boots", "work boots", "combat boots", "military boots", "cowboy boots",
            "ankle boots", "knee-high boots", "thigh-high boots", "rain boots", "snow boots",
            "high heels", "stiletto heels", "pump shoes", "platform shoes", "wedge shoes",
            "block heels", "kitten heels", "court shoes", "dress shoes", "evening shoes",
            "sandals", "flip flops", "slides", "gladiator sandals", "wedge sandals",
            "espadrilles", "mules", "loafers", "oxford shoes", "derby shoes",
            "brogues", "monk straps", "boat shoes", "deck shoes", "driving shoes",
            "slip-on shoes", "ballet flats", "ballerina shoes", "mary janes", "dolly shoes",
            "clogs", "crocs", "orthopedic shoes", "safety shoes", "dance shoes",
            "ski boots", "snowboard boots", "ice skates", "roller skates", "cycling shoes",
            "wrestling shoes", "boxing shoes", "climbing shoes", "yoga shoes", "pilates shoes",
            
            # === ВЕРХНЯЯ ОДЕЖДА ===
            "t-shirt", "tank top", "crop top", "blouse", "shirt", "dress shirt", "button-down shirt",
            "polo shirt", "henley shirt", "flannel shirt", "hawaiian shirt", "peasant blouse",
            "tunic", "bodysuit", "corset", "bustier", "camisole", "shell top", "halter top",
            "tube top", "off-shoulder top", "cold shoulder top", "batwing top", "kimono top",
            "sweater", "pullover", "cardigan", "jumper", "turtleneck", "polo neck", "roll neck",
            "hoodie", "sweatshirt", "fleece", "quarter zip", "hooded sweatshirt",
            
            # === ПЛАТЬЯ ===
            "cocktail dress", "evening gown", "wedding dress", "bridesmaid dress", "prom dress",
            "maxi dress", "midi dress", "mini dress", "shift dress", "a-line dress",
            "bodycon dress", "sheath dress", "wrap dress", "shirt dress", "t-shirt dress",
            "sweater dress", "knit dress", "jersey dress", "bandage dress", "slip dress",
            "babydoll dress", "empire waist dress", "fit and flare dress", "ball gown",
            "mermaid dress", "trumpet dress", "tea length dress", "high-low dress",
            
            # === НИЖНЯЯ ОДЕЖДА ===
            "jeans", "skinny jeans", "straight jeans", "bootcut jeans", "flare jeans",
            "boyfriend jeans", "mom jeans", "high-waisted jeans", "low-rise jeans",
            "jeggings", "denim shorts", "denim skirt", "chino pants", "khaki pants",
            "dress pants", "trousers", "slacks", "pleated pants", "flat front pants",
            "cargo pants", "harem pants", "palazzo pants", "wide-leg pants", "culottes",
            "capri pants", "cropped pants", "ankle pants", "leggings", "yoga pants",
            "joggers", "sweatpants", "track pants", "athletic pants", "cycling shorts",
            
            # === ЮБКИ ===
            "pencil skirt", "a-line skirt", "pleated skirt", "wrap skirt", "circle skirt",
            "tulip skirt", "bubble skirt", "tiered skirt", "maxi skirt", "midi skirt",
            "mini skirt", "micro skirt", "denim skirt", "leather skirt", "suede skirt",
            
            # === ВЕРХНЯЯ ОДЕЖДА (ПАЛЬТО/КУРТКИ) ===
            "winter coat", "peacoat", "duffle coat", "trench coat", "raincoat",
            "overcoat", "topcoat", "parka", "anorak", "puffer jacket", "down jacket",
            "bomber jacket", "flight jacket", "leather jacket", "motorcycle jacket",
            "denim jacket", "trucker jacket", "blazer", "sports coat", "suit jacket",
            "cardigan", "poncho", "cape", "shawl", "windbreaker", "softshell jacket",
            
            # === НИЖНЕЕ БЕЛЬЕ ===
            "bra", "bralette", "sports bra", "push-up bra", "strapless bra",
            "panties", "briefs", "hipster", "thong", "g-string", "boyshorts",
            "lingerie set", "chemise", "babydoll", "teddy", "bodystocking",
            "camisole", "slip", "shapewear", "corset", "bustier",
            
            # === СПОРТИВНАЯ ОДЕЖДА ===
            "tracksuit", "jogging suit", "training suit", "yoga set", "gym set",
            "swimsuit", "bikini", "tankini", "one-piece swimsuit", "swim trunks",
            "board shorts", "rash guard", "wetsuit", "drysuit", "ski suit",
            "snowboard pants", "cycling jersey", "running tights", "compression wear",
            
            # === ТРАДИЦИОННАЯ/ЭТНИЧЕСКАЯ ОДЕЖДА ===
            "kimono", "yukata", "hakama", "sari", "lehenga", "salwar kameez",
            "cheongsam", "qipao", "hanbok", "ao dai", "thobe", "dishdasha",
            "kilt", "dirndl", "lederhosen", "poncho", "sarong", "pareo",
            "kaftan", "djellaba", "abaya", "hijab", "burqa", "niqab",
            
            # === АКСЕССУАРЫ ===
            "backpack", "rucksack", "knapsack", "hiking backpack", "laptop backpack",
            "messenger bag", "crossbody bag", "shoulder bag", "tote bag", "shopping bag",
            "clutch", "evening bag", "wristlet", "belt bag", "fanny pack",
            "handbag", "purse", "satchel", "duffle bag", "gym bag",
            "suitcase", "travel bag", "briefcase", "portfolio", "beach bag",
            
            # === ГОЛОВНЫЕ УБОРЫ ===
            "baseball cap", "snapback", "trucker hat", "beanie", "winter hat",
            "fedora", "trilby", "panama hat", "bucket hat", "sun hat",
            "beret", "newsboy cap", "flat cap", "bowler hat", "top hat",
            "cowboy hat", "sombrero", "ushanka", "trapper hat",
            
            # === ДРУГИЕ АКСЕССУАРЫ ===
            "scarf", "wrap", "pashmina", "shawl", "stole",
            "gloves", "mittens", "leather gloves", "driving gloves",
            "belt", "waist belt", "chain belt", "suspenders", "braces",
            "tie", "neck tie", "bow tie", "bolo tie", "cravat",
            "socks", "ankle socks", "knee-high socks", "thigh-high socks",
            "stockings", "pantyhose", "tights", "leggings",
            "jewelry", "necklace", "bracelet", "earrings", "ring"
        ],

        'material_fabric': [
            # === НАТУРАЛЬНЫЕ ТКАНИ ===
            "cotton fabric", "organic cotton", "egyptian cotton", "pima cotton",
            "linen fabric", "flax linen", "pure linen", "washed linen",
            "silk fabric", "pure silk", "silk satin", "silk chiffon", "silk crepe",
            "silk dupioni", "silk organza", "silk taffeta", "silk velvet",
            "wool fabric", "merino wool", "cashmere wool", "lambswool", "shetland wool",
            "angora wool", "mohair wool", "alpaca wool", "camel hair",
            "leather material", "genuine leather", "nappa leather", "suede leather",
            "patent leather", "exotic leather", "vegan leather", "bonded leather",
            "fur material", "real fur", "faux fur", "shearling", "sheepskin",
            "denim fabric", "raw denim", "selvedge denim", "stretch denim",
            
            # === СИНТЕТИЧЕСКИЕ ТКАНИ ===
            "polyester fabric", "polyester blend", "microfiber", "polyamide",
            "nylon fabric", "ripstop nylon", "tactical nylon", "cordura nylon",
            "rayon fabric", "viscose rayon", "modal fabric", "lyocell", "tencel",
            "spandex fabric", "elastane", "lycra", "stretch fabric",
            "acrylic fabric", "fleece", "polar fleece", "sherpa",
            "velvet fabric", "velour", "velveteen", "crushed velvet",
            "satin fabric", "charmeuse", "sateen", "duchesse satin",
            "chiffon fabric", "georgette", "crepe fabric", "crepe de chine",
            
            # === ТЕХНИЧЕСКИЕ ТКАНИ ===
            "waterproof fabric", "water-resistant", "windproof", "breathable fabric",
            "moisture-wicking", "quick-dry", "thermal insulation", "heat-retaining",
            "performance fabric", "technical fabric", "athletic fabric", "compression fabric",
            "ripstop fabric", "canvas fabric", "duck canvas", "waxed canvas",
            "neoprene", "rubber", "latex", "vinyl", "pvc"
        ],

        'color_pattern': [
            # === БАЗОВЫЕ ЦВЕТА ===
            "black color", "jet black", "carbon black", "matte black", "glossy black",
            "white color", "pure white", "bright white", "off-white", "ivory white",
            "cream color", "eggshell", "vanilla", "beige color", "tan", "khaki",
            "nude color", "flesh tone", "camel color", "taupe", "greige",
            "gray color", "charcoal gray", "slate gray", "silver gray", "steel gray",
            "ash gray", "heather gray", "smoke gray", "pewter",
            
            # === ЯРКИЕ ЦВЕТА ===
            "red color", "crimson red", "scarlet red", "ruby red", "burgundy",
            "wine red", "maroon", "oxblood", "candy apple red", "fire engine red",
            "pink color", "hot pink", "bubblegum pink", "pastel pink", "blush pink",
            "rose pink", "fuchsia", "magenta", "cerise", "salmon pink",
            "orange color", "bright orange", "tangerine", "coral", "peach",
            "apricot", "burnt orange", "pumpkin orange", "safety orange",
            "yellow color", "sunshine yellow", "lemon yellow", "golden yellow",
            "mustard yellow", "canary yellow", "amber", "honey yellow",
            "green color", "emerald green", "forest green", "olive green", "sage green",
            "mint green", "seafoam green", "lime green", "army green", "hunter green",
            "blue color", "navy blue", "royal blue", "cobalt blue", "sky blue",
            "baby blue", "turquoise", "teal", "aqua", "sapphire blue",
            "purple color", "royal purple", "lavender", "lilac", "violet",
            "plum purple", "eggplant", "amethyst", "orchid", "mauve",
            "brown color", "chocolate brown", "espresso brown", "chestnut brown",
            "camel brown", "caramel", "hazelnut", "mahogany", "walnut",
            
            # === ПАТТЕРНЫ И ПРИНТЫ ===
            "solid color", "plain color", "no pattern",
            "striped pattern", "pinstripe", "candy stripe", "awning stripe",
            "checkered pattern", "gingham", "buffalo check", "windowpane check",
            "plaid pattern", "tartan", "madras plaid", "argyle pattern",
            "floral print", "botanical print", "tropical print", "abstract floral",
            "animal print", "leopard print", "zebra print", "tiger print", "cheetah print",
            "geometric pattern", "polka dot", "houndstooth", "herringbone",
            "paisley pattern", "damask pattern", "jacquard pattern", "brocade",
            "camouflage print", "digital camo", "woodland camo", "urban camo",
            "tie-dye pattern", "psychedelic print", "batik", "ikat",
            "ethnic pattern", "tribal print", "aztec print", "native print",
            "graphic print", "logo print", "text print", "slogan print",
            
            # === ЭФФЕКТЫ И ОТДЕЛКИ ===
            "metallic finish", "gold metallic", "silver metallic", "bronze metallic",
            "shimmer effect", "iridescent", "holographic", "chrome finish",
            "glitter details", "sequin accents", "beaded work", "rhinestone details",
            "ombre effect", "gradient colors", "color fade", "dip-dye",
            "color block", "two-tone", "multi-color", "rainbow colors",
            "neutral colors", "earth tones", "pastel colors", "jewel tones"
        ],

        'style_occasion': [
            # === СТИЛИ ===
            "casual style", "everyday casual", "weekend casual", "smart casual",
            "formal style", "black tie", "white tie", "cocktail attire",
            "business style", "business formal", "business professional", "corporate attire",
            "business casual", "office appropriate", "workplace attire",
            "sporty style", "athletic wear", "activewear", "gym wear", "training clothes",
            "streetwear style", "urban fashion", "hip-hop style", "skatewear",
            "vintage style", "retro fashion", "old-school", "nostalgic style",
            "bohemian style", "boho chic", "hippie style", "festival wear",
            "minimalist style", "minimal fashion", "simple style", "Scandinavian style",
            "preppy style", "ivy league", "trad style", "country club attire",
            "romantic style", "feminine style", "soft romantic", "ethereal style",
            "edgy style", "punk style", "gothic style", "rock and roll style",
            "military style", "utility wear", "tactical gear", "cargo style",
            "resort wear", "vacation style", "cruise wear", "beach resort",
            
            # === ПОВОДЫ ===
            "wedding attire", "bridal wear", "wedding guest", "formal wedding",
            "evening occasion", "night out", "date night", "dinner party",
            "cocktail party", "holiday party", "celebration", "festive occasion",
            "job interview", "professional meeting", "corporate event", "conference",
            "graduation ceremony", "prom night", "homecoming", "formal dance",
            "vacation wear", "travel clothes", "airport style", "holiday travel",
            "beach wear", "pool party", "summer vacation", "tropical holiday",
            "winter wear", "ski vacation", "snow season", "cold weather attire",
            "spring fashion", "summer style", "autumn wear", "fall fashion",
            
            # === СПЕЦИФИЧЕСКИЕ НАЗНАЧЕНИЯ ===
            "uniform", "work uniform", "professional uniform", "service uniform",
            "protective wear", "safety gear", "workwear", "industrial clothing",
            "performance wear", "competition attire", "stage costume", "theatrical costume",
            "ceremonial dress", "traditional ceremony", "cultural event", "religious attire",
            "fancy dress", "costume party", "cosplay", "character costume"
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