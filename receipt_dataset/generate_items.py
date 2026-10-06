#!/usr/bin/env python3
"""Synthetic dataset for normalising receipt item names to generic products.

Turns raw receipt line items like

    LINDT DARK CHOC 100G      ->  Dark Chocolate
    GV WHL MLK 1GAL           ->  Whole Milk
    COKE ZERO 12PK 355ML      ->  Diet Cola

The model learns to drop brands, sizes, pack counts and store codes, and to
expand receipt abbreviations. The abbreviations come from a curated table
plus automatic vowel-dropping and truncation. Optional OCR noise reuses the
noise model in generate.py.

Usage:
    python generate_items.py --out data_items --train 20000 --val 2000 --test 2000
    python generate_items.py --holdout-brands 0.15   # test on unseen brands
"""

import argparse
import json
import random
import re
from pathlib import Path

from generate import LEVELS, corrupt_line, write_jsonl

# --------------------------------------------------------------------------
# Product catalogue
#
# target | category | brands (comma sep, "-" = unbranded/store) |
#   sizes (comma sep) | raw name variants as printed (semicolon sep)
# --------------------------------------------------------------------------

CATALOGUE = """
Dark Chocolate|confectionery|LINDT,GHIRARDELLI,GREEN & BLACKS,HERSHEYS,GODIVA,TONYS CHOCOLONELY,ALTER ECO|100G,3.5OZ,80G,4.4OZ,200G|DARK CHOCOLATE;DARK CHOCOLATE BAR;CHOCOLATE DARK 70%;DARK CHOCOLATE 85% COCOA;EXCELLENCE DARK CHOCOLATE
Milk Chocolate|confectionery|CADBURY,HERSHEYS,LINDT,MILKA,GALAXY,TOBLERONE|100G,1.55OZ,200G,45G|MILK CHOCOLATE;MILK CHOCOLATE BAR;DAIRY MILK CHOCOLATE;CHOCOLATE MILK BAR
White Chocolate|confectionery|LINDT,GHIRARDELLI,MILKYBAR,HERSHEYS|100G,3.5OZ|WHITE CHOCOLATE;WHITE CHOCOLATE BAR
Chocolate Truffles|confectionery|LINDT,FERRERO,GODIVA|200G,5.1OZ,12CT|CHOCOLATE TRUFFLES;LINDOR TRUFFLES;ROCHER TRUFFLES ASSORTED
Gummy Candy|confectionery|HARIBO,TROLLI,SOUR PATCH|5OZ,140G,200G|GUMMY BEARS;GUMMY CANDY;SOUR GUMMY WORMS;GOLDBEARS GUMMY
Chewing Gum|confectionery|TRIDENT,EXTRA,ORBIT,WRIGLEYS|14CT,15PC,3PK|CHEWING GUM;SUGAR FREE GUM;SPEARMINT GUM;PEPPERMINT GUM
Whole Milk|dairy|HORIZON,ORGANIC VALLEY,FAIRLIFE,LAND O LAKES,-|1GAL,1/2GAL,2L,4PT,1L,64OZ|WHOLE MILK;MILK WHOLE;VITAMIN D MILK;WHOLE MILK HOMOGENIZED
Skim Milk|dairy|HORIZON,FAIRLIFE,-|1GAL,1/2GAL,2L,1L|SKIM MILK;FAT FREE MILK;NONFAT MILK;MILK SKIM
Low Fat Milk|dairy|HORIZON,ORGANIC VALLEY,-|1GAL,1/2GAL,2L|2% MILK;REDUCED FAT MILK;LOW FAT MILK;1% LOWFAT MILK
Oat Milk|dairy|OATLY,CALIFIA,SILK,PLANET OAT|64OZ,1L,32OZ|OAT MILK;OATMILK ORIGINAL;OAT BEVERAGE BARISTA
Almond Milk|dairy|SILK,ALMOND BREEZE,CALIFIA|64OZ,1/2GAL,1L|ALMOND MILK;ALMONDMILK UNSWEETENED;ALMOND BEVERAGE
Butter|dairy|KERRYGOLD,LAND O LAKES,PLUGRA,-|8OZ,1LB,250G,4 STICKS|BUTTER;SALTED BUTTER;UNSALTED BUTTER;SWEET CREAM BUTTER;IRISH BUTTER
Cheddar Cheese|dairy|TILLAMOOK,CABOT,KRAFT,CRACKER BARREL,-|8OZ,2LB,200G,16OZ|CHEDDAR CHEESE;SHARP CHEDDAR;MILD CHEDDAR CHEESE;SHREDDED CHEDDAR;EXTRA SHARP CHEDDAR BLOCK
Mozzarella Cheese|dairy|GALBANI,POLLY-O,KRAFT,-|8OZ,16OZ,125G|MOZZARELLA CHEESE;SHREDDED MOZZARELLA;FRESH MOZZARELLA;MOZZARELLA LOW MOISTURE
Parmesan Cheese|dairy|BELGIOIOSO,KRAFT,-|5OZ,8OZ,200G|PARMESAN CHEESE;GRATED PARMESAN;PARMIGIANO REGGIANO;SHAVED PARMESAN
Cream Cheese|dairy|PHILADELPHIA,-|8OZ,227G,12OZ|CREAM CHEESE;CREAM CHEESE BRICK;WHIPPED CREAM CHEESE;CREAM CHEESE SPREAD
Greek Yogurt|dairy|CHOBANI,FAGE,OIKOS,-|5.3OZ,32OZ,500G,4PK|GREEK YOGURT;PLAIN GREEK YOGURT;GREEK YOGURT VANILLA;NONFAT GREEK YOGURT;TOTAL 0% GREEK YOGURT
Yogurt|dairy|YOPLAIT,DANNON,ACTIVIA,-|6OZ,32OZ,4PK,8PK|YOGURT;STRAWBERRY YOGURT;LOW FAT YOGURT;ORIGINAL YOGURT
Sour Cream|dairy|DAISY,BREAKSTONE,-|16OZ,8OZ|SOUR CREAM;LIGHT SOUR CREAM;SOUR CREAM REGULAR
Heavy Cream|dairy|HORIZON,-|16OZ,1PT,32OZ|HEAVY CREAM;HEAVY WHIPPING CREAM;WHIPPING CREAM
Eggs|dairy|EGGLANDS BEST,VITAL FARMS,PETE AND GERRYS,-|12CT,18CT,6CT,1DZ|LARGE EGGS;EGGS LARGE;GRADE A LARGE EGGS;BROWN EGGS;FREE RANGE EGGS;PASTURE RAISED EGGS
White Bread|bakery|WONDER,SUNBEAM,-|20OZ,24OZ,800G|WHITE BREAD;BREAD WHITE;SANDWICH BREAD WHITE;CLASSIC WHITE BREAD
Wheat Bread|bakery|DAVES KILLER BREAD,NATURES OWN,PEPPERIDGE FARM,-|20OZ,24OZ,27OZ|WHOLE WHEAT BREAD;WHEAT BREAD;100% WHOLE WHEAT BREAD;HONEY WHEAT BREAD;21 WHOLE GRAINS BREAD
Bagels|bakery|THOMAS,LENDERS,-|6CT,20OZ|BAGELS;PLAIN BAGELS;EVERYTHING BAGELS;BAGELS PRE-SLICED
Tortillas|bakery|MISSION,GUERRERO,-|10CT,20CT,8CT|FLOUR TORTILLAS;CORN TORTILLAS;TORTILLAS FLOUR SOFT TACO
Croissants|bakery|-|4CT,6CT|CROISSANTS;BUTTER CROISSANTS;MINI CROISSANTS
Bananas|produce|DOLE,CHIQUITA,-|LB,KG,EA|BANANAS;BANANA;BANANAS YELLOW;ORGANIC BANANAS
Apples|produce|-|LB,KG,3LB BAG|GALA APPLES;APPLES GALA;HONEYCRISP APPLES;FUJI APPLE;GRANNY SMITH APPLES;APPLE RED DELICIOUS
Avocados|produce|-|EA,4CT,BAG|AVOCADO;HASS AVOCADO;AVOCADOS LARGE;AVOCADO BAG
Tomatoes|produce|-|LB,KG|TOMATOES;ROMA TOMATOES;TOMATOES ON THE VINE;CHERRY TOMATOES;GRAPE TOMATOES
Potatoes|produce|-|5LB,LB,10LB|RUSSET POTATOES;POTATOES RUSSET;YUKON GOLD POTATOES;RED POTATOES;SWEET POTATOES
Onions|produce|-|LB,3LB,KG|YELLOW ONIONS;ONIONS YELLOW;RED ONION;WHITE ONIONS;SWEET ONIONS
Spinach|produce|EARTHBOUND FARM,-|5OZ,10OZ,16OZ|BABY SPINACH;SPINACH;ORGANIC BABY SPINACH;SPINACH LEAVES
Lettuce|produce|-|EA,HEAD|ROMAINE LETTUCE;ICEBERG LETTUCE;LETTUCE ROMAINE HEARTS;BUTTER LETTUCE;LETTUCE GREEN LEAF
Carrots|produce|BOLTHOUSE,GRIMMWAY,-|1LB,2LB,5LB|CARROTS;BABY CARROTS;CARROTS WHOLE;SHREDDED CARROTS
Strawberries|produce|DRISCOLLS,-|1LB,16OZ,2LB|STRAWBERRIES;STRAWBERRY;STRAWBERRIES CLAMSHELL
Blueberries|produce|DRISCOLLS,-|6OZ,1PT,18OZ|BLUEBERRIES;BLUEBERRY;BLUEBERRIES PINT
Lemons|produce|-|EA,2LB BAG|LEMONS;LEMON;LEMONS BAG;MEYER LEMONS
Chicken Breast|meat|PERDUE,TYSON,FOSTER FARMS,-|LB,1.5LB,KG,2PK|CHICKEN BREAST;BONELESS SKINLESS CHICKEN BREAST;CHICKEN BREAST FILLETS;BNLS SKNLS CHICKEN BREAST
Chicken Thighs|meat|PERDUE,TYSON,-|LB,2LB|CHICKEN THIGHS;BONELESS CHICKEN THIGHS;CHICKEN THIGH BONE IN
Ground Beef|meat|-|1LB,2LB,500G|GROUND BEEF;GROUND BEEF 80/20;LEAN GROUND BEEF 93/7;GROUND CHUCK;BEEF MINCE
Steak|meat|-|LB,KG|RIBEYE STEAK;NEW YORK STRIP STEAK;SIRLOIN STEAK;BEEF STEAK TOP SIRLOIN
Bacon|meat|OSCAR MAYER,SMITHFIELD,WRIGHT,-|12OZ,16OZ,1LB|BACON;THICK CUT BACON;HICKORY SMOKED BACON;BACON SLICED
Sliced Ham|meat|OSCAR MAYER,HILLSHIRE FARM,BOARS HEAD,-|9OZ,16OZ,LB|SLICED HAM;HAM DELI SLICED;HONEY HAM;BLACK FOREST HAM;SMOKED HAM
Sausages|meat|JOHNSONVILLE,HILLSHIRE FARM,-|19OZ,14OZ,5CT|SAUSAGES;ITALIAN SAUSAGE;BRATWURST;BREAKFAST SAUSAGE LINKS;SMOKED SAUSAGE
Salmon|seafood|-|LB,KG,8OZ|SALMON FILLET;ATLANTIC SALMON;SALMON FILLETS SKIN ON;WILD SOCKEYE SALMON
Shrimp|seafood|-|1LB,2LB,12OZ|SHRIMP;LARGE SHRIMP RAW;COOKED SHRIMP;SHRIMP PEELED DEVEINED
Canned Tuna|seafood|STARKIST,BUMBLE BEE,CHICKEN OF THE SEA|5OZ,4PK,12OZ|TUNA;CHUNK LIGHT TUNA;SOLID WHITE ALBACORE TUNA;TUNA IN WATER
Spaghetti|pantry|BARILLA,DE CECCO,RONZONI,-|16OZ,1LB,500G|SPAGHETTI;SPAGHETTI PASTA;THIN SPAGHETTI;SPAGHETTI NO 5
Penne Pasta|pantry|BARILLA,DE CECCO,-|16OZ,1LB,500G|PENNE;PENNE RIGATE;PENNE PASTA;MINI PENNE
White Rice|pantry|MAHATMA,UNCLE BENS,-|2LB,5LB,1KG|WHITE RICE;LONG GRAIN WHITE RICE;JASMINE RICE;BASMATI RICE;RICE ENRICHED
Brown Rice|pantry|LUNDBERG,-|2LB,1KG|BROWN RICE;LONG GRAIN BROWN RICE;BROWN RICE WHOLE GRAIN
Pasta Sauce|pantry|RAO'S,PREGO,RAGU,BARILLA,-|24OZ,32OZ,680G|MARINARA SAUCE;PASTA SAUCE;TOMATO BASIL SAUCE;TRADITIONAL PASTA SAUCE
Ketchup|pantry|HEINZ,HUNTS,-|20OZ,32OZ,500ML|KETCHUP;TOMATO KETCHUP;KETCHUP SQUEEZE
Mayonnaise|pantry|HELLMANNS,BEST FOODS,DUKES,KEWPIE|30OZ,15OZ,400ML|MAYONNAISE;REAL MAYONNAISE;MAYO;LIGHT MAYONNAISE
Mustard|pantry|FRENCHS,GREY POUPON,-|14OZ,8OZ|YELLOW MUSTARD;DIJON MUSTARD;MUSTARD;SPICY BROWN MUSTARD
Peanut Butter|pantry|JIF,SKIPPY,SMUCKERS,-|16OZ,28OZ,40OZ|PEANUT BUTTER;CREAMY PEANUT BUTTER;CRUNCHY PEANUT BUTTER;NATURAL PEANUT BUTTER
Jam|pantry|SMUCKERS,BONNE MAMAN,-|12OZ,18OZ,340G|STRAWBERRY JAM;GRAPE JELLY;RASPBERRY PRESERVES;JAM APRICOT
Honey|pantry|NATURE NATE'S,-|12OZ,16OZ,32OZ|HONEY;RAW HONEY;CLOVER HONEY;HONEY BEAR SQUEEZE
Olive Oil|pantry|BERTOLLI,COLAVITA,CALIFORNIA OLIVE RANCH,-|500ML,1L,16.9OZ,25.5OZ|OLIVE OIL;EXTRA VIRGIN OLIVE OIL;EVOO;OLIVE OIL EXTRA VIRGIN
Vegetable Oil|pantry|CRISCO,WESSON,-|48OZ,1GAL,1L|VEGETABLE OIL;CANOLA OIL;OIL VEGETABLE PURE
All Purpose Flour|pantry|GOLD MEDAL,KING ARTHUR,PILLSBURY,-|5LB,2LB,1KG|ALL PURPOSE FLOUR;FLOUR ALL PURPOSE;UNBLEACHED FLOUR;PLAIN FLOUR
Sugar|pantry|DOMINO,C&H,-|4LB,2LB,1KG|GRANULATED SUGAR;SUGAR;PURE CANE SUGAR;BROWN SUGAR;POWDERED SUGAR
Salt|pantry|MORTON,DIAMOND CRYSTAL,-|26OZ,3LB|SALT;IODIZED SALT;KOSHER SALT;SEA SALT
Black Pepper|pantry|MCCORMICK,-|3OZ,6OZ|BLACK PEPPER;GROUND BLACK PEPPER;PEPPERCORNS BLACK
Breakfast Cereal|pantry|KELLOGGS,GENERAL MILLS,POST,QUAKER|12OZ,18OZ,500G|CORN FLAKES;CHEERIOS;FROSTED FLAKES;RAISIN BRAN;HONEY NUT CHEERIOS;CEREAL
Oatmeal|pantry|QUAKER,BOBS RED MILL,-|42OZ,18OZ,10CT|OLD FASHIONED OATS;ROLLED OATS;QUICK OATS;INSTANT OATMEAL;STEEL CUT OATS
Granola Bars|snacks|NATURE VALLEY,KIND,CLIF,QUAKER|6CT,12CT,8.9OZ|GRANOLA BARS;CRUNCHY GRANOLA BARS;PROTEIN BAR;CHEWY GRANOLA BARS;NUT BARS
Potato Chips|snacks|LAYS,KETTLE,UTZ,CAPE COD,PRINGLES|8OZ,10OZ,1.5OZ,150G|POTATO CHIPS;CLASSIC POTATO CHIPS;SEA SALT CHIPS;KETTLE CHIPS;SOUR CREAM ONION CHIPS
Tortilla Chips|snacks|TOSTITOS,DORITOS,SIETE,-|11OZ,13OZ|TORTILLA CHIPS;RESTAURANT STYLE TORTILLA CHIPS;NACHO CHEESE TORTILLA CHIPS
Crackers|snacks|RITZ,TRISCUIT,CHEEZ-IT,WHEAT THINS,GOLDFISH|8.8OZ,12OZ,200G|CRACKERS;ORIGINAL CRACKERS;CHEESE CRACKERS;WHOLE GRAIN CRACKERS
Cookies|snacks|OREO,CHIPS AHOY,PEPPERIDGE FARM,-|14OZ,13OZ,300G|COOKIES;CHOCOLATE CHIP COOKIES;SANDWICH COOKIES;OATMEAL RAISIN COOKIES
Popcorn|snacks|ORVILLE,SKINNYPOP,SMARTFOOD,-|3CT,6CT,4.4OZ|POPCORN;MICROWAVE POPCORN;BUTTER POPCORN;WHITE CHEDDAR POPCORN
Almonds|snacks|BLUE DIAMOND,WONDERFUL,-|16OZ,6OZ,1LB|ALMONDS;ROASTED ALMONDS;RAW ALMONDS;SALTED ALMONDS
Mixed Nuts|snacks|PLANTERS,-|10.3OZ,16OZ|MIXED NUTS;DELUXE MIXED NUTS;TRAIL MIX
Coffee Beans|beverages|STARBUCKS,LAVAZZA,PEETS,DUNKIN,ILLY|12OZ,1LB,250G,2LB|WHOLE BEAN COFFEE;COFFEE BEANS;MEDIUM ROAST COFFEE;FRENCH ROAST WHOLE BEAN
Ground Coffee|beverages|FOLGERS,MAXWELL HOUSE,STARBUCKS,DUNKIN|30.5OZ,12OZ,250G|GROUND COFFEE;CLASSIC ROAST GROUND COFFEE;COFFEE GROUND MEDIUM
Coffee Pods|beverages|KEURIG,NESPRESSO,STARBUCKS,GREEN MOUNTAIN|12CT,24CT,10CT|K-CUP PODS;COFFEE PODS;CAPSULES ESPRESSO;BREAKFAST BLEND K CUPS
Tea Bags|beverages|LIPTON,TWININGS,BIGELOW,TAZO|20CT,100CT,40CT|TEA BAGS;BLACK TEA;GREEN TEA;ENGLISH BREAKFAST TEA;CHAMOMILE TEA
Orange Juice|beverages|TROPICANA,SIMPLY,MINUTE MAID,-|52OZ,89OZ,1.75L,1L|ORANGE JUICE;OJ;PURE PREMIUM ORANGE JUICE;ORANGE JUICE NO PULP
Apple Juice|beverages|MOTTS,TREE TOP,-|64OZ,1L|APPLE JUICE;100% APPLE JUICE;APPLE JUICE UNSWEETENED
Cola|beverages|COCA-COLA,PEPSI,RC|2L,12PK,12OZ,6PK,355ML|COLA;COKE;COCA COLA;PEPSI COLA;CLASSIC COLA
Diet Cola|beverages|COCA-COLA,PEPSI|2L,12PK,12OZ|DIET COKE;COKE ZERO;DIET PEPSI;PEPSI ZERO SUGAR;DIET COLA
Lemon Lime Soda|beverages|SPRITE,7UP,SIERRA MIST|2L,12PK|LEMON LIME SODA;SPRITE;7UP SODA;LEMON LIME
Sparkling Water|beverages|LACROIX,PERRIER,SAN PELLEGRINO,BUBLY,TOPO CHICO|12PK,8PK,750ML,1L|SPARKLING WATER;SPARKLING MINERAL WATER;LIME SPARKLING WATER;SELTZER
Bottled Water|beverages|DASANI,AQUAFINA,FIJI,EVIAN,POLAND SPRING|24PK,1L,1.5L,16.9OZ|BOTTLED WATER;SPRING WATER;PURIFIED WATER;NATURAL SPRING WATER
Energy Drink|beverages|RED BULL,MONSTER,CELSIUS,ROCKSTAR|8.4OZ,16OZ,4PK,250ML|ENERGY DRINK;SUGAR FREE ENERGY DRINK;ENERGY DRINK ORIGINAL
Sports Drink|beverages|GATORADE,POWERADE,BODYARMOR|28OZ,8PK,20OZ|SPORTS DRINK;THIRST QUENCHER;ELECTROLYTE DRINK
Beer|alcohol|BUDWEISER,CORONA,HEINEKEN,MILLER LITE,STELLA ARTOIS|6PK,12PK,24PK,12OZ|BEER;LAGER;LIGHT BEER;IPA;PALE ALE;EXTRA BEER
Red Wine|alcohol|BAREFOOT,YELLOW TAIL,JOSH CELLARS,19 CRIMES|750ML,1.5L|RED WINE;CABERNET SAUVIGNON;MERLOT;PINOT NOIR;RED BLEND
White Wine|alcohol|BAREFOOT,KENDALL JACKSON,KIM CRAWFORD|750ML,1.5L|WHITE WINE;CHARDONNAY;SAUVIGNON BLANC;PINOT GRIGIO
Frozen Pizza|frozen|DIGIORNO,RED BARON,TOTINOS,NEWMANS OWN|12IN,28OZ,10.7OZ|FROZEN PIZZA;PEPPERONI PIZZA;CHEESE PIZZA RISING CRUST;SUPREME PIZZA
Ice Cream|frozen|BEN & JERRYS,HAAGEN-DAZS,BREYERS,TILLAMOOK|1PT,1.5QT,48OZ|ICE CREAM;VANILLA ICE CREAM;CHOCOLATE ICE CREAM;COOKIE DOUGH ICE CREAM
Frozen Vegetables|frozen|BIRDS EYE,GREEN GIANT,-|12OZ,16OZ,1KG|FROZEN VEGETABLES;MIXED VEGETABLES FROZEN;STEAMFRESH BROCCOLI;FROZEN PEAS;FROZEN CORN
French Fries|frozen|ORE-IDA,MCCAIN,-|32OZ,1KG|FRENCH FRIES;CRINKLE CUT FRIES;SHOESTRING FRIES;GOLDEN FRIES
Toilet Paper|household|CHARMIN,COTTONELLE,SCOTT,ANGEL SOFT|12RL,24RL,6 ROLLS,18 MEGA|TOILET PAPER;BATH TISSUE;ULTRA SOFT TOILET PAPER;MEGA ROLL BATH TISSUE
Paper Towels|household|BOUNTY,VIVA,SPARKLE,-|6RL,8RL,12 ROLLS|PAPER TOWELS;SELECT A SIZE PAPER TOWELS;PAPER TOWEL DOUBLE ROLLS
Dish Soap|household|DAWN,PALMOLIVE,SEVENTH GENERATION,METHOD|19.4OZ,28OZ,500ML|DISH SOAP;DISHWASHING LIQUID;ULTRA DISH SOAP;DISH LIQUID LEMON
Laundry Detergent|household|TIDE,PERSIL,GAIN,ARM & HAMMER,ALL|92OZ,150OZ,42CT,64LD|LAUNDRY DETERGENT;LIQUID DETERGENT;DETERGENT PODS;HE LAUNDRY DETERGENT
Trash Bags|household|GLAD,HEFTY,-|13GAL,40CT,30GAL,80CT|TRASH BAGS;KITCHEN TRASH BAGS;GARBAGE BAGS DRAWSTRING
Aluminum Foil|household|REYNOLDS,-|75SQFT,200SQFT|ALUMINUM FOIL;FOIL WRAP;HEAVY DUTY FOIL
Toothpaste|personal care|COLGATE,CREST,SENSODYNE,TOMS|4OZ,6OZ,2PK,100ML|TOOTHPASTE;WHITENING TOOTHPASTE;FLUORIDE TOOTHPASTE;CAVITY PROTECTION TOOTHPASTE
Shampoo|personal care|PANTENE,HEAD & SHOULDERS,DOVE,TRESEMME,SUAVE|12OZ,400ML,28OZ|SHAMPOO;DAILY MOISTURE SHAMPOO;DANDRUFF SHAMPOO;CLARIFYING SHAMPOO
Body Wash|personal care|DOVE,OLD SPICE,DIAL,IRISH SPRING|22OZ,16OZ,500ML|BODY WASH;MOISTURIZING BODY WASH;SHOWER GEL
Deodorant|personal care|DEGREE,OLD SPICE,DOVE,SECRET,NATIVE|2.6OZ,2PK,50ML|DEODORANT;ANTIPERSPIRANT;DEODORANT STICK;INVISIBLE SOLID ANTIPERSPIRANT
Pain Reliever|pharmacy|ADVIL,TYLENOL,ALEVE,MOTRIN,-|100CT,50CT,24CT,200MG|IBUPROFEN;ACETAMINOPHEN;PAIN RELIEVER;IBUPROFEN TABLETS 200MG;EXTRA STRENGTH TABLETS
Vitamins|pharmacy|NATURE MADE,CENTRUM,ONE A DAY,-|60CT,100CT,90CT|MULTIVITAMIN;VITAMIN C;VITAMIN D3;DAILY MULTIVITAMIN TABLETS;GUMMY VITAMINS
Bandages|pharmacy|BAND-AID,CURAD,-|30CT,60CT,ASST|BANDAGES;ADHESIVE BANDAGES;FLEXIBLE FABRIC BANDAGES;ASSORTED BANDAGES
Dog Food|pet|PURINA,PEDIGREE,BLUE BUFFALO,IAMS|30LB,15LB,13OZ,6KG|DOG FOOD;DRY DOG FOOD;ADULT DOG FOOD CHICKEN;WET DOG FOOD
Cat Food|pet|FRISKIES,FANCY FEAST,MEOW MIX,PURINA|3OZ,16LB,24CT|CAT FOOD;DRY CAT FOOD;WET CAT FOOD PATE;INDOOR CAT FOOD
Cat Litter|pet|TIDY CATS,ARM & HAMMER,FRESH STEP|20LB,40LB,14LB|CAT LITTER;CLUMPING CAT LITTER;CLUMPING LITTER MULTI CAT
"""

# Curated receipt abbreviations seen on real POS systems.
ABBREV = {
    "CHOCOLATE": ["CHOC", "CHOC", "CHC", "CHOCO", "CHOCLT"],
    "MILK": ["MLK", "MILK", "MK"], "WHOLE": ["WHL", "WHOLE", "WH"],
    "ORGANIC": ["ORG", "ORGNC", "OG"], "BUTTER": ["BTR", "BUTTR"],
    "CHEESE": ["CHS", "CHSE", "CHEES"], "CHEDDAR": ["CHED", "CHDR", "CHEDR"],
    "MOZZARELLA": ["MOZZ", "MOZZ", "MOZ", "MOZZRLA"], "PARMESAN": ["PARM", "PARMSN"],
    "YOGURT": ["YOG", "YGRT", "YGT", "YOGRT"], "GREEK": ["GRK", "GRK", "GREEK"],
    "CREAM": ["CRM", "CRM", "CRMY"], "HEAVY": ["HVY"], "WHIPPING": ["WHIP", "WHPG"],
    "LARGE": ["LG", "LRG", "LGE"], "EGGS": ["EGG", "EGGS"], "BREAD": ["BRD", "BRD"],
    "WHEAT": ["WHT", "WHEAT"], "WHITE": ["WHT", "WHITE", "WH"],
    "BAGELS": ["BAGEL", "BGLS"], "TORTILLAS": ["TORT", "TORTLA", "TORTS"],
    "BANANAS": ["BANANA", "BAN", "BNNA"], "APPLES": ["APPLE", "APPL"],
    "TOMATOES": ["TOM", "TOMATO", "TOMS"], "POTATOES": ["POT", "POTATO", "POTS"],
    "ONIONS": ["ONION", "ONI"], "SPINACH": ["SPIN", "SPNCH"],
    "LETTUCE": ["LETT", "LTCE"], "ROMAINE": ["ROM", "ROMN"],
    "CARROTS": ["CARROT", "CRT", "CARR"], "STRAWBERRIES": ["STRAWB", "STRWBRY", "STRAWBRY"],
    "BLUEBERRIES": ["BLUEB", "BLUBRY", "BLUEBRY"], "CHICKEN": ["CHKN", "CHKN", "CHIX", "CHCKN"],
    "BREAST": ["BRST", "BRST", "BRT"], "BONELESS": ["BNLS", "BNLS", "B/L"],
    "SKINLESS": ["SKNLS", "S/L"], "THIGHS": ["THGH", "THIGH"],
    "GROUND": ["GRD", "GRND", "GR"], "BEEF": ["BF", "BEEF"], "STEAK": ["STK"],
    "SAUSAGE": ["SAUS", "SSG"], "SAUSAGES": ["SAUS", "SSGS"], "BACON": ["BCN"],
    "SALMON": ["SALM", "SLMN"], "SHRIMP": ["SHRMP", "SHRP"], "TUNA": ["TUNA", "TNA"],
    "SPAGHETTI": ["SPAG", "SPAGH", "SPGHTI"], "PASTA": ["PST", "PSTA"],
    "RICE": ["RCE", "RICE"], "SAUCE": ["SCE", "SAUC", "SC"], "MARINARA": ["MARIN", "MRNRA"],
    "KETCHUP": ["KETCH", "KTCHP"], "MAYONNAISE": ["MAYO", "MAYO", "MAYONN"],
    "MUSTARD": ["MUST", "MSTRD"], "PEANUT": ["PNUT", "PNT", "P/NUT"],
    "OLIVE": ["OLV", "OLIV"], "OIL": ["OIL", "OL"], "EXTRA": ["XTRA", "EX", "X"],
    "VIRGIN": ["VRGN", "VIRG"], "VEGETABLE": ["VEG", "VEGE", "VEGTBL"],
    "VEGETABLES": ["VEG", "VEGS", "VEGGIES"], "FLOUR": ["FLR"],
    "PURPOSE": ["PURP", "PRPS"], "SUGAR": ["SUG", "SUGR"],
    "GRANULATED": ["GRAN", "GRANLTD"], "PEPPER": ["PEPP", "PPR"],
    "CEREAL": ["CRL", "CEREAL"], "OATS": ["OAT", "OATS"],
    "GRANOLA": ["GRAN", "GRNLA"], "BARS": ["BR", "BARS"], "BAR": ["BR", "BAR"],
    "POTATO": ["POT", "POTATO", "PTO"], "CHIPS": ["CHP", "CHIP", "CHPS"],
    "TORTILLA": ["TORT", "TORTLA"], "CRACKERS": ["CRKR", "CRCKR", "CRKRS"],
    "COOKIES": ["COOKIE", "CKIES", "CKY"], "POPCORN": ["POPCRN", "PPCRN"],
    "ALMONDS": ["ALMND", "ALMD"], "COFFEE": ["COF", "COFF", "CFE"],
    "BEAN": ["BN", "BEAN"], "ROAST": ["RST", "RST"], "MEDIUM": ["MED", "MD"],
    "JUICE": ["JCE", "JUIC", "JC"], "ORANGE": ["ORNG", "OR", "ORG"],
    "SPARKLING": ["SPRKLG", "SPARK", "SPKLG"], "WATER": ["WTR", "WATR"],
    "MINERAL": ["MIN", "MINRL"], "SPRING": ["SPRG", "SPR"], "ENERGY": ["ENRGY", "ENGY"],
    "DRINK": ["DRK", "DRNK"], "SODA": ["SODA", "SDA"], "WINE": ["WN", "WINE"],
    "CABERNET": ["CAB", "CABERNT"], "SAUVIGNON": ["SAUV", "SAUVG"],
    "CHARDONNAY": ["CHARD", "CHARDNY"], "FROZEN": ["FRZN", "FRZ", "FZN"],
    "PIZZA": ["PIZ", "PZA", "PIZZ"], "PEPPERONI": ["PEPP", "PEPRONI", "PEP"],
    "VANILLA": ["VAN", "VANL", "VNLA"], "TOILET": ["TLT", "TOIL"],
    "PAPER": ["PPR", "PAPR"], "TOWELS": ["TWL", "TWLS", "TOWL"],
    "TISSUE": ["TISS", "TSSU"], "DISH": ["DSH", "DISH"], "SOAP": ["SP", "SOAP"],
    "LAUNDRY": ["LNDRY", "LAUND"], "DETERGENT": ["DET", "DETRG", "DTRGNT"],
    "LIQUID": ["LIQ", "LQD"], "TRASH": ["TRSH", "TRASH"], "BAGS": ["BG", "BGS", "BAGS"],
    "KITCHEN": ["KIT", "KTCHN"], "ALUMINUM": ["ALUM", "ALUMN"],
    "TOOTHPASTE": ["TP", "TOOTHPST", "TTHPST"], "WHITENING": ["WHTNG", "WHITEN"],
    "SHAMPOO": ["SHAMP", "SHMP", "SHPO"], "BODY": ["BDY", "BODY"], "WASH": ["WSH", "WASH"],
    "DEODORANT": ["DEOD", "DEO", "DEODRNT"], "ANTIPERSPIRANT": ["ANTIPERS", "AP"],
    "IBUPROFEN": ["IBUPRO", "IBU", "IBPRFN"], "ACETAMINOPHEN": ["ACETAMIN", "APAP", "ACET"],
    "TABLETS": ["TABS", "TAB", "TBLT"], "STRENGTH": ["STR", "STRGTH"],
    "MULTIVITAMIN": ["MULTIVIT", "MULTI VIT", "MVI"], "VITAMIN": ["VIT", "VITMN"],
    "BANDAGES": ["BNDG", "BAND", "BANDG"], "ADHESIVE": ["ADH", "ADHSV"],
    "ASSORTED": ["ASST", "ASSTD", "AST"], "FOOD": ["FD", "FOOD"],
    "LITTER": ["LTR", "LITR"], "CLUMPING": ["CLMP", "CLUMP"],
    "UNSWEETENED": ["UNSWT", "UNSWTND", "UNSW"], "UNSALTED": ["UNSLTD", "UNSALT"],
    "SALTED": ["SLTD", "SALT"], "SHREDDED": ["SHRD", "SHRED", "SHRDD"],
    "SLICED": ["SLCD", "SLI", "SLC"], "SMOKED": ["SMKD", "SMK"],
    "ORIGINAL": ["ORIG", "ORGNL", "OG"], "CLASSIC": ["CLSC", "CLASS"],
    "NATURAL": ["NAT", "NATRL"], "FRESH": ["FRSH", "FR"], "ROASTED": ["RSTD", "ROAST"],
    "STRAWBERRY": ["STRAWB", "STRWBRY"], "RASPBERRY": ["RASPB", "RSPBRY"],
    "BLACK": ["BLK", "BLK"], "GREEN": ["GRN", "GRN"], "BROWN": ["BRN", "BRWN"],
    "RED": ["RD", "RED"], "YELLOW": ["YEL", "YLW"], "SWEET": ["SWT", "SWEET"],
    "CHERRY": ["CHRY", "CHER"], "LEMON": ["LMN", "LEM"], "LIME": ["LM", "LIME"],
    "PREMIUM": ["PREM", "PRM"], "REGULAR": ["REG", "RGLR"], "REDUCED": ["RED", "RDCD"],
    "FAT": ["FT", "FAT"], "FREE": ["FR", "FREE"], "LIGHT": ["LT", "LITE", "LGT"],
    "ULTRA": ["ULT", "ULTR"], "INSTANT": ["INST", "INSTNT"],
    "MICROWAVE": ["MICRO", "MW", "MCRWV"], "TRADITIONAL": ["TRAD", "TRADTNL"],
    "HOMOGENIZED": ["HOMO", "HOMOG"], "BEVERAGE": ["BEV", "BEVG"],
    "TRUFFLES": ["TRUFF", "TRFL"], "ASSORTMENT": ["ASST"], "GUMMY": ["GUM", "GMMY"],
    "CANDY": ["CNDY", "CAND"], "CHEWING": ["CHEW", "CHWG"], "SPEARMINT": ["SPRMNT", "SPEARMT"],
    "PEPPERMINT": ["PPRMNT", "PEPPERMT"], "CROISSANTS": ["CROISS", "CRSNT"],
    "AVOCADO": ["AVO", "AVOC"], "HASS": ["HASS"], "HONEYCRISP": ["HNYCRSP", "HONEYCR"],
    "STICKS": ["STK", "STKS"], "PACK": ["PK"], "COUNT": ["CT"],
}

# Store / private-label brand codes that print before the product name.
STORE_BRANDS = ["GV", "KS", "365", "TJ", "SIG", "SB", "GG", "KRO", "MM", "PL",
                "FRESH MART", "VALU", "HARVEST", "SUNRISE"]

FILLERS_PRE = ["", "", "", "", "", "", "F ", "N ", "T ", "* ", "#"]
# Tax / department flags POS systems print after the item name.
FILLERS_POST = ["", "", "", "", "", "", " N", " F", " T", " X", " TF", " FT", " B",
                " EA", " NF"]


def parse_catalogue():
    products = []
    for line in CATALOGUE.strip().splitlines():
        target, category, brands, sizes, variants = line.split("|")
        products.append({
            "target": target,
            "category": category,
            "brands": [b.strip() for b in brands.split(",")],
            "sizes": [s.strip() for s in sizes.split(",")],
            "variants": [v.strip() for v in variants.split(";")],
        })
    return products


# --------------------------------------------------------------------------
# Raw receipt-string synthesis
# --------------------------------------------------------------------------

VOWELS = set("AEIOU")


def auto_abbrev(word, rng):
    """Generic POS-style abbreviation for words missing from ABBREV."""
    if len(word) <= 4 or not word.isalpha():
        return word
    mode = rng.random()
    if mode < 0.45:  # drop interior vowels: SPARKLING -> SPRKLNG
        out = word[0] + "".join(c for c in word[1:] if c not in VOWELS)
        return out if len(out) >= 3 else word
    if mode < 0.85:  # truncate: GRANOLA -> GRAN
        return word[:rng.randint(3, min(6, len(word) - 1))]
    return word


def abbreviate_word(word, rng, strength):
    if rng.random() > strength:
        return word
    if word in ABBREV and rng.random() < 0.85:
        return rng.choice(ABBREV[word])
    return auto_abbrev(word, rng)


def abbreviate_brand(brand, rng, strength):
    if rng.random() > strength * 0.6:
        return brand
    words = brand.replace("'", "").split()
    mode = rng.random()
    if len(words) > 1 and mode < 0.4:  # initials: GREEN & BLACKS -> G&B
        return "".join(w[0] for w in words if w[0].isalnum() or w == "&")
    if mode < 0.7:  # keep first word only
        return words[0]
    return " ".join(auto_abbrev(w, rng) for w in words)


def format_size(size, rng):
    s = size
    if rng.random() < 0.3:  # 100G -> 100 G
        s = re.sub(r"(\d)([A-Z])", r"\1 \2", s, count=1)
    if rng.random() < 0.2:  # unit spelling variants
        s = re.sub(r"OZ$", rng.choice(["Z", "OZ.", "OZ"]), s)
        s = re.sub(r"(\d ?)G$", lambda m: m.group(1) + rng.choice(["G", "GM", "GR"]), s)
    return s


def make_raw(product, rng, allowed_brands, max_width):
    """Build one raw receipt string for a product. Returns (raw, brand, size)."""
    strength = rng.choice([0.0, 0.3, 0.6, 0.85, 1.0])
    variant = rng.choice(product["variants"])
    name = " ".join(abbreviate_word(w, rng, strength) for w in variant.split())

    brands = [b for b in product["brands"] if b in allowed_brands or b == "-"]
    brand = rng.choice(brands) if brands else "-"
    if brand == "-":
        brand_text = rng.choice(STORE_BRANDS) if rng.random() < 0.5 else ""
        brand = brand_text or None
    else:
        brand_text = abbreviate_brand(brand, rng, strength)

    size = rng.choice(product["sizes"]) if rng.random() < 0.75 else None
    size_text = format_size(size, rng) if size else ""

    order = rng.random()
    if order < 0.75:
        parts = [brand_text, name, size_text]
    elif order < 0.9:
        parts = [brand_text, size_text, name]
    else:
        parts = [name, brand_text, size_text]
    raw = " ".join(p for p in parts if p)

    if rng.random() < 0.08:
        raw = "ORG " + raw if "ORG" not in raw else raw
    raw = rng.choice(FILLERS_PRE) + raw + rng.choice(FILLERS_POST)
    if rng.random() < 0.1:  # PLU / SKU code before the name
        raw = f"{rng.randint(1000, 99999)} {raw}"

    # POS item-name fields are fixed width; only cut off trailing size/brand
    # text, never the product name itself, so the target stays recoverable.
    if len(raw) > max_width:
        name_end = raw.find(name) + len(name) if name in raw else len(raw)
        cut = max(max_width, name_end)
        raw = raw[:cut].rstrip()
    return raw.strip(), brand, size


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

def build_split(n, rng, products, allowed_brands, noise_weights, seen):
    rows = []
    attempts = 0
    levels = list(noise_weights)
    weights = list(noise_weights.values())
    while len(rows) < n and attempts < n * 50:
        attempts += 1
        product = rng.choice(products)
        max_width = rng.choice([18, 20, 22, 24, 28, 32, 40])
        raw, brand, size = make_raw(product, rng, allowed_brands, max_width)

        level = rng.choices(levels, weights=weights)[0]
        if level != "none":
            raw = corrupt_line(raw, rng, LEVELS[level]).strip()
        if not raw or raw in seen:  # no duplicate inputs across splits
            continue
        seen.add(raw)
        rows.append({
            "input": raw,
            "target": product["target"],
            "category": product["category"],
            "brand": brand,
            "size": size,
            "ocr_noise": level,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_items")
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", default="none:0.6,light:0.3,medium:0.1",
                    help="OCR noise mix on top of abbreviations "
                         "(levels: none, light, medium, heavy)")
    ap.add_argument("--holdout-brands", type=float, default=0.0,
                    help="fraction of brands that appear only in val/test, "
                         "to measure generalisation to unseen brands")
    ap.add_argument("--target-case", choices=["title", "upper", "lower"],
                    default="title")
    args = ap.parse_args()

    noise_weights = {}
    for part in args.noise.split(","):
        name, w = part.split(":")
        if name != "none" and name not in LEVELS:
            ap.error(f"unknown noise level {name!r}")
        noise_weights[name] = float(w)

    products = parse_catalogue()
    all_brands = sorted({b for p in products for b in p["brands"] if b != "-"})
    rng = random.Random(args.seed)
    held = set(rng.sample(all_brands, int(len(all_brands) * args.holdout_brands)))
    train_brands = set(all_brands) - held

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = set()
    for split, n in [("train", args.train), ("val", args.val), ("test", args.test)]:
        split_rng = random.Random(f"{args.seed}-{split}")
        allowed = train_brands if split == "train" else set(all_brands)
        rows = build_split(n, split_rng, products, allowed, noise_weights, seen)
        for i, r in enumerate(rows):
            r["id"] = f"{split}-{i:06d}"
            if args.target_case == "upper":
                r["target"] = r["target"].upper()
            elif args.target_case == "lower":
                r["target"] = r["target"].lower()
            r["unseen_brand"] = r["brand"] in held
        rows = [{"id": r.pop("id"), **r} for r in rows]
        count = write_jsonl(out / f"{split}.jsonl", rows)
        print(f"wrote {count:>6} examples -> {out / f'{split}.jsonl'}")
        if count < n:
            print(f"  note: only {count} unique inputs found for {split}")

    labels = sorted({p["target"] for p in products})
    (out / "labels.txt").write_text("\n".join(labels) + "\n")
    print(f"wrote {len(labels):>6} labels   -> {out / 'labels.txt'}")


if __name__ == "__main__":
    main()
