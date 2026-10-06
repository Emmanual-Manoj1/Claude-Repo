#!/usr/bin/env python3
"""Synthetic dataset for normalising receipt item names to generic products.

Turns raw OCR'd receipt lines into clean item names:

    MLIGHT STRAWBRY 160G £O 40   ->  Muller Light Strawberry   (--target name)
    TUNA & SWEETCORN SW 2 05     ->  Tuna & Sweetcorn Sandwich
    LINDT DARK CHOC 100G 3.49    ->  Lindt Dark Chocolate
    13 Sub Total- 2245           ->  Non-Item

or, with --target generic, into a product type (Yogurt, Sandwich, Dark
Chocolate). Every row carries both, as "name" and "generic".

The model learns to drop prices, sizes, pack counts and store codes, expand
receipt abbreviations, see through OCR errors, and flag lines that are not
products (headers, totals, promotions) as "Non-Item".

Products live in catalogue.txt (US) and catalogue_uk.txt (UK). Output is
JSONL, plus chat-style JSONL for LLM fine-tuning and CSV.

Usage:
    python generate_items.py --locale uk              # UK receipts, item names
    python generate_items.py --target generic         # product types instead
    python generate_items.py --holdout-brands 0.15    # test on unseen brands
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from generate import LEVELS, corrupt_line, write_jsonl

HERE = Path(__file__).parent

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

# Abbreviations for the wider catalogue.
ABBREV.update({
    "CARAMEL": ["CARML", "CRML"], "LICORICE": ["LICOR", "LIC"], "MINTS": ["MNTS"],
    "MARSHMALLOWS": ["MRSHMLW", "MARSH", "MLLW"], "HAZELNUT": ["HZLNT", "HAZ"],
    "SPREAD": ["SPRD"], "CREAMER": ["CRMR", "CREAMR"], "COTTAGE": ["COTT", "CTG"],
    "SWISS": ["SWS"], "CRUMBLED": ["CRMBL", "CRUMB"], "SINGLES": ["SNGL"],
    "PROVOLONE": ["PROV", "PROVO"], "RICOTTA": ["RICOT", "RCTA"],
    "MARGARINE": ["MARG", "MRGRN"], "WHIPPED": ["WHPD", "WHIP"],
    "HAMBURGER": ["HAMB", "HMBRGR"], "BUNS": ["BUN", "BNS"], "ROLLS": ["RLS", "ROLL"],
    "SOURDOUGH": ["SRDGH", "SOURD"], "BAGUETTE": ["BAGT", "BAGUET"],
    "ENGLISH": ["ENG", "ENGL"], "MUFFINS": ["MUFF", "MFN"], "DONUTS": ["DNTS", "DONUT"],
    "ORANGES": ["ORNG", "ORANGE"], "SEEDLESS": ["SDLS", "SDLSS"], "GRAPES": ["GRP", "GRAPE"],
    "WATERMELON": ["WTRMLN", "WMELON"], "PINEAPPLE": ["PINE", "PNAPL"],
    "RASPBERRIES": ["RASPB", "RSPBRY"], "BROCCOLI": ["BROC", "BRCLI"],
    "CAULIFLOWER": ["CAULI", "CLFLWR"], "CUCUMBER": ["CUKE", "CUC", "CUCMBR"],
    "PEPPERS": ["PEPP", "PPRS"], "MUSHROOMS": ["MUSH", "MSHRM"],
    "ZUCCHINI": ["ZUCC", "ZUCH"], "ASPARAGUS": ["ASPAR", "ASP"],
    "PORK": ["PRK"], "CHOPS": ["CHP", "CHPS"], "TENDERLOIN": ["TNDRLN", "TENDER"],
    "TURKEY": ["TRKY", "TURK"], "WINGS": ["WNGS", "WING"], "FRANKS": ["FRNK", "FRK"],
    "ROASTED": ["RSTD", "RST"], "SALAMI": ["SLMI"], "PEPPERONI": ["PEP", "PEPRONI"],
    "ROTISSERIE": ["ROTIS", "ROT", "RTSR"], "FIRM": ["FRM"], "BURGER": ["BRGR", "BURG"],
    "FILLET": ["FLT", "FIL"], "FILLETS": ["FLTS", "FIL"], "SARDINES": ["SARD", "SRDN"],
    "MACARONI": ["MAC", "MACR"], "NOODLES": ["NDL", "NOOD", "NDLS"],
    "LASAGNA": ["LASAG", "LSGNA"], "DICED": ["DCD", "DICE"], "CRUSHED": ["CRSHD", "CRUSH"],
    "BEANS": ["BNS", "BEAN"], "KIDNEY": ["KIDNY", "KDNY"], "CHICKPEAS": ["CHKPEA", "CHKPS"],
    "SOUP": ["SP", "SOUP"], "CONDENSED": ["COND", "CNDNSD"], "BROTH": ["BRTH", "BROTH"],
    "SALSA": ["SLSA"], "BARBECUE": ["BBQ", "BARBQ"], "DRESSING": ["DRSG", "DRESS", "DRS"],
    "RANCH": ["RNCH"], "VINEGAR": ["VIN", "VINEG", "VNGR"], "BALSAMIC": ["BALS", "BLSMC"],
    "PICKLES": ["PCKL", "PICKL"], "OLIVES": ["OLV", "OLVS"], "SYRUP": ["SYR", "SYRP"],
    "PANCAKE": ["PNCK", "PANCK"], "BAKING": ["BKG", "BAKG"], "POWDER": ["PWD", "PWDR"],
    "SODA": ["SODA", "SDA"], "EXTRACT": ["EXT", "EXTR"], "CINNAMON": ["CINN", "CIN"],
    "SEASONING": ["SEAS", "SSNG"], "BREADCRUMBS": ["BRDCRMB"], "CRUMBS": ["CRMB", "CRMBS"],
    "QUINOA": ["QUIN", "QNOA"], "LENTILS": ["LENT", "LNTL"], "DOUGH": ["DGH", "DOUGH"],
    "CRESCENT": ["CRES", "CRSCNT"], "BISCUITS": ["BISC", "BSCT"],
    "RAISINS": ["RAIS", "RSN"], "CRANBERRIES": ["CRAN", "CRNBRY"], "DRIED": ["DRD", "DRY"],
    "PEANUTS": ["PNUTS", "PNTS"], "CASHEWS": ["CASH", "CSHW"], "PISTACHIOS": ["PIST", "PSTCH"],
    "PRETZELS": ["PRTZL", "PRETZ"], "JERKY": ["JRKY", "JERK"], "HUMMUS": ["HUMM", "HMMS"],
    "GUACAMOLE": ["GUAC", "GUACA"], "APPLESAUCE": ["APLSCE", "APPLSC"],
    "PUDDING": ["PUDD", "PDNG"], "PROTEIN": ["PROT", "PRTN"], "LEMONADE": ["LMNADE", "LEMONAD"],
    "CRANBERRY": ["CRAN", "CRNBRY"], "GINGER": ["GING", "GNGR"], "KOMBUCHA": ["KOMB", "KMBCHA"],
    "COCOA": ["COCO", "CCOA"], "COCONUT": ["COCO", "CCNT"], "SMOOTHIE": ["SMTHY", "SMOOTH"],
    "SELTZER": ["SLTZR", "SELTZ"], "CIDER": ["CIDR", "CDR"], "PROSECCO": ["PROS", "PRSCO"],
    "CHAMPAGNE": ["CHAMP", "CHMPGN"], "VODKA": ["VDKA", "VOD"], "WHISKEY": ["WHSKY", "WHSK"],
    "BOURBON": ["BRBN", "BOURB"], "TEQUILA": ["TEQ", "TEQL"], "DINNER": ["DNR", "DINR"],
    "ENTREE": ["ENTR", "ENT"], "BURRITOS": ["BURR", "BRTO"], "WAFFLES": ["WAFF", "WFL"],
    "NUGGETS": ["NUGG", "NGTS"], "BREADED": ["BRDD", "BRD"], "DUMPLINGS": ["DMPLNG", "DUMP"],
    "POTSTICKERS": ["POTSTK", "PTSTKR"], "NAPKINS": ["NAPK", "NPKN"],
    "FACIAL": ["FCL", "FAC"], "TISSUES": ["TISS", "TSS"], "PLASTIC": ["PLST", "PLAS"],
    "WRAP": ["WRP"], "STORAGE": ["STOR", "STRG"], "GALLON": ["GAL", "GLN"],
    "SANDWICH": ["SAND", "SNDWCH", "SW"], "DISHWASHER": ["DSHWSHR", "DISHW", "DW"],
    "FABRIC": ["FAB", "FBRC"], "SOFTENER": ["SFTNR", "SOFT"], "DRYER": ["DRYR", "DRY"],
    "SHEETS": ["SHTS", "SHT"], "BLEACH": ["BLCH"], "CLEANER": ["CLNR", "CLEAN"],
    "DISINFECTING": ["DISINF", "DSNFCT"], "WIPES": ["WIPE", "WPS"], "SPONGES": ["SPNG", "SPONG"],
    "BULBS": ["BLB", "BULB"], "BATTERIES": ["BATT", "BTRY"], "ALKALINE": ["ALK", "ALKLN"],
    "PLATES": ["PLT", "PLTS"], "CUPS": ["CUP", "CPS"], "FRESHENER": ["FRSHNR", "FRESH"],
    "CHARCOAL": ["CHARC", "CHRCL"], "CONDITIONER": ["COND", "CNDTNR"], "HAND": ["HND"],
    "TOOTHBRUSH": ["TOOTHBR", "TBRUSH", "TB"], "MOUTHWASH": ["MTHWSH", "MOUTHW"],
    "FLOSS": ["FLS"], "RAZOR": ["RZR"], "SHAVING": ["SHAV", "SHV"], "LOTION": ["LOT", "LTN"],
    "SUNSCREEN": ["SUNSCR", "SNSCRN"], "SWABS": ["SWB", "SWAB"], "DIAPERS": ["DIAP", "DPR"],
    "BABY": ["BBY", "BB"], "FORMULA": ["FORM", "FRML"], "INFANT": ["INF", "INFNT"],
    "MEDICINE": ["MED", "MEDS"], "RELIEF": ["RLF", "REL"], "ALLERGY": ["ALRGY", "ALLRG"],
    "ANTACID": ["ANTAC", "ANTCD"], "COUGH": ["CGH", "COF"], "DROPS": ["DRP", "DRPS"],
    "OINTMENT": ["OINT", "ONTMNT"], "ANTIBIOTIC": ["ANTIB", "ABX"], "MELATONIN": ["MELAT", "MLTN"],
    "THERMOMETER": ["THERM", "THRMTR"], "TREATS": ["TRT", "TRTS"], "BISCUIT": ["BISC"],
    "SALAD": ["SLD", "SAL"], "SUSHI": ["SUSH"], "FLOWERS": ["FLWR", "FLWRS"],
    "BOUQUET": ["BQT", "BOUQ"], "GREETING": ["GRTG", "GREET"], "MAGAZINE": ["MAG", "MAGZ"],
    "MOTOR": ["MTR"], "SYNTHETIC": ["SYN", "SYNTH"], "WINDSHIELD": ["WNDSHLD", "WSHLD"],
    "WASHER": ["WSHR"], "FLUID": ["FLD"], "UNLEADED": ["UNL", "UNLD"], "GASOLINE": ["GAS"],
})

# Abbreviations common on UK receipts.
ABBREV.update({
    "SANDWICH": ["SW", "SW", "SW", "SAND", "SNDWCH"], "SUMMER": ["SUM", "SMR"],
    "FRUITS": ["FRT", "FRTS", "FRUIT"], "FRUIT": ["FRT", "FRUIT"],
    "BLACKCURRANT": ["BLK", "BLKCRNT", "BLKCURR"], "ADVENTURE": ["AD", "ADV", "ADVNTR"],
    "SWEETCORN": ["SWTCRN", "SWCORN"], "MATURE": ["MAT", "MATR"],
    "SEMI": ["S/", "SEMI"], "SKIMMED": ["SKIM", "SKMD"], "SMOKED": ["SMK", "SMKD"],
    "UNSMOKED": ["UNSMK", "UNSMKD"], "STREAKY": ["STRKY"], "SAUSAGES": ["SSGS", "SAUS"],
    "SAUSAGE": ["SSG", "SAUS"], "HOUMOUS": ["HOUMS", "HMS"], "WHOLEMEAL": ["WHLML", "W/MEAL"],
    "SEEDED": ["SDD"], "FREE": ["FR", "FREE"], "RANGE": ["RNG", "RANGE"],
    "MEDIUM": ["MED", "MDM"], "LARGE": ["LGE", "LRG"], "DIGESTIVES": ["DIG", "DIGS"],
    "BISCUITS": ["BISC", "BSCTS"], "CRUMPETS": ["CRMPTS"], "SQUASH": ["SQ", "SQSH"],
    "LEMONADE": ["LMNDE"], "STRAWBERRY": ["STRAWBRY", "STRAWB", "STRWBRY"],
    "RASPBERRY": ["RASPB", "RSPBRY"], "CHOCOLATE": ["CHOC", "CHC"],
    "BUTTERY": ["BUTTERY", "BTRY"], "SPREAD": ["SPRD", "SPREAD"],
    "PASSIONFRUIT": ["PASSION", "PSSNFRT"], "WASHING": ["WASH", "WSHG"],
    "CONDITIONER": ["COND", "CONDTNR"], "TOMATOES": ["TOMS", "TOM"],
})

# Brands printed inside a variant name, so the brand must not be printed again.
VARIANT_BRANDS = {
    "COKE": "COCA-COLA", "PEPSI": "PEPSI", "SPRITE": "SPRITE", "7UP": "7UP",
    "CHEERIOS": "GENERAL MILLS", "CORN FLAKES": "KELLOGGS", "FROSTED FLAKES": "KELLOGGS",
    "RAISIN BRAN": "POST", "LINDOR": "LINDT", "ROCHER": "FERRERO", "DAIRY MILK": "CADBURY",
    "GOLDBEARS": "HARIBO", "EXCELLENCE": "LINDT", "NOOKS": "THOMAS", "BEYOND": "BEYOND MEAT",
    "IMPOSSIBLE": "IMPOSSIBLE", "SHIN RAMYUN": "NONGSHIM", "CUP NOODLES": "NISSIN",
    "FANTA": "FANTA", "K-CUP": "KEURIG", "K CUPS": "KEURIG", "GOLD STANDARD": "OPTIMUM NUTRITION",
    "GREEN MACHINE": "NAKED", "MIGHTY MANGO": "NAKED", "STEAMFRESH": "BIRDS EYE",
    "SINGLES": "KRAFT", "GRANDS": "PILLSBURY", "TOTAL 0%": "FAGE", "DENTASTIX": "PEDIGREE",
    "DREAMIES": "WHISKAS", "BISTO": "BISTO", "THATCHERS GOLD": "THATCHERS",
    "WEETABIX": "WEETABIX", "OAT SO SIMPLE": "QUAKER", "DIET COKE": "COCA COLA",
}

NON_ITEM = "Non-Item"


# --------------------------------------------------------------------------
# Catalogue loading
# --------------------------------------------------------------------------

KEEP_UPPER = {"BBQ", "BLT", "IPA", "LED", "AA", "AAA", "9V", "SPF", "OK", "TV", "BBC",
              "UK", "DOC", "KP", "HP", "PG", "JS", "M&S", "G&T", "VO5", "KTC", "TTD",
              "XL", "USA", "RC", "GE", "A&W", "C&H", "J2O", "WKD", "FT", "OJ", "B&J"}


def display(text):
    """Readable item name: keep mixed-case text as written, else title-case."""
    if any(c.islower() for c in text):
        return text
    words = []
    for w in text.split():
        if w in KEEP_UPPER or (len(w) <= 3 and not set(w) & set("AEIOUY") and w.isalpha()):
            words.append(w)
        else:
            words.append(w[:1] + w[1:].lower())
    return " ".join(words)


def parse_named(field):
    """'Clean Name=PRINTED ONE/PRINTED TWO' -> (clean, [printed...])."""
    if "=" in field:
        clean, printed = field.split("=", 1)
        forms = [p.strip().upper() for p in printed.split("/") if p.strip()]
    else:
        clean, forms = field, [field.strip().upper()]
    return clean.strip(), forms or [clean.strip().upper()]


def parse_catalogue(path):
    """Load products, rejecting malformed rows and ambiguous item names."""
    products, owner = [], {}
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 '|' fields, got {len(fields)}")
        generic, category, brands, sizes, variants = (f.strip() for f in fields)
        brand_list = []
        for b in brands.split(","):
            b = b.strip()
            if b in ("-", "NONE"):
                brand_list.append(b)
            elif b:
                clean, printed = parse_named(b)
                brand_list.append({"clean": clean, "printed": printed})
        product = {
            "target": generic,
            "category": category,
            "brands": brand_list,
            "sizes": [s.strip() for s in sizes.split(",") if s.strip()],
            "variants": [dict(zip(("clean", "printed"), parse_named(v)))
                         for v in variants.split(";") if v.strip()],
        }
        for v in product["variants"]:
            key = v["clean"].upper()
            if owner.setdefault(key, generic) != generic:
                raise ValueError(f"{path}:{lineno}: item {v['clean']!r} is listed under "
                                 f"both {owner[key]!r} and {generic!r}")
        products.append(product)
    return products


def brand_key(brand):
    return brand["printed"][0] if isinstance(brand, dict) else brand


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
    if len(words) > 1 and mode < 0.3:  # initials: GREEN & BLACKS -> G&B
        return "".join(w[0] for w in words if w[0].isalnum() or w == "&")
    if len(words) > 1 and mode < 0.45:  # MULLER LIGHT -> MLIGHT, TIC TAC -> TICTAC
        return words[0][0] + "".join(words[1:]) if rng.random() < 0.5 else "".join(words)
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


def add_price_us(raw, product, rng):
    """Append the price columns that often get OCR'd onto the item line."""
    expensive = product["category"] in ("alcohol", "baby", "automotive")
    price = rng.uniform(5, 60) if expensive else rng.uniform(0.5, 25.0)
    mode = rng.random()
    if mode < 0.45:
        tail = f"{price:.2f}"
    elif mode < 0.65:
        tail = f"{price:.2f} {rng.choice(['F', 'N', 'T', 'TF', 'B', 'A'])}"
    elif mode < 0.8:
        qty = rng.randint(2, 6)
        tail = f"{qty} @ {price:.2f} {qty * price:.2f}"
    elif mode < 0.92 and product["category"] in ("produce", "meat", "seafood", "deli"):
        weight = rng.uniform(0.2, 4.0)
        unit = rng.choice(["LB", "KG"])
        tail = f"{weight:.2f} {unit} @ {price / 4:.2f}/{unit} {weight * price / 4:.2f}"
    else:
        tail = f"${price:.2f}"
    return raw + " " * rng.choice([1, 2, 3, 6, 10]) + tail


# How this OCR misreads the £ sign, from real scans: £4.50, E450, C0.80,
# L22.20, e1.30, A1 50, P22.45, 9160 (£1.60), 0155 (£1.55), 22.05 (£2.05).
POUND_MISREADS = ["£"] * 6 + [""] * 6 + ["E", "E", "C", "C", "L", "e", "A", "P", "9", "0", "2"]


def uk_amount(amount, rng):
    pounds, pence = divmod(round(amount * 100), 100)
    sep = rng.choice([".", ".", ".", " ", " ", "", ","])
    text = f"{pounds}{sep}{pence:02d}"
    if rng.random() < 0.25:  # 0 read as O: £O 90, O1.55
        text = text.replace("0", "O", rng.choice([1, 1, 2]))
    return rng.choice(POUND_MISREADS) + text


def add_price_uk(raw, product, rng):
    expensive = product["category"] in ("alcohol", "baby")
    amount = rng.uniform(4, 30) if expensive else rng.uniform(0.3, 8.0)
    return raw + " " * rng.choice([1, 1, 1, 2, 4]) + uk_amount(amount, rng)


def vary_case(raw, rng):
    """Most receipts print in upper case, but some POS systems don't."""
    r = rng.random()
    if r < 0.10:
        return raw.title()
    if r < 0.14:
        return raw.lower()
    return raw


def variant_brand(printed, product):
    """Return the product brand already named inside a printed variant."""
    words = f" {printed} "
    for alias, brand in VARIANT_BRANDS.items():
        if f" {alias} " in words:
            for b in product["brands"]:
                if isinstance(b, dict) and brand_key(b) == brand:
                    return b
            return {"clean": brand, "printed": [brand], "inline": True}
    for b in product["brands"]:
        if isinstance(b, dict) and f" {b['printed'][0].split()[0]} " in words:
            return b
    return None


def join_name(brand_clean, variant_clean):
    if not brand_clean:
        return display(variant_clean)
    b, v = display(brand_clean), display(variant_clean)
    return v if v.lower().startswith(b.lower()) else f"{b} {v}"


def make_raw(product, rng, allowed_brands, max_width, price_ratio, loc):
    """Build one raw receipt line. Returns (raw, item name, brand, size)."""
    strength = rng.choice([0.0, 0.3, 0.6, 0.85, 1.0])
    variant = rng.choice(product["variants"])
    printed = rng.choice(variant["printed"])
    name_text = " ".join(abbreviate_word(w, rng, strength) for w in printed.split())

    inline = variant_brand(printed, product)
    brands = [b for b in product["brands"]
              if not isinstance(b, dict) or brand_key(b) in allowed_brands]
    if inline is not None:
        brand, brand_text = inline, ""
    else:
        brand = rng.choice(brands) if brands else "NONE"
        if brand == "-":
            brand_text = rng.choice(loc["store_brands"]) if rng.random() < 0.5 else ""
        elif brand == "NONE":
            brand_text = ""
        else:
            brand_text = abbreviate_brand(rng.choice(brand["printed"]), rng, strength)
    brand_clean = brand["clean"] if isinstance(brand, dict) else ""

    # Some lines print only the brand: MONSTER MUNCH, TICTAC.
    brand_only = isinstance(brand, dict) and brand_text and not inline \
        and rng.random() < 0.08
    if brand_only:
        name_text = ""

    size = rng.choice(product["sizes"]) if product["sizes"] and rng.random() < 0.7 else None
    size_text = format_size(size, rng) if size else ""

    order = rng.random()
    if order < 0.8:
        parts = [brand_text, name_text, size_text]
    elif order < 0.9:
        parts = [brand_text, size_text, name_text]
    else:
        parts = [name_text, brand_text, size_text]
    raw = " ".join(p for p in parts if p)
    raw = rng.choice(loc["fillers_pre"]) + raw + rng.choice(loc["fillers_post"])
    if loc["plu_codes"] and rng.random() < 0.1:
        raw = f"{rng.randint(1000, 99999)} {raw}"
    if rng.random() < 0.06:  # multi-buy printed in front: 2 X Meal Deal
        raw = f"{rng.randint(2, 4)} {rng.choice(['X', 'x', '@'])} {raw}"

    # POS name fields are fixed width. Never cut into the product name; if
    # the brand gets cut off, it is dropped from the target too.
    protect = name_text or brand_text
    if len(raw) > max_width:
        end = raw.find(protect) + len(protect) if protect in raw else len(raw)
        raw = raw[:max(max_width, end)].rstrip()
    brand_visible = bool(brand_text) and brand_text in raw

    if brand_only:
        name = display(brand_clean)
    elif inline is not None:
        name = display(variant["clean"])
    elif isinstance(brand, dict) and brand_visible:
        name = join_name(brand_clean, variant["clean"])
    else:
        name = display(variant["clean"])

    if rng.random() < price_ratio:
        raw = loc["add_price"](raw, product, rng)
    brand_out = brand_clean or (brand_text if brand == "-" and brand_text else None)
    return vary_case(raw.strip(), rng), name, brand_out, size


# --------------------------------------------------------------------------
# Non-item lines
#
# Real receipt text also contains headers, totals, fees, coupons and payment
# lines. Teaching the model to label these Non-Item stops it inventing an
# item for "PROMOTIONS -£1.55" or "COUPON LINDT CHOC".
# --------------------------------------------------------------------------

US_NON_ITEMS = [
    "SUBTOTAL", "SUB TOTAL", "TOTAL", "TAX", "SALES TAX", "TAX 1", "TAX 8.25%",
    "BALANCE DUE", "AMOUNT DUE", "CHANGE DUE", "CHANGE", "CASH", "CASH TEND",
    "VISA", "MASTERCARD", "DEBIT", "AMEX TEND", "EBT", "GIFT CARD TEND",
    "BOTTLE DEPOSIT", "CRV", "BTL DEP", "CAN DEPOSIT", "BAG FEE", "PAPER BAG",
    "BAG CHARGE", "CARRYOUT BAG", "STORE COUPON", "MFR COUPON", "MANUFACTURER CPN",
    "DIGITAL COUPON", "LOYALTY SAVINGS", "MEMBER SAVINGS", "YOU SAVED", "INSTANT SAVINGS",
    "PROMO DISCOUNT", "PRICE OVERRIDE", "VOID", "ITEM VOIDED", "REFUND", "RETURN",
    "TOTAL SAVINGS", "NUMBER OF ITEMS", "ITEMS SOLD", "TIP", "GRATUITY",
    "SERVICE CHARGE", "DELIVERY FEE", "ROUNDING", "AUTH CODE", "APPROVED",
    "THANK YOU", "CASHIER", "REG", "TRAN", "STORE", "MEMBER #",
]
US_PRODUCT_PREFIXES = ["COUPON", "CPN", "MFR CPN", "SC", "DISC", "SAVINGS",
                       "REWARDS", "BOGO", "VOID", "RETURN", "DEP"]

UK_NON_ITEMS = [
    "SUB TOTAL", "SUBTOTAL", "BALANCE DUE", "TOTAL TO PAY", "TOTAL", "PROMOTIONS",
    "TOTAL SAVINGS", "MULTIBUY SAVINGS", "PRICE CUT", "NECTAR POINTS", "NECTAR PRICE SAVING",
    "CLUBCARD PRICE", "CLUBCARD POINTS", "CLUBCARD SAVINGS", "COLLEAGUE DISCOUNT",
    "MORE CARD", "ASDA REWARDS", "VISA DEBIT", "MASTERCARD", "CONTACTLESS", "CARD",
    "CASH", "CHANGE", "CHANGE DUE", "AMOUNT", "VAT", "VAT INCLUDED", "CARRIER BAG",
    "BAG FOR LIFE", "BAG CHARGE", "REFUND", "VOID", "STAFF DISCOUNT", "ITEMS",
    "AUTH CODE", "MERCHANT", "TERMINAL", "PLEASE KEEP YOUR RECEIPT", "THANK YOU FOR SHOPPING",
    "CUSTOMER COPY", "CARDHOLDER COPY", "PIN VERIFIED", "NO CARDHOLDER VERIFICATION",
    "COUNTER", "TILL", "OPERATOR", "SALE",
]
UK_PRODUCT_PREFIXES = ["MEAL DEAL SAVING", "MULTIBUY", "PRICE CUT", "NECTAR", "CLUBCARD",
                       "SAVING", "REDUCED", "PROMO", "DISCOUNT", "VOID", "REFUND"]
UK_STORES = [
    ("SAINSBURY'S", "SAINSBURYS SUPERMARKETS LTD", "HELPING EVERYONE EAT BETTER",
     "WWW.SAINSBURYS.CO.UK", "33 HOLBORN LONDON EC1N 2HT"),
    ("TESCO", "TESCO STORES LTD", "EVERY LITTLE HELPS", "WWW.TESCO.COM",
     "TESCO HOUSE WELWYN GARDEN CITY AL7 1GA"),
    ("ASDA", "ASDA STORES LTD", "SAVE MONEY LIVE BETTER", "WWW.ASDA.COM",
     "ASDA HOUSE LEEDS LS11 5AD"),
    ("CO-OP", "CO-OPERATIVE GROUP LTD", "OWNED BY YOU", "WWW.COOP.CO.UK",
     "1 ANGEL SQUARE MANCHESTER M60 0AG"),
    ("MORRISONS", "WM MORRISON SUPERMARKETS LTD", "MAKERS AND SHOPKEEPERS",
     "WWW.MORRISONS.COM", "HILMORE HOUSE BRADFORD BD8 9AX"),
    ("WAITROSE", "WAITROSE LTD", "WAITROSE & PARTNERS", "WWW.WAITROSE.COM",
     "DONCASTLE ROAD BRACKNELL RG12 8YA"),
    ("M&S", "MARKS AND SPENCER PLC", "M&S FOOD", "WWW.MARKSANDSPENCER.COM",
     "WATERSIDE HOUSE LONDON W2 1NW"),
    ("ALDI", "ALDI STORES LTD", "EVERYDAY AMAZING", "WWW.ALDI.CO.UK",
     "HOLLY LANE ATHERSTONE CV9 2SQ"),
    ("LIDL", "LIDL GREAT BRITAIN LTD", "BIG ON QUALITY LIDL ON PRICE", "WWW.LIDL.CO.UK",
     "19 STROUDLEY ROAD SURREY KT18 5YP"),
]
UK_BRANCHES = ["CAMDEN ABBEY ROAD LOC", "HOLBORN CIRCUS", "KINGS CROSS", "ISLINGTON",
               "CLAPHAM JUNCTION", "MANCHESTER PICCADILLY", "LEEDS CITY", "BRISTOL BROADMEAD",
               "BIRMINGHAM NEW ST", "GLASGOW CENTRAL", "EDINBURGH WAVERLEY", "CARDIFF QUEEN ST",
               "NOTTINGHAM VICTORIA", "OXFORD WESTGATE", "BRIGHTON CHURCHILL SQ", "YORK FOSS"]


def uk_header_line(rng):
    name, company, slogan, url, address = rng.choice(UK_STORES)
    return rng.choice([
        name, company, slogan, url, address, rng.choice(UK_BRANCHES),
        f"0{rng.randint(1, 3)}{rng.randint(0, 99):02d} {rng.randint(100, 999)} "
        f"{rng.randint(1000, 9999)}",
        f"VAT NUMBER {rng.randint(100, 999)} {rng.randint(1000, 9999)} {rng.randint(10, 99)}",
        f"VAT NO GB{rng.randint(100000000, 999999999)}",
        f"STORE {rng.randint(1, 9999)} TILL {rng.randint(1, 40)} OP {rng.randint(1, 999)}",
        f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(20, 26)} "
        f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
        f"{name} {rng.choice(UK_BRANCHES)}",
    ])


def make_non_item(rng, products, loc):
    r = rng.random()
    if loc["name"] == "uk" and r < 0.35:  # receipt header: store, address, VAT ...
        raw = uk_header_line(rng)
    elif r < 0.65:  # totals, payment, fees
        raw = rng.choice(loc["non_items"])
        if loc["name"] == "uk" and rng.random() < 0.3:
            raw = f"{rng.randint(2, 40)} {raw}"  # "13 Sub Total"
        if rng.random() < 0.75:
            amount = rng.uniform(0.05, 150)
            amt = uk_amount(amount, rng) if loc["name"] == "uk" else f"{amount:.2f}"
            raw += " " * rng.choice([1, 1, 3, 8]) + ("-" + amt if rng.random() < 0.2 else amt)
    elif r < 0.85:  # a promotion or coupon that names a product: a hard negative
        product = rng.choice(products)
        printed = rng.choice(rng.choice(product["variants"])["printed"])
        name = " ".join(abbreviate_word(w, rng, 0.8) for w in printed.split())
        amount = rng.uniform(0.25, 5)
        amt = uk_amount(amount, rng) if loc["name"] == "uk" else f"{amount:.2f}"
        raw = f"{rng.choice(loc['product_prefixes'])} {name} -{amt}"
    elif r < 0.93 and loc["name"] == "uk":  # a lone price: "C23.75"
        raw = uk_amount(rng.uniform(0.3, 150), rng)
    else:  # stray reference numbers, card numbers, separators
        raw = rng.choice([
            f"#{rng.randint(1000, 999999)}",
            f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(20, 26)}",
            f"ST# {rng.randint(1, 9999)} OP# {rng.randint(1, 99)} TE# {rng.randint(1, 40)}",
            "*" * rng.randint(5, 30), "-" * rng.randint(5, 30), "=" * rng.randint(5, 30),
            f"**** **** **** {rng.randint(0, 9999):04d}",
        ])
    return vary_case(raw, rng)


LOCALES = {
    "us": {
        "name": "us", "catalogue": HERE / "catalogue.txt",
        "store_brands": ["GV", "KS", "365", "TJ", "SIG", "SB", "GG", "KRO", "MM", "PL",
                         "FRESH MART", "VALU", "HARVEST", "SUNRISE"],
        "fillers_pre": ["", "", "", "", "", "", "F ", "N ", "T ", "* ", "#"],
        # Tax / department flags US POS systems print after the item name.
        "fillers_post": ["", "", "", "", "", "", " N", " F", " T", " X", " TF", " FT",
                         " B", " EA", " NF"],
        "plu_codes": True, "add_price": add_price_us,
        "non_items": US_NON_ITEMS, "product_prefixes": US_PRODUCT_PREFIXES,
        "price_ratio": 0.3, "non_item_ratio": 0.06,
    },
    "uk": {
        "name": "uk", "catalogue": HERE / "catalogue_uk.txt",
        "store_brands": ["JS", "JS", "JS", "SAINSBURYS", "TTD", "TESCO", "TESCO FINEST",
                         "ASDA", "ASDA EXTRA SPECIAL", "M&S", "CO-OP", "WAITROSE",
                         "MORRISONS", "ESSENTIAL", "STOCKWELL", "BY SAINSBURYS"],
        # Sainsbury's marks some lines with *; OCR sometimes reads it as x or |.
        "fillers_pre": [""] * 12 + ["*", "*", "*", "x", "|"],
        "fillers_post": [""],
        "plu_codes": False, "add_price": add_price_uk,
        "non_items": UK_NON_ITEMS, "product_prefixes": UK_PRODUCT_PREFIXES,
        "price_ratio": 0.85, "non_item_ratio": 0.3,
    },
}


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

def build_split(n, rng, products, all_products, allowed_brands, noise_weights,
                seen, args, loc):
    rows = []
    attempts = 0
    levels = list(noise_weights)
    weights = list(noise_weights.values())
    while len(rows) < n and attempts < n * 50:
        attempts += 1
        if rng.random() < args.non_item_ratio:
            raw = make_non_item(rng, all_products, loc)
            row = {"name": NON_ITEM, "generic": NON_ITEM, "category": "non_item",
                   "brand": None, "size": None}
        else:
            product = rng.choice(products)
            max_width = rng.choice([18, 20, 22, 24, 28, 32, 40])
            raw, name, brand, size = make_raw(product, rng, allowed_brands, max_width,
                                              args.price_ratio, loc)
            row = {"name": name, "generic": product["target"],
                   "category": product["category"], "brand": brand, "size": size}

        level = rng.choices(levels, weights=weights)[0]
        if level != "none":
            raw = corrupt_line(raw, rng, LEVELS[level]).strip()
        if not raw or raw in seen:  # no duplicate inputs across splits
            continue
        seen.add(raw)
        rows.append({"input": raw, "target": row[args.target], **row, "ocr_noise": level})
    return rows


SYSTEM_PROMPTS = {
    "name": ("You read one line of OCR'd receipt text. Reply with only the item name, "
             f"without price, size or quantity, or \"{NON_ITEM}\" if the line is not "
             "a purchased item."),
    "generic": ("You normalise receipt line items. Reply with only the generic product "
                f"type, or \"{NON_ITEM}\" if the line is not a product."),
}


def write_exports(out, split, rows, formats, target):
    if "chat" in formats:
        with open(out / f"{split}_chat.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS[target]},
                    {"role": "user", "content": r["input"]},
                    {"role": "assistant", "content": r["target"]},
                ]}, ensure_ascii=False) + "\n")
    if "csv" in formats:
        cols = ["id", "input", "target", "name", "generic", "category", "brand", "size",
                "ocr_noise", "unseen_brand", "unseen_product"]
        with open(out / f"{split}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_items")
    ap.add_argument("--locale", choices=sorted(LOCALES), default="us",
                    help="receipt style: prices, store codes, header lines, catalogue")
    ap.add_argument("--catalogue", help="product file (default: the locale's catalogue)")
    ap.add_argument("--target", choices=["name", "generic"], default="name",
                    help="name = clean item name (Muller Light Strawberry); "
                         "generic = product type (Yogurt)")
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--val", type=int, default=5000)
    ap.add_argument("--test", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", default="none:0.45,light:0.3,medium:0.18,heavy:0.07",
                    help="OCR noise mix on top of abbreviations "
                         "(levels: none, light, medium, heavy)")
    ap.add_argument("--holdout-brands", type=float, default=0.0,
                    help="fraction of brands that appear only in val/test, "
                         "to measure generalisation to unseen brands")
    ap.add_argument("--holdout-products", type=float, default=0.0,
                    help="fraction of product groups that appear only in val/test")
    ap.add_argument("--non-item-ratio", type=float,
                    help="fraction of lines that are headers, totals, fees etc. "
                         "(default: 0.06 us, 0.3 uk)")
    ap.add_argument("--price-ratio", type=float,
                    help="fraction of item lines with a price attached "
                         "(default: 0.3 us, 0.85 uk)")
    ap.add_argument("--formats", default="jsonl,chat,csv",
                    help="comma-separated: jsonl (always written), chat, csv")
    args = ap.parse_args()

    loc = LOCALES[args.locale]
    if args.non_item_ratio is None:
        args.non_item_ratio = loc["non_item_ratio"]
    if args.price_ratio is None:
        args.price_ratio = loc["price_ratio"]
    noise_weights = {}
    for part in args.noise.split(","):
        name, w = part.split(":")
        if name != "none" and name not in LEVELS:
            ap.error(f"unknown noise level {name!r}")
        noise_weights[name] = float(w)
    formats = set(args.formats.split(","))
    if formats - {"jsonl", "chat", "csv"}:
        ap.error(f"unknown format(s) {sorted(formats - {'jsonl', 'chat', 'csv'})}")

    products = parse_catalogue(args.catalogue or loc["catalogue"])
    all_brands = sorted({brand_key(b) for p in products for b in p["brands"]
                         if isinstance(b, dict)})
    groups = sorted({p["target"] for p in products})
    rng = random.Random(args.seed)
    held_brands = set(rng.sample(all_brands, int(len(all_brands) * args.holdout_brands)))
    held_products = set(rng.sample(groups, int(len(groups) * args.holdout_products)))
    train_products = [p for p in products if p["target"] not in held_products]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = set()
    stats = {"seed": args.seed, "locale": args.locale, "target": args.target,
             "product_groups": len(groups),
             "items": sum(len(p["variants"]) for p in products),
             "held_out_brands": sorted(held_brands),
             "held_out_products": sorted(held_products), "splits": {}}
    all_targets = set()
    for split, n in [("train", args.train), ("val", args.val), ("test", args.test)]:
        split_rng = random.Random(f"{args.seed}-{args.locale}-{split}")
        if split == "train":
            pool, allowed = train_products, set(all_brands) - held_brands
        else:
            pool, allowed = products, set(all_brands)
        rows = build_split(n, split_rng, pool, products, allowed, noise_weights, seen,
                           args, loc)
        held_brand_names = {b["clean"].upper() for p in products for b in p["brands"]
                            if isinstance(b, dict) and brand_key(b) in held_brands}
        for i, r in enumerate(rows):
            r["unseen_brand"] = (r["brand"] or "").upper() in held_brand_names
            r["unseen_product"] = r["generic"] in held_products
            rows[i] = {"id": f"{split}-{i:06d}", **r}
            all_targets.add(r["target"])
        count = write_jsonl(out / f"{split}.jsonl", rows)
        write_exports(out, split, rows, formats, args.target)
        print(f"wrote {count:>6} examples -> {out / f'{split}.jsonl'}")
        if count < n:
            print(f"  note: only {count} unique inputs found for {split}")
        stats["splits"][split] = {
            "examples": count,
            "distinct_targets": len({r["target"] for r in rows}),
            "non_item": sum(r["target"] == NON_ITEM for r in rows),
            "unseen_brand": sum(r["unseen_brand"] for r in rows),
            "unseen_product": sum(r["unseen_product"] for r in rows),
            "ocr_noise": dict(Counter(r["ocr_noise"] for r in rows)),
            "categories": dict(sorted(Counter(r["category"] for r in rows).items())),
        }

    (out / "labels.txt").write_text("\n".join(sorted(all_targets)) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {len(all_targets):>6} labels   -> {out / 'labels.txt'}")
    print(f"wrote dataset stats -> {out / 'stats.json'}")


if __name__ == "__main__":
    main()
