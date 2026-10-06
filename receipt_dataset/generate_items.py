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

Products live in catalogue.txt. Lines that are not products (totals, fees,
coupons) are labelled "Non-Item". Output is written as JSONL, plus chat-style
JSONL for LLM fine-tuning and CSV.

Usage:
    python generate_items.py                          # 50k / 5k / 5k
    python generate_items.py --holdout-brands 0.15    # test on unseen brands
    python generate_items.py --holdout-products 0.1   # test on unseen products
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from generate import LEVELS, corrupt_line, write_jsonl

CATALOGUE_PATH = Path(__file__).with_name("catalogue.txt")

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

# Store / private-label brand codes that print before the product name.
STORE_BRANDS = ["GV", "KS", "365", "TJ", "SIG", "SB", "GG", "KRO", "MM", "PL",
                "FRESH MART", "VALU", "HARVEST", "SUNRISE"]

FILLERS_PRE = ["", "", "", "", "", "", "F ", "N ", "T ", "* ", "#"]
# Tax / department flags POS systems print after the item name.
FILLERS_POST = ["", "", "", "", "", "", " N", " F", " T", " X", " TF", " FT", " B",
                " EA", " NF"]


def parse_catalogue(path=CATALOGUE_PATH):
    """Load products from catalogue.txt, rejecting malformed or ambiguous rows."""
    products, owner = [], {}
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 '|' fields, got {len(fields)}")
        target, category, brands, sizes, variants = (f.strip() for f in fields)
        product = {
            "target": target,
            "category": category,
            "brands": [b.strip() for b in brands.split(",")],
            "sizes": [s.strip() for s in sizes.split(",")],
            "variants": [v.strip() for v in variants.split(";") if v.strip()],
        }
        for v in product["variants"]:
            if owner.setdefault(v, target) != target:
                raise ValueError(f"{path}:{lineno}: variant {v!r} is used by "
                                 f"both {owner[v]!r} and {target!r}")
        products.append(product)
    targets = [p["target"] for p in products]
    dupes = {t for t in targets if targets.count(t) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate targets {sorted(dupes)}")
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


def add_price(raw, product, rng):
    """Append the price columns that often get OCR'd onto the item line."""
    price = rng.uniform(0.5, 25.0) if product["category"] not in ("alcohol", "baby", "automotive") else rng.uniform(5, 60)
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


def vary_case(raw, rng):
    """Most receipts print in upper case, but some POS systems don't."""
    r = rng.random()
    if r < 0.10:
        return raw.title()
    if r < 0.14:
        return raw.lower()
    return raw


def make_raw(product, rng, allowed_brands, max_width, price_ratio):
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

    if rng.random() < price_ratio:
        raw = add_price(raw, product, rng)
    return vary_case(raw.strip(), rng), brand, size


# --------------------------------------------------------------------------
# Non-item lines
#
# Real receipt text also contains totals, fees, coupons and payment lines.
# Teaching the model to label these NON_ITEM stops it inventing a product
# for "COUPON LINDT CHOC" or "BOTTLE DEPOSIT".
# --------------------------------------------------------------------------

NON_ITEM = "Non-Item"

NON_ITEM_LINES = [
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
NON_ITEM_PRODUCT_PREFIXES = ["COUPON", "CPN", "MFR CPN", "SC", "DISC", "SAVINGS",
                             "REWARDS", "BOGO", "VOID", "RETURN", "DEP"]


def make_non_item(rng, products):
    r = rng.random()
    if r < 0.45:
        raw = rng.choice(NON_ITEM_LINES)
        if rng.random() < 0.7:
            raw += " " * rng.choice([1, 3, 8]) + f"{rng.uniform(0.05, 150):.2f}"
            if rng.random() < 0.3:
                raw += "-"
    elif r < 0.85:  # coupon or discount that names a product: a hard negative
        product = rng.choice(products)
        name = " ".join(abbreviate_word(w, rng, 0.8)
                        for w in rng.choice(product["variants"]).split())
        raw = f"{rng.choice(NON_ITEM_PRODUCT_PREFIXES)} {name} -{rng.uniform(0.25, 5):.2f}"
    else:  # stray reference numbers, dates, separators
        raw = rng.choice([
            f"#{rng.randint(1000, 999999)}",
            f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(20, 26)}",
            f"ST# {rng.randint(1, 9999)} OP# {rng.randint(1, 99)} TE# {rng.randint(1, 40)}",
            "*" * rng.randint(5, 30), "-" * rng.randint(5, 30), "=" * rng.randint(5, 30),
            f"**** **** **** {rng.randint(0, 9999):04d}",
        ])
    return vary_case(raw, rng)


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

def build_split(n, rng, products, all_products, allowed_brands, noise_weights,
                seen, non_item_ratio, price_ratio):
    rows = []
    attempts = 0
    levels = list(noise_weights)
    weights = list(noise_weights.values())
    while len(rows) < n and attempts < n * 50:
        attempts += 1
        if rng.random() < non_item_ratio:
            raw = make_non_item(rng, all_products)
            row = {"target": NON_ITEM, "category": "non_item", "brand": None, "size": None}
        else:
            product = rng.choice(products)
            max_width = rng.choice([18, 20, 22, 24, 28, 32, 40])
            raw, brand, size = make_raw(product, rng, allowed_brands, max_width, price_ratio)
            row = {"target": product["target"], "category": product["category"],
                   "brand": brand, "size": size}

        level = rng.choices(levels, weights=weights)[0]
        if level != "none":
            raw = corrupt_line(raw, rng, LEVELS[level]).strip()
        if not raw or raw in seen:  # no duplicate inputs across splits
            continue
        seen.add(raw)
        rows.append({"input": raw, **row, "ocr_noise": level})
    return rows


SYSTEM_PROMPT = ("You normalise receipt line items. Reply with only the generic "
                 f"product name, or \"{NON_ITEM}\" if the line is not a product.")


def write_exports(out, split, rows, formats):
    if "chat" in formats:
        with open(out / f"{split}_chat.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["input"]},
                    {"role": "assistant", "content": r["target"]},
                ]}, ensure_ascii=False) + "\n")
    if "csv" in formats:
        cols = ["id", "input", "target", "category", "brand", "size", "ocr_noise",
                "unseen_brand", "unseen_product"]
        with open(out / f"{split}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_items")
    ap.add_argument("--catalogue", default=str(CATALOGUE_PATH))
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--val", type=int, default=5000)
    ap.add_argument("--test", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", default="none:0.55,light:0.3,medium:0.12,heavy:0.03",
                    help="OCR noise mix on top of abbreviations "
                         "(levels: none, light, medium, heavy)")
    ap.add_argument("--holdout-brands", type=float, default=0.0,
                    help="fraction of brands that appear only in val/test, "
                         "to measure generalisation to unseen brands")
    ap.add_argument("--holdout-products", type=float, default=0.0,
                    help="fraction of products that appear only in val/test "
                         "(for generative models; a classifier cannot get these)")
    ap.add_argument("--non-item-ratio", type=float, default=0.06,
                    help="fraction of lines that are totals, fees, coupons etc.")
    ap.add_argument("--price-ratio", type=float, default=0.3,
                    help="fraction of item lines with a price or quantity attached")
    ap.add_argument("--formats", default="jsonl,chat,csv",
                    help="comma-separated: jsonl (always written), chat, csv")
    ap.add_argument("--target-case", choices=["title", "upper", "lower"],
                    default="title")
    args = ap.parse_args()

    noise_weights = {}
    for part in args.noise.split(","):
        name, w = part.split(":")
        if name != "none" and name not in LEVELS:
            ap.error(f"unknown noise level {name!r}")
        noise_weights[name] = float(w)
    formats = set(args.formats.split(","))
    if formats - {"jsonl", "chat", "csv"}:
        ap.error(f"unknown format(s) {sorted(formats - {'jsonl', 'chat', 'csv'})}")

    products = parse_catalogue(args.catalogue)
    all_brands = sorted({b for p in products for b in p["brands"] if b != "-"})
    rng = random.Random(args.seed)
    held_brands = set(rng.sample(all_brands, int(len(all_brands) * args.holdout_brands)))
    held_products = set(rng.sample([p["target"] for p in products],
                                   int(len(products) * args.holdout_products)))
    train_products = [p for p in products if p["target"] not in held_products]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = set()
    stats = {"seed": args.seed, "catalogue_products": len(products),
             "held_out_brands": sorted(held_brands),
             "held_out_products": sorted(held_products), "splits": {}}
    for split, n in [("train", args.train), ("val", args.val), ("test", args.test)]:
        split_rng = random.Random(f"{args.seed}-{split}")
        if split == "train":
            pool, allowed = train_products, set(all_brands) - held_brands
        else:
            pool, allowed = products, set(all_brands)
        rows = build_split(n, split_rng, pool, products, allowed, noise_weights, seen,
                           args.non_item_ratio, args.price_ratio)
        for i, r in enumerate(rows):
            if args.target_case == "upper":
                r["target"] = r["target"].upper()
            elif args.target_case == "lower":
                r["target"] = r["target"].lower()
            r["unseen_brand"] = r["brand"] in held_brands
            r["unseen_product"] = r["target"] in held_products
            rows[i] = {"id": f"{split}-{i:06d}", **r}
        count = write_jsonl(out / f"{split}.jsonl", rows)
        write_exports(out, split, rows, formats)
        print(f"wrote {count:>6} examples -> {out / f'{split}.jsonl'}")
        if count < n:
            print(f"  note: only {count} unique inputs found for {split}")
        stats["splits"][split] = {
            "examples": count,
            "labels": len({r["target"] for r in rows}),
            "non_item": sum(r["target"] == NON_ITEM for r in rows),
            "unseen_brand": sum(r["unseen_brand"] for r in rows),
            "unseen_product": sum(r["unseen_product"] for r in rows),
            "ocr_noise": dict(Counter(r["ocr_noise"] for r in rows)),
            "categories": dict(sorted(Counter(r["category"] for r in rows).items())),
        }

    labels = sorted({p["target"] for p in products} | {NON_ITEM})
    (out / "labels.txt").write_text("\n".join(labels) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {len(labels):>6} labels   -> {out / 'labels.txt'}")
    print(f"wrote dataset stats -> {out / 'stats.json'}")


if __name__ == "__main__":
    main()
