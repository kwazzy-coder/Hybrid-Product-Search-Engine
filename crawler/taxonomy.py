"""
Fashion vocabulary used to turn messy store data into clean fields.

Stores label things inconsistently ("Sarees", "SHIRT", "Bottom", "" ...), so the
category is inferred from product_type, title and tags together.
"""

from __future__ import annotations

import re

# (canonical category, subcategory/group, regex alternatives). Order matters only
# as a tie-breaker; scoring in normalize.infer_category does the real work.
CATEGORIES: list[tuple[str, str, list[str]]] = [
    ("kurta set", "ethnicwear", [
        r"kurt[ai]s? sets?", r"suit sets?", r"salwar suits?", r"anarkali sets?", r"sharara sets?",
        r"kurt[ai]s?\b.*\b(?:with|and|&)\b.*\b(?:trousers?|pants?|palazzos?|dupattas?|shararas?|salwars?|churidars?|skirts?)",
    ]),
    ("saree", "ethnicwear", [r"sarees?", r"saris?", r"sarree"]),
    ("lehenga", "ethnicwear", [r"lehengas?", r"lehenga cholis?", r"ghagras?"]),
    ("kurta", "ethnicwear", [r"kurtas?", r"kurtis?", r"anarkalis?", r"tunics?"]),
    ("dupatta", "ethnicwear", [r"dupattas?", r"stoles?", r"shawls?"]),
    ("co-ord set", "sets", [r"co-?ords?(?: sets?)?", r"coord sets?", r"two[- ]piece sets?", r"matching sets?"]),
    ("jumpsuit", "dresses", [r"jumpsuits?", r"playsuits?", r"rompers?", r"dungarees?"]),
    ("dress", "dresses", [r"dress(?:es)?", r"gowns?", r"frocks?"]),
    ("hoodie", "topwear", [r"hoodies?", r"hooded sweatshirts?"]),
    ("sweatshirt", "topwear", [r"sweatshirts?"]),
    ("tshirt", "topwear", [r"t-?shirts?", r"tees?", r"polos?(?: t-?shirts?)?", r"henleys?", r"tank tops?", r"vests? tops?"]),
    ("shirt", "topwear", [r"shirts?", r"overshirts?"]),
    ("top", "topwear", [r"tops?", r"crop tops?", r"camis?", r"blouses?", r"bodysuits?", r"corsets?"]),
    ("sweater", "topwear", [r"sweaters?", r"cardigans?", r"pullovers?", r"jumpers?", r"knitwear"]),
    ("jacket", "outerwear", [r"jackets?", r"bombers?", r"shackets?", r"coats?", r"blazers?", r"waistcoats?", r"gilets?", r"puffers?"]),
    ("jeans", "bottomwear", [r"jeans", r"denims?"]),
    ("trousers", "bottomwear", [
        r"trousers?", r"pants?", r"chinos?", r"cargos?", r"joggers?", r"track ?pants?", r"palazzos?",
        r"leggings?", r"churidars?", r"salwars?", r"pyjamas?", r"shararas?", r"culottes?", r"trackpants?",
    ]),
    ("shorts", "bottomwear", [r"shorts", r"bermudas?"]),
    ("skirt", "bottomwear", [r"skirts?"]),
    ("shoes", "footwear", [
        r"shoes?", r"sneakers?", r"loafers?", r"boots?", r"sandals?", r"sliders?", r"slides", r"flip[- ]?flops?",
        r"heels?", r"flats", r"mules?", r"juttis?", r"mojaris?", r"kolhapuris?", r"footwear", r"derbys?",
        r"brogues?", r"oxfords?", r"clogs?", r"espadrilles?",
    ]),
    ("bag", "accessories", [r"bags?", r"handbags?", r"totes?", r"backpacks?", r"clutch(?:es)?", r"sling bags?", r"potlis?", r"duffles?"]),
    ("wallet", "accessories", [r"wallets?", r"card holders?"]),
    ("cap", "accessories", [r"caps?", r"hats?", r"beanies?", r"bucket hats?"]),
    ("belt", "accessories", [r"belts?"]),
    ("socks", "accessories", [r"socks?"]),
    ("jewellery", "accessories", [
        r"jewell?e?ry", r"earrings?", r"necklaces?", r"bracelets?", r"bangles?", r"jhumkas?", r"anklets?",
        r"pendants?", r"chokers?", r"maang tikkas?", r"nose pins?",
    ]),
    ("watch", "accessories", [r"watch(?:es)?"]),
    ("sunglasses", "accessories", [r"sunglasses", r"eyewear"]),
    ("scarf", "accessories", [r"scarf", r"scarves", r"mufflers?", r"bandanas?"]),
    ("innerwear", "innerwear", [r"boxers?", r"briefs?", r"trunks?", r"bras?", r"panties", r"innerwear", r"camisoles?", r"lingerie"]),
    ("nightwear", "nightwear", [r"night ?suits?", r"nightwear", r"nighties?", r"loungewear", r"lounge sets?", r"sleepwear", r"night dress(?:es)?"]),
    ("fragrance", "beauty", [r"perfumes?", r"fragrances?", r"deodorants?", r"body mists?", r"eau de (?:parfum|toilette)"]),
]

# Categories that win whenever they appear in the title, because they describe
# the whole product even when another garment word comes later
# ("Saree with Blouse Piece", "Kurta with Trousers & Dupatta").
DOMINANT = {"kurta set", "saree", "lehenga", "co-ord set", "jumpsuit"}

NON_FASHION = re.compile(
    r"\b(?:e-?gift|gift ?cards?|gift ?vouchers?|dinner ?sets?|dinnerware|mugs?|cushions?|candles?|"
    r"bedsheets?|bed ?linen|plates?|bowls?|coasters?|planters?|vases?|diffusers?|phone ?covers?|mobile ?covers?|"
    r"shipping ?protection|insurance|donation)\b",
    re.IGNORECASE,
)

COMPILED_CATEGORIES = [
    (name, group, re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE))
    for name, group, patterns in CATEGORIES
]
CATEGORY_GROUP = {name: group for name, group, _ in CATEGORIES}

# Specific colour name -> families it should also match. The query parser
# understands the family names (red, blue, navy, ...), so we emit both.
COLORS: dict[str, tuple[str, ...]] = {
    "jet black": ("black",), "black": ("black",),
    "charcoal": ("grey",), "grey": ("grey",), "gray": ("grey",), "silver": ("grey",), "ash": ("grey",),
    "off white": ("white",), "off-white": ("white",), "white": ("white",), "ivory": ("white",),
    "cream": ("beige",), "beige": ("beige",), "khaki": ("beige",), "nude": ("beige",), "sand": ("beige",),
    "tan": ("brown",), "camel": ("brown",), "brown": ("brown",), "coffee": ("brown",), "chocolate": ("brown",),
    "rust": ("orange",), "orange": ("orange",), "peach": ("orange",), "coral": ("orange",),
    "red": ("red",), "maroon": ("maroon", "red"), "wine": ("maroon", "red"), "burgundy": ("maroon", "red"),
    "pink": ("pink",), "rose": ("pink",), "fuchsia": ("pink",), "magenta": ("pink",), "blush": ("pink",),
    "purple": ("purple",), "lavender": ("purple",), "lilac": ("purple",), "mauve": ("purple",), "violet": ("purple",),
    "navy blue": ("navy", "blue"), "navy": ("navy", "blue"), "sky blue": ("blue",), "royal blue": ("blue",),
    "powder blue": ("blue",), "blue": ("blue",), "turquoise": ("blue",), "aqua": ("blue",), "indigo": ("blue",),
    "teal": ("green", "blue"), "olive": ("green",), "mint": ("green",), "sea green": ("green",),
    "bottle green": ("green",), "emerald": ("green",), "sage": ("green",), "green": ("green",),
    "mustard": ("yellow",), "lemon": ("yellow",), "yellow": ("yellow",), "gold": ("gold", "yellow"), "golden": ("gold", "yellow"),
    "multicolor": ("multi",), "multicolour": ("multi",), "multi": ("multi",),
}
# Longest names first so "navy blue" wins over "blue".
COLOR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(COLORS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

MATERIALS = [
    "cotton blend", "silk blend", "linen blend", "cotton", "linen", "silk", "satin", "rayon", "viscose",
    "polyester", "georgette", "chiffon", "crepe", "denim", "wool", "woollen", "leather", "suede", "velvet",
    "organza", "chanderi", "khadi", "modal", "lyocell", "tencel", "nylon", "jute", "canvas", "muslin",
    "mulmul", "corduroy", "fleece", "jersey", "terry", "chikankari", "banarasi", "kanjeevaram", "ikat",
    "bandhani", "block print", "handloom",
]
MATERIAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in sorted(MATERIALS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

GENDER_PATTERNS = [
    ("women", re.compile(r"\b(?:women'?s?|woman|female|ladies|girls?|her)\b", re.IGNORECASE)),
    ("men", re.compile(r"\b(?:men'?s?|man|male|boys?|gents|him)\b", re.IGNORECASE)),
    ("unisex", re.compile(r"\bunisex\b", re.IGNORECASE)),
]
