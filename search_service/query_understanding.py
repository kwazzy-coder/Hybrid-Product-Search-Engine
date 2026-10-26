import os
import json
import re
import redis
from pydantic import BaseModel, Field
# LangChain/Anthropic are imported only when an API key is set (see parse_query),
# so the lightweight deployment doesn't need them in memory.

from sizes import extract_size

class QueryIntent(BaseModel):
    original_query: str
    expanded_terms: list[str] = Field(default_factory=list)
    category: str | None = None
    color: list[str] = Field(default_factory=list)
    color_bindings: dict[str, str] = Field(default_factory=dict)
    max_price: float | None = None
    min_rating: float | None = None
    brand: str | None = None
    size: str | None = None


# Canonical categories (the crawler emits these) and the words people use for them.
# Order matters: more specific entries first ("kurta set" before "kurta", "tshirt" before "shirt").
CATEGORY_SYNONYMS: dict[str, list[str]] = {
    "kurta set": ["kurta set", "kurta sets", "suit set", "salwar suit", "anarkali set", "sharara set"],
    "saree": ["saree", "sarees", "sari", "saris"],
    "lehenga": ["lehenga", "lehengas", "lehenga choli", "ghagra"],
    "kurta": ["kurta", "kurtas", "kurti", "kurtis", "anarkali", "ethnic top", "tunic"],
    "dupatta": ["dupatta", "dupattas", "stole", "shawl"],
    "co-ord set": ["co-ord", "co-ords", "coord", "co-ord set", "coord set", "matching set"],
    "jumpsuit": ["jumpsuit", "jumpsuits", "playsuit", "romper", "dungarees"],
    "dress": ["dress", "dresses", "frock", "gown", "maxi", "one-piece"],
    "hoodie": ["hoodie", "hoodies", "hooded sweatshirt"],
    "sweatshirt": ["sweatshirt", "sweatshirts"],
    "tshirt": ["tshirt", "tshirts", "t-shirt", "t-shirts", "t shirt", "tee", "tees", "polo", "polos", "henley"],
    "shirt": ["shirt", "shirts", "overshirt", "formal shirt", "casual shirt"],
    "top": ["top", "tops", "crop top", "cami", "blouse", "bodysuit"],
    "sweater": ["sweater", "sweaters", "cardigan", "pullover", "jumper"],
    "jacket": ["jacket", "jackets", "bomber", "shacket", "coat", "blazer", "waistcoat", "nehru jacket"],
    "jeans": ["jeans", "denim", "denims"],
    "trousers": ["trousers", "trouser", "pants", "chinos", "cargo", "cargos", "joggers", "track pants", "palazzo", "leggings"],
    "shorts": ["shorts", "bermuda"],
    "skirt": ["skirt", "skirts"],
    "shoes": ["shoes", "shoe", "sneakers", "sneaker", "footwear", "loafers", "boots", "sandals", "slides",
              "flip flops", "heels", "flats", "juttis", "mojaris", "kolhapuris", "running shoes"],
    "bag": ["bag", "bags", "handbag", "handbags", "tote", "backpack", "clutch", "sling bag", "potli"],
    "wallet": ["wallet", "wallets", "card holder"],
    "cap": ["cap", "caps", "hat", "hats", "beanie"],
    "belt": ["belt", "belts"],
    "socks": ["socks", "sock"],
    "jewellery": ["jewellery", "jewelry", "earrings", "necklace", "bracelet", "bangles", "jhumka", "jhumkas"],
    "watch": ["watch", "watches"],
    "sunglasses": ["sunglasses", "shades", "eyewear"],
    "scarf": ["scarf", "scarves", "muffler"],
    "innerwear": ["innerwear", "boxers", "briefs", "trunks", "bra", "bras", "lingerie"],
    "nightwear": ["nightwear", "night suit", "pyjama set", "loungewear", "sleepwear"],
    "fragrance": ["perfume", "perfumes", "fragrance", "deodorant", "body mist"],
}
KNOWN_CATEGORIES = set(CATEGORY_SYNONYMS)


def category_matches(product: dict, category: str | None) -> bool:
    """Does a product belong to `category`? Word-boundary matching, so "tee" never matches "steel"."""
    if not category:
        return True
    category = category.strip().lower()
    names = {category, *CATEGORY_SYNONYMS.get(category, [])}
    p_cat = str(product.get("category", "") or "").strip().lower()
    if p_cat in names:
        return True
    if p_cat in KNOWN_CATEGORIES:
        # The product has a clean category and it's a different one (e.g. a tshirt isn't a shirt).
        return False
    # Older catalogs use free-form categories ("watches", "handbags"): fall back to text.
    text = " ".join([
        str(product.get("subcategory", "") or ""), p_cat, str(product.get("name", "") or ""),
        " ".join(str(t) for t in product.get("tags", []) or []),
    ]).lower()
    return any(re.search(rf"\b{re.escape(n)}\b", text) for n in names)

# Initialize Redis client
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = None
try:
    # Set connect and socket timeouts to 1 second
    client = redis.Redis.from_url(
        REDIS_URL, 
        socket_connect_timeout=1.0, 
        socket_timeout=1.0, 
        decode_responses=True
    )
    # Ping to check if server is actually online
    client.ping()
    redis_client = client
    print("Connected to Redis successfully. Caching is enabled.")
except Exception as e:
    print(f"Warning: Redis is offline ({e}). Caching will be disabled.")


SYSTEM_PROMPT = """
You are a search query parser for an Indian e-commerce platform.
Given a user query, extract structured intent and expand the query
with synonyms and related terms relevant to Indian fashion/products.
Return valid JSON only. No explanation.
"category" must be one of: {categories} — or null if none fits.
"size" is the clothing or shoe size if the user gives one (e.g. "M", "XL", "32", "8"), else null.

Example:
Input: "red kurta under 500"
Output: {
  "original_query": "red kurta under 500",
  "expanded_terms": ["kurta", "kurti", "ethnic top", "Indian wear", "red", "maroon"],
  "category": "kurta",
  "color": ["red"],
  "max_price": 500,
  "min_rating": null,
  "brand": null,
  "size": null
}
"""

def parse_query_local_fallback(raw_query: str) -> QueryIntent:
    """Robust local parsing fallback when Anthropic API is not available"""
    print(f"Using local rule-based fallback parser for query: '{raw_query}'")
    size, query_lower = extract_size(raw_query)
    
    # 1. Extract colors
    colors_list = ["red", "blue", "green", "black", "white", "yellow", "pink", "maroon", "navy", "grey", "beige", "purple", "orange"]
    extracted_colors = [c for c in colors_list if re.search(rf"\b{c}\b", query_lower)]
    
    # 2. Extract max price ("under 500", "below 1000", "under rs 600", etc.)
    max_price = None
    price_patterns = [
        r"under\s+(?:rs\.?\s*)?(\d+)",
        r"below\s+(?:rs\.?\s*)?(\d+)",
        r"less\s+than\s+(?:rs\.?\s*)?(\d+)",
        r"under\s*(\d+)",
        r"<\s*(\d+)"
    ]
    for pattern in price_patterns:
        match = re.search(pattern, query_lower)
        if match:
            max_price = float(match.group(1))
            break
            
    # 3. Extract rating ("above 4 star", "4+ star", "4 star", etc.)
    min_rating = None
    rating_match = re.search(r"(\d+(\.\d+)?)\s*\+?\s*(?:star|rating)", query_lower)
    if rating_match:
        min_rating = float(rating_match.group(1))
    else:
        # Check simple pattern like "above 4"
        above_match = re.search(r"above\s+(\d+(\.\d+)?)", query_lower)
        if above_match:
            min_rating = float(above_match.group(1))

    # 4. Extract brand
    brands_list = ["libas", "biba", "anouk", "sangria", "roadster", "hrx", "zara", "h&m", "nike", "adidas"]
    extracted_brand = None
    for b in brands_list:
        if re.search(rf"\b{b}\b", query_lower):
            extracted_brand = b.capitalize()
            break

    # 5. Extract category & expand terms (known categories only)
    category = None
    expanded_terms = []

    for cat, synonyms in CATEGORY_SYNONYMS.items():
        matched = [syn for syn in [cat, *synonyms] if re.search(rf"\b{re.escape(syn)}\b", query_lower)]
        if matched:
            category = cat
            # The canonical name plus the words the user actually typed ("kurti" -> kurta, kurti).
            expanded_terms = list(dict.fromkeys([cat, *matched]))
            break

    # Keep the user's own words too, so "silk", "wedding" or "oversized" still reach BM25.
    stop = {"under", "below", "above", "less", "than", "star", "stars", "rating", "rs", "for", "with", "and", "the", "a", "in", "of"}
    words = [w for w in re.findall(r"[a-z0-9-]+", query_lower) if w not in stop and not w.isdigit()]
    expanded_terms.extend(words)
    if not expanded_terms:
        expanded_terms = list(extracted_colors) if extracted_colors else [query_lower]

    # Add color to expanded terms
    for col in extracted_colors:
        expanded_terms.append(col)
        # Add some color synonyms
        if col == "red":
            expanded_terms.append("maroon")
        elif col == "blue":
            expanded_terms.append("navy")
            
    # Detect color bindings: "red kurta and white shirt" → {"kurta": "red", "shirt": "white"}
    color_bindings = {}
    binding_pattern = r'(\w+)\s+(\w+)\s+(?:and|with|&)\s+(\w+)\s+(\w+)'
    match = re.search(binding_pattern, query_lower)
    if match:
        c1, item1, c2, item2 = match.groups()
        if c1 in colors_list and c2 in colors_list:
            color_bindings = {item1: c1, item2: c2}
        elif c1 in colors_list:
            color_bindings = {item1: c1}
        elif c2 in colors_list:
            color_bindings = {item2: c2}
            
    return QueryIntent(
        original_query=raw_query,
        expanded_terms=list(set(expanded_terms)),
        category=category,
        color=extracted_colors,
        color_bindings=color_bindings,
        max_price=max_price,
        min_rating=min_rating,
        brand=extracted_brand,
        size=size,
    )

def parse_query(raw_query: str) -> QueryIntent:
    # 1. Check Redis Cache
    cache_key = f"query_intent:{raw_query.strip().lower()}"
    if redis_client:
        try:
            cached_val = redis_client.get(cache_key)
            if cached_val:
                print(f"Redis Cache Hit for: '{raw_query}'")
                return QueryIntent(**json.loads(cached_val))
        except Exception as e:
            print(f"Redis error: {e}")

    # 2. Check API key presence
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        # Fallback if no API key is configured
        intent = parse_query_local_fallback(raw_query)
    else:
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.output_parsers import JsonOutputParser

            # Setup LangChain ChatAnthropic with JSON parser
            llm = ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
                temperature=0.0,
                api_key=api_key,
                timeout=5.0
            )
            parser = JsonOutputParser(pydantic_object=QueryIntent)
            
            system = SYSTEM_PROMPT.replace("{categories}", ", ".join(CATEGORY_SYNONYMS))
            prompt = f"System: {system}\nUser query: \"{raw_query}\"\nJSON output:"
            response = llm.invoke(prompt)
            
            # Extract content from response
            res_content = response.content if hasattr(response, "content") else str(response)
            # Find JSON block if any markdown wrapper is returned
            json_match = re.search(r"(\{.*\})", res_content, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(1))
            else:
                parsed_json = json.loads(res_content)
                
            intent = QueryIntent(**parsed_json)
            if not intent.size:
                intent.size = extract_size(raw_query)[0]
        except Exception as e:
            print(f"Error calling Anthropic API: {e}. Falling back to rule-based parser.")
            intent = parse_query_local_fallback(raw_query)

    # 3. Save to Redis Cache (1 hour TTL)
    if redis_client:
        try:
            redis_client.setex(cache_key, 3600, intent.model_dump_json())
        except Exception as e:
            print(f"Redis write error: {e}")

    return intent
