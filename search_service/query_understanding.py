import os
import json
import re
import redis
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser

class QueryIntent(BaseModel):
    original_query: str
    expanded_terms: list[str] = Field(default_factory=list)
    category: str | None = None
    color: list[str] = Field(default_factory=list)
    color_bindings: dict[str, str] = Field(default_factory=dict)
    max_price: float | None = None
    min_rating: float | None = None
    brand: str | None = None

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

Example:
Input: "red kurta under 500"
Output: {
  "original_query": "red kurta under 500",
  "expanded_terms": ["kurta", "kurti", "ethnic top", "Indian wear", "red", "maroon"],
  "category": "kurta",
  "color": ["red"],
  "max_price": 500,
  "min_rating": null,
  "brand": null
}
"""

def parse_query_local_fallback(raw_query: str) -> QueryIntent:
    """Robust local parsing fallback when Anthropic API is not available"""
    print(f"Using local rule-based fallback parser for query: '{raw_query}'")
    query_lower = raw_query.lower()
    
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

    # 5. Extract category & expand terms
    categories_map = {
        "kurta": ["kurta", "kurti", "ethnic top", "Indian wear", "ethnic"],
        "shirt": ["shirt", "shirts", "formal shirt", "casual shirt"],
        "tshirt": ["tshirt", "t-shirt", "tee", "polos", "top"],
        "jeans": ["jeans", "denim", "pants", "trousers"],
        "dress": ["dress", "frock", "maxi", "one-piece"],
        "shoes": ["shoes", "sneakers", "footwear", "running shoes"]
    }
    
    category = None
    expanded_terms = []
    
    for cat, synonyms in categories_map.items():
        if re.search(rf"\b{cat}\b", query_lower) or any(re.search(rf"\b{syn}\b", query_lower) for syn in synonyms):
            category = cat
            expanded_terms = list(synonyms)
            break
            
    if not category:
        # Default category extraction based on nouns
        # Use first word as category if nothing found
        words = [w for w in query_lower.split() if w not in colors_list and w not in ["under", "below", "above", "star", "rating", "rs"]]
        if words:
            category = words[0]
            expanded_terms = [category]
        else:
            # Pure color/attribute query (e.g., "green", "red") — don't force a category
            category = None
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
        brand=extracted_brand
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
            # Setup LangChain ChatAnthropic with JSON parser
            llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.0,
                api_key=api_key,
                timeout=5.0
            )
            parser = JsonOutputParser(pydantic_object=QueryIntent)
            
            prompt = f"System: {SYSTEM_PROMPT}\nUser query: \"{raw_query}\"\nJSON output:"
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
