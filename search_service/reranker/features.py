from query_understanding import QueryIntent

def price_match_score(price: float, max_price: float | None) -> float:
    if max_price is None or max_price == 0:
        return 1.0
    if price > max_price:
        return 0.0
    return 1.0 - (price / max_price) * 0.3   # slight boost for cheaper items

def build_feature_vector(query_intent: QueryIntent, product: dict, rrf_score: float, clip_score: float = 0.0) -> list[float]:
    # Extract categories and normalize safely
    q_cat = (query_intent.category or "").strip().lower()
    
    category_match = 0.0
    if q_cat:
        p_cat = str(product.get("category", "") or "").strip().lower()
        p_subcat = str(product.get("subcategory", "") or "").strip().lower()
        p_name = str(product.get("name", "") or "").strip().lower()
        p_tags = [str(t).lower() for t in product.get("tags", [])]
        
        # Check direct or synonym match
        synonyms = {
            "kurta": ["kurta", "kurti", "ethnic"],
            "shirt": ["shirt", "shirts"],
            "tshirt": ["tshirt", "t-shirt", "tee"],
            "jeans": ["jeans", "denim"],
            "dress": ["dress", "frock", "gown"],
            "shoes": ["shoes", "sneakers", "footwear"]
        }
        
        candidates = [q_cat]
        if q_cat in synonyms:
            candidates.extend(synonyms[q_cat])
            
        if p_cat == q_cat or any(c in p_subcat or c in p_name or c in p_tags for c in candidates):
            category_match = 1.0
    
    # Color match safety check
    p_color_val = product.get("color", [])
    if p_color_val is None:
        p_color_val = []
    elif not isinstance(p_color_val, list):
        p_color_val = [p_color_val]
    p_colors = [str(c).strip().lower() for c in p_color_val if c is not None]
    
    q_color_val = query_intent.color
    if q_color_val is None:
        q_color_val = []
    q_colors = [str(c).strip().lower() for c in q_color_val if c is not None]
    color_match = 1.0 if any(c in p_colors for c in q_colors) else 0.0
    
    # Price proximity safety check
    price_val = product.get("price")
    p_price = float(price_val) if price_val is not None else 0.0
    p_max_price = query_intent.max_price
    p_match = price_match_score(p_price, p_max_price)
    
    # Normalized ratings and reviews safety check
    rating_val = product.get("rating")
    rating = (float(rating_val) if rating_val is not None else 0.0) / 5.0
    
    reviews_val = product.get("review_count")
    reviews = min((float(reviews_val) if reviews_val is not None else 0.0) / 1000.0, 1.0)
    
    # Discount signal safety check
    discount_val = product.get("discount_percent")
    discount = (float(discount_val) if discount_val is not None else 0.0) / 100.0
    
    # In stock status
    in_stock = 1.0 if product.get("in_stock", True) else 0.0
    
    # f10: attribute match
    query_attrs = []
    if q_cat:
        query_attrs.append(q_cat)
    if q_colors:
        query_attrs.extend(q_colors)
    
    if len(query_attrs) == 0:
        attribute_match = 0.5
    else:
        matched = 0
        if category_match > 0:
            matched += 1
        for qc in q_colors:
            if any(qc in pc for pc in p_colors):
                matched += 1
        attribute_match = matched / len(query_attrs)

    # f11: binding score
    color_bindings = getattr(query_intent, 'color_bindings', {})
    if not color_bindings:
        binding_score = 1.0
    else:
        p_name_tags = str(product.get("name", "")).lower() + " " + " ".join([str(t).lower() for t in product.get("tags", [])])
        matched_bindings = 0
        for item, color in color_bindings.items():
            if item in p_name_tags and any(color in pc for pc in p_colors):
                matched_bindings += 1
        binding_score = matched_bindings / len(color_bindings)
    
    return [
        rrf_score,          # f1: RRF score
        category_match,     # f2: category match
        color_match,        # f3: color match
        p_match,            # f4: price match score
        rating,             # f5: rating
        reviews,            # f6: review count
        discount,           # f7: discount percentage
        in_stock,           # f8: in stock
        clip_score,         # f9: clip score
        attribute_match,    # f10: attribute match
        binding_score       # f11: binding score
    ]
