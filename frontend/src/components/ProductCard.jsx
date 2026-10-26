import React from 'react';
import './ProductCard.css';

const BACKEND_URL = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000';

const SOURCE_COLORS = {
  amazon: '#FF9900',
  myntra: '#FF3F6C',
  ajio: '#3880FF',
  flipkart: '#2874F0',
};

// Stable colour per store for crawled sources we don't have a brand colour for.
function storeColor(source) {
  if (SOURCE_COLORS[source]) return SOURCE_COLORS[source];
  let hash = 0;
  for (const ch of source || '') hash = (hash * 31 + ch.charCodeAt(0)) % 360;
  return `hsl(${hash}, 55%, 42%)`;
}

const formatPrice = (value) => Number(value).toLocaleString('en-IN', { maximumFractionDigits: 0 });

export function ProductCard({ product }) {
  const {
    name,
    brand,
    price,
    mrp,
    discount_percent,
    rating,
    review_count,
    in_stock,
    image_url,
    source,
    source_name,
    source_url,
    scraped_at,
    available_sizes,
    price_history,
    _score,
    _cross_platform
  } = product;

  // Crawled products link to the real product page; the demo catalog's URLs are placeholders.
  const isLive = Boolean(scraped_at && source_url && source_url.startsWith('http'));
  const hasRating = Number(review_count) > 0 && Number(rating) > 0;
  const priceDrop = price_history?.price_dropped ? price_history.previous_price : null;

  // Format score to a readable decimal if present
  const formattedScore = _score !== undefined ? _score.toFixed(3) : null;

  // Build full image URL — image_url is like "/static/images/1234.jpg"
  const fullImageUrl = image_url
    ? (image_url.startsWith('http') ? image_url : `${BACKEND_URL}${image_url}`)
    : 'https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?auto=format&fit=crop&w=400&q=80';

  const sourceColor = storeColor(source);
  const sourceName = source_name || (source ? source.charAt(0).toUpperCase() + source.slice(1) : 'Unknown');
  const Wrapper = isLive ? 'a' : 'div';
  const wrapperProps = isLive
    ? { href: source_url, target: '_blank', rel: 'noopener noreferrer', title: `View on ${sourceName}` }
    : {};

  return (
    <Wrapper {...wrapperProps} className={`product-card glass ${!in_stock ? 'out-of-stock-card' : ''} ${isLive ? 'product-card-link' : ''}`}>
      <div className="image-container">
        <img
          src={fullImageUrl}
          alt={name}
          className="product-image"
          loading="lazy"
        />
        {/* Source badge */}
        <span className="source-badge" style={{ backgroundColor: sourceColor }}>
          {sourceName}
        </span>
        {!in_stock && <span className="out-of-stock-badge">Out of Stock</span>}
        {discount_percent > 0 && in_stock && (
          <span className="discount-badge">{discount_percent}% OFF</span>
        )}
        {priceDrop && in_stock && (
          <span className="price-drop-badge" title={`Was ₹${formatPrice(priceDrop)} before the last price change`}>
            ↓ Price drop
          </span>
        )}
        {formattedScore && (
          <span className="score-badge" title="Learning to Rank prediction score">
            LTR: {formattedScore}
          </span>
        )}
      </div>

      <div className="card-info">
        <span className="product-brand">{brand || 'Generic'}</span>
        <h4 className="product-name" title={name}>
          {name}
        </h4>

        {hasRating && (
          <div className="rating-row">
            <span className="rating-star-badge">
              <span className="star-icon-filled">★</span> {Number(rating).toFixed(1)}
            </span>
            <span className="reviews-count">({review_count} reviews)</span>
          </div>
        )}

        <div className="price-row">
          <span className="current-price">₹{formatPrice(price)}</span>
          {mrp > price && <span className="mrp-price">₹{formatPrice(mrp)}</span>}
          {priceDrop && <span className="was-price">was ₹{formatPrice(priceDrop)}</span>}
        </div>

        {isLive && available_sizes?.length > 0 && (
          <div className="sizes-row" title="Sizes in stock">
            {available_sizes.slice(0, 6).join(' · ')}
            {available_sizes.length > 6 ? ' …' : ''}
          </div>
        )}

        {/* Cross-platform pricing info */}
        {_cross_platform && (
          <div className="cross-platform-info">
            {_cross_platform.price_rank === 1 ? (
              <span className="best-price-badge">✓ Best Price</span>
            ) : (
              <span className="cheaper-elsewhere">
                ₹{formatPrice(_cross_platform.cheapest_price)} on {_cross_platform.cheapest_source}
              </span>
            )}
            <span className="available-on">
              {_cross_platform.available_on} sources
            </span>
          </div>
        )}

        {isLive && <span className="view-on-store">View on {sourceName} ↗</span>}
      </div>
    </Wrapper>
  );
}
