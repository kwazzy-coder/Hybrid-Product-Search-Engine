import React from 'react';
import './ProductCard.css';

const BACKEND_URL = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000';

const marketplaceSearchUrls = (name) => {
  const query = encodeURIComponent(name || 'fashion');
  return [
    { label: 'Flipkart', url: `https://www.flipkart.com/search?q=${query}` },
    { label: 'Amazon', url: `https://www.amazon.in/s?k=${query}` },
    { label: 'Myntra', url: `https://www.myntra.com/${query}` }
  ];
};

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
    _score
  } = product;

  // Format score to a readable decimal if present
  const formattedScore = _score !== undefined ? _score.toFixed(3) : null;

  // Build full image URL — image_url is like "/static/images/1234.jpg"
  const fullImageUrl = image_url
    ? (image_url.startsWith('http') ? image_url : `${BACKEND_URL}${image_url}`)
    : 'https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?auto=format&fit=crop&w=400&q=80';
  const marketplaceLinks = marketplaceSearchUrls(name);

  return (
    <div className={`product-card glass ${!in_stock ? 'out-of-stock-card' : ''}`}>
      <div className="image-container">
        <img
          src={fullImageUrl}
          alt={name}
          className="product-image"
          loading="lazy"
        />
        {!in_stock && <span className="out-of-stock-badge">Out of Stock</span>}
        {discount_percent > 0 && in_stock && (
          <span className="discount-badge">{discount_percent}% OFF</span>
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

        <div className="rating-row">
          <span className="rating-star-badge">
            <span className="star-icon-filled">★</span> {rating.toFixed(1)}
          </span>
          <span className="reviews-count">({review_count} reviews)</span>
        </div>

        <div className="price-row">
          <span className="current-price">₹{price}</span>
          {mrp > price && <span className="mrp-price">₹{mrp}</span>}
        </div>
        <div className="marketplace-links" aria-label={`Find ${name} on marketplaces`}>
          {marketplaceLinks.map(({ label, url }) => (
            <a
              key={label}
              className="marketplace-link"
              href={url}
              target="_blank"
              rel="noreferrer"
            >
              {label}
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
