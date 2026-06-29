import React from 'react';
import './ProductCard.css';

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

  return (
    <div className={`product-card glass ${!in_stock ? 'out-of-stock-card' : ''}`}>
      <div className="image-container">
        <img
          src={image_url || 'https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?auto=format&fit=crop&w=400&q=80'}
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
      </div>
    </div>
  );
}
