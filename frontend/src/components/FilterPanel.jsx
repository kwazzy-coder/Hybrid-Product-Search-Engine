import React from 'react';
import './FilterPanel.css';

const DEFAULT_CATEGORIES = ['kurta', 'shirt', 'tshirt', 'jeans', 'dress', 'shoes'];
const MAX_PRICE = 10000;

const label = (cat) => (cat === 'tshirt' ? 'T-shirt' : cat.charAt(0).toUpperCase() + cat.slice(1));

export function FilterPanel({ filters, onChange, categories, sizes = [], showRating = true }) {
  // Use the categories that actually exist in the indexed catalog when the API reports them.
  const CATEGORIES = ['All', ...(categories?.length ? categories : DEFAULT_CATEGORIES)];

  const handleCategoryClick = (cat) => {
    onChange({
      ...filters,
      category: cat === 'All' ? null : cat.toLowerCase()
    });
  };

  const handlePriceChange = (e) => {
    onChange({
      ...filters,
      maxPrice: Number(e.target.value) >= MAX_PRICE ? null : parseFloat(e.target.value)
    });
  };

  const handleRatingClick = (rating) => {
    onChange({
      ...filters,
      minRating: filters.minRating === rating ? null : rating
    });
  };

  const handleSizeClick = (size) => {
    onChange({
      ...filters,
      size: filters.size === size ? null : size
    });
  };

  const handleStockToggle = () => {
    onChange({
      ...filters,
      inStockOnly: !filters.inStockOnly
    });
  };

  const resetFilters = () => {
    onChange({
      category: null,
      maxPrice: null,
      minRating: null,
      inStockOnly: true,
      size: null
    });
  };

  return (
    <div className="filter-panel glass">
      <div className="filter-header">
        <h3>Filters</h3>
        <button onClick={resetFilters} className="clear-btn">
          Clear All
        </button>
      </div>

      {/* Category Section */}
      <div className="filter-section">
        <h4>Category</h4>
        <div className="category-chips">
          {CATEGORIES.map((cat) => {
            const isSelected =
              cat === 'All'
                ? filters.category === null
                : filters.category === cat.toLowerCase();
            return (
              <button
                key={cat}
                onClick={() => handleCategoryClick(cat)}
                className={`chip-btn ${isSelected ? 'selected' : ''}`}
              >
                {cat === 'All' ? cat : label(cat)}
              </button>
            );
          })}
        </div>
      </div>

      {/* Price Section */}
      <div className="filter-section">
        <div className="section-header-row">
          <h4>Max Price</h4>
          <span className="price-display">
            {filters.maxPrice ? `₹${filters.maxPrice.toLocaleString('en-IN')}` : 'Any'}
          </span>
        </div>
        <input
          type="range"
          min="200"
          max={MAX_PRICE}
          step="100"
          value={filters.maxPrice || MAX_PRICE}
          onChange={handlePriceChange}
          className="range-slider"
        />
        <div className="range-labels">
          <span>₹200</span>
          <span>₹{MAX_PRICE.toLocaleString('en-IN')}+</span>
        </div>
      </div>

      {/* Size Section: only items in stock in that size (one-size items always show) */}
      {sizes.length > 0 && (
        <div className="filter-section">
          <div className="section-header-row">
            <h4>Size in stock</h4>
            {filters.size && <span className="price-display">{filters.size}</span>}
          </div>
          <div className="category-chips">
            {sizes.map((size) => (
              <button
                key={size}
                onClick={() => handleSizeClick(size)}
                className={`chip-btn ${filters.size === size ? 'selected' : ''}`}
                aria-pressed={filters.size === size}
              >
                {size}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Rating Section (hidden when the catalog has no ratings, e.g. crawled stores) */}
      {showRating && (
      <div className="filter-section">
        <h4>Minimum Rating</h4>
        <div className="rating-selector">
          {[4, 3, 2].map((stars) => {
            const isSelected = filters.minRating === stars;
            return (
              <button
                key={stars}
                onClick={() => handleRatingClick(stars)}
                className={`rating-btn ${isSelected ? 'selected' : ''}`}
              >
                <span className="stars-row">
                  {Array.from({ length: 5 }).map((_, idx) => (
                    <span
                      key={idx}
                      className={`star-icon ${idx < stars ? 'filled' : 'empty'}`}
                    >
                      ★
                    </span>
                  ))}
                </span>
                <span className="rating-label">& Up</span>
              </button>
            );
          })}
        </div>
      </div>
      )}

      {/* Availability Section */}
      <div className="filter-section">
        <label className="toggle-label">
          <span>In Stock Only</span>
          <div className="toggle-switch">
            <input
              type="checkbox"
              checked={filters.inStockOnly}
              onChange={handleStockToggle}
            />
            <span className="switch-slider"></span>
          </div>
        </label>
      </div>
    </div>
  );
}
