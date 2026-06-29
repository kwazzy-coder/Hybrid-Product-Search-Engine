import React from 'react';
import './FilterPanel.css';

const CATEGORIES = ['All', 'Kurta', 'Shirt', 'Tshirt', 'Jeans', 'Dress', 'Shoes'];

export function FilterPanel({ filters, onChange }) {
  const handleCategoryClick = (cat) => {
    onChange({
      ...filters,
      category: cat === 'All' ? null : cat.toLowerCase()
    });
  };

  const handlePriceChange = (e) => {
    onChange({
      ...filters,
      maxPrice: e.target.value === '3000' ? null : parseFloat(e.target.value)
    });
  };

  const handleRatingClick = (rating) => {
    onChange({
      ...filters,
      minRating: filters.minRating === rating ? null : rating
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
      inStockOnly: true
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
                {cat}
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
            {filters.maxPrice ? `₹${filters.maxPrice}` : 'Any'}
          </span>
        </div>
        <input
          type="range"
          min="200"
          max="3000"
          step="50"
          value={filters.maxPrice || 3000}
          onChange={handlePriceChange}
          className="range-slider"
        />
        <div className="range-labels">
          <span>₹200</span>
          <span>₹3000+</span>
        </div>
      </div>

      {/* Rating Section */}
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
