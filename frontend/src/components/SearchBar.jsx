import React, { useState, useEffect } from 'react';
import './SearchBar.css';

export function SearchBar({ onSearch, intent, totalResults, loading }) {
  const [input, setInput] = useState('');

  // Debounce search input by 300ms
  useEffect(() => {
    const handler = setTimeout(() => {
      onSearch(input);
    }, 300);

    return () => {
      clearTimeout(handler);
    };
  }, [input, onSearch]);

  return (
    <div className="search-section">
      <div className="search-bar-wrapper glass">
        <svg className="search-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
        </svg>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Search products (e.g. 'red kurta under 500' or 'blue shirt')"
          className="search-input"
        />
        {loading && <div className="loader-ring"></div>}
      </div>

      {intent && (
        <div className="intent-display glass animate-fade-in">
          <div className="intent-meta">
            <span className="results-count">
              Found <strong>{totalResults}</strong> matches
            </span>
            {intent.category && (
              <span className="intent-badge category">
                Category: <strong>{intent.category}</strong>
              </span>
            )}
            {intent.color && intent.color.length > 0 && (
              <span className="intent-badge color">
                Color: <strong>{intent.color.join(', ')}</strong>
              </span>
            )}
            {intent.max_price && (
              <span className="intent-badge price">
                Max Price: <strong>₹{intent.max_price}</strong>
              </span>
            )}
          </div>
          
          {intent.expanded_terms && intent.expanded_terms.length > 0 && (
            <div className="expanded-terms">
              <span className="terms-label">Query Expansion Synonyms:</span>
              <div className="terms-container">
                {intent.expanded_terms.map((term, i) => (
                  <span key={i} className="term-chip">
                    {term}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
