import React, { useState, useEffect, useRef, useMemo } from 'react';
import './SearchBar.css';

export function SearchBar({ onSearch, intent, totalResults, loading, image, onImageChange, photoEnabled = true, photoError = null }) {
  const [input, setInput] = useState('');
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef(null);

  // Debounce search input by 300ms
  useEffect(() => {
    const handler = setTimeout(() => {
      onSearch(input);
    }, 300);

    return () => {
      clearTimeout(handler);
    };
  }, [input, onSearch]);

  // Preview of the uploaded photo
  const previewUrl = useMemo(() => (image ? URL.createObjectURL(image) : null), [image]);
  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  const pickFile = (file) => {
    const name = (file?.name || '').toLowerCase();
    if (!file || !(file.type.startsWith('image/') || name.endsWith('.heic') || name.endsWith('.heif'))) return;
    // A new photo starts a fresh search: leftover text would silently narrow it.
    setInput('');
    onSearch('');
    onImageChange(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (photoEnabled) pickFile(e.dataTransfer.files?.[0]);
  };

  // The server builds its photo index after starting; say so instead of looking broken.
  const photoWarming = !photoEnabled && Boolean(photoError?.startsWith('getting ready'));
  const photoTitle = photoEnabled
    ? 'Search by photo: upload or drop an image'
    : photoWarming
      ? `Photo search is ${photoError}`
      : `Photo search is unavailable on this server${photoError ? `: ${photoError}` : ''}`;

  return (
    <div className="search-section">
      <div
        className={`search-bar-wrapper glass ${dragging ? 'dragging' : ''}`}
        onDragOver={(e) => {
          if (!photoEnabled) return;
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <svg className="search-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
        </svg>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            image
              ? "Optional: narrow it down (e.g. 'in blue' or 'under 1500')"
              : "Search products (e.g. 'red kurta under 500' or 'linen shirt size L')"
          }
          className="search-input"
          aria-label="Search products"
        />
        {loading && <div className="loader-ring"></div>}
        <button
          type="button"
          className="photo-btn"
          onClick={() => fileInput.current?.click()}
          disabled={!photoEnabled}
          title={photoTitle}
          aria-label={photoTitle}
        >
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8"
              d="M3 8.5A2.5 2.5 0 015.5 6h1.6l1.2-1.8A1.5 1.5 0 019.55 3.5h4.9a1.5 1.5 0 011.25.7L16.9 6h1.6A2.5 2.5 0 0121 8.5v9a2.5 2.5 0 01-2.5 2.5h-13A2.5 2.5 0 013 17.5v-9z" />
            <circle cx="12" cy="13" r="3.6" strokeWidth="1.8" />
          </svg>
          <span className="photo-btn-label">Photo</span>
        </button>
        <input
          ref={fileInput}
          type="file"
          accept="image/*,.heic,.heif"
          hidden
          onChange={(e) => {
            pickFile(e.target.files?.[0]);
            e.target.value = '';
          }}
        />
      </div>

      {photoWarming && !image && (
        <p className="photo-warming" role="status">
          Photo search is {photoError}. It switches on by itself.
        </p>
      )}

      {image && (
        <div className="photo-chip glass animate-fade-in">
          <img src={previewUrl} alt="Your photo" />
          <div className="photo-chip-text">
            <strong>Searching by photo</strong>
            <span>Showing visually similar products. Type above to narrow it down.</span>
          </div>
          <button type="button" className="photo-chip-clear" onClick={() => onImageChange(null)} aria-label="Remove photo">
            ×
          </button>
        </div>
      )}

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
            {intent.size && (
              <span className="intent-badge size">
                Size: <strong>{intent.size}</strong>
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
