import React from 'react';
import { ProductCard } from './ProductCard';
import './ResultsList.css';

export function ResultsList({ products, loading, onLoadMore, hasMore, page }) {
  // Skeleton Loader elements
  const Skeletons = () => (
    <div className="products-grid">
      {Array.from({ length: 8 }).map((_, idx) => (
        <div key={idx} className="product-card glass skeleton-card">
          <div className="skeleton-image skeleton"></div>
          <div className="skeleton-info">
            <div className="skeleton-line brand skeleton"></div>
            <div className="skeleton-line name skeleton"></div>
            <div className="skeleton-line rating skeleton"></div>
            <div className="skeleton-line price skeleton"></div>
          </div>
        </div>
      ))}
    </div>
  );

  if (loading && page === 1) {
    return <Skeletons />;
  }

  if (products.length === 0) {
    return (
      <div className="empty-results glass animate-fade-in">
        <svg className="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
        </svg>
        <h3>No Products Found</h3>
        <p>Try adjusting your filters or expanding your search query details.</p>
      </div>
    );
  }

  return (
    <div className="results-container animate-fade-in">
      <div className="products-grid">
        {products.map((product) => (
          <ProductCard key={product._id} product={product} />
        ))}
      </div>

      {hasMore && (
        <div className="load-more-wrapper">
          <button
            onClick={onLoadMore}
            disabled={loading}
            className="glow-btn load-more-btn"
          >
            {loading ? (
              <>
                <div className="loader-ring white-loader"></div>
                Loading...
              </>
            ) : (
              'Load More Products'
            )}
          </button>
        </div>
      )}
    </div>
  );
}
