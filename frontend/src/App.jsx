import React, { useState } from 'react';
import { SearchBar } from './components/SearchBar';
import { FilterPanel } from './components/FilterPanel';
import { ResultsList } from './components/ResultsList';
import { useSearch } from './hooks/useSearch';
import './App.css';

export default function App() {
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(1);
  const [filters, setFilters] = useState({
    category: null,
    maxPrice: null,
    minRating: null,
    inStockOnly: true
  });

  const { loading, error, results, total, intent, hasMore } = useSearch(
    query,
    page,
    filters
  );

  const handleSearch = (newQuery) => {
    if (newQuery !== query) {
      setQuery(newQuery);
      setPage(1);
    }
  };

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
    setPage(1);
  };

  const handleLoadMore = () => {
    setPage((prev) => prev + 1);
  };

  const totalResults = typeof total === 'number' ? total : 0;
  const activeCategory = filters.category || 'All';
  const resultLabel = totalResults === 1 ? 'item' : 'items';

  return (
    <div className="app-container">
      <header className="glass header-panel">
        <div className="logo">
          <span className="logo-icon">⚡</span>
          <div>
            <h1>
              Antigravity<span className="gradient-text">Search</span>
            </h1>
            <p className="logo-tagline">
              Hybrid product search with semantic relevance and instant results.
            </p>
          </div>
        </div>
        <div className="sys-status">
          <span className="status-indicator online"></span>
          <span className="status-text">Pipeline Active</span>
        </div>
      </header>

      <section className="hero-panel glass">
        <div className="hero-copy">
          <span className="hero-eyebrow">AI powered search</span>
          <h2>Discover products faster with hybrid ranking and smart filtering.</h2>
          <p>
            Search naturally and let the engine combine lexical precision,
            semantic matching, and learning-to-rank scoring to surface the most
            relevant items.
          </p>

          <div className="hero-features">
            <div className="hero-feature-card">
              <strong>Fast retrieval</strong>
              <span>Results appear instantly across the full catalog.</span>
            </div>
            <div className="hero-feature-card">
              <strong>Intent aware</strong>
              <span>Understanding color, price range, and category preferences.</span>
            </div>
            <div className="hero-feature-card">
              <strong>Clear filters</strong>
              <span>Refine product discovery with rating, price and stock options.</span>
            </div>
          </div>
        </div>

        <aside className="hero-metric-card glass">
          <div className="metric-top">
            <span className="metric-label">Catalog indexed</span>
            <span className="metric-value">{totalResults.toLocaleString()}</span>
          </div>
          <p>
            Explore curated products with rich ranking signals and intent-aware
            relevance.
          </p>
          <div className="metric-row">
            <div className="metric-item">
              <span className="metric-small">Category</span>
              <strong>{activeCategory}</strong>
            </div>
            <div className="metric-item">
              <span className="metric-small">Page</span>
              <strong>{page}</strong>
            </div>
          </div>
        </aside>
      </section>

      <SearchBar
        onSearch={handleSearch}
        intent={intent}
        totalResults={totalResults}
        loading={loading && page === 1}
      />

      <div className="main-content">
        <FilterPanel filters={filters} onChange={handleFilterChange} />

        <div className="results-wrapper">
          {error && (
            <div className="error-message glass">
              <span className="error-icon">⚠</span>
              <div>
                <strong>Query Execution Error:</strong> {error}
              </div>
            </div>
          )}

          <div className="results-header">
            <div>
              <h3>Search results</h3>
              <p>
                {totalResults.toLocaleString()} {resultLabel} matched for “
                {query || 'all products'}”
              </p>
            </div>
          </div>

          <ResultsList
            products={results}
            loading={loading}
            onLoadMore={handleLoadMore}
            hasMore={hasMore}
            page={page}
          />
        </div>
      </div>
    </div>
  );
}
