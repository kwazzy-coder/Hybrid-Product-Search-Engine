import React, { useCallback, useEffect, useRef, useState } from 'react';
import { SearchBar } from './components/SearchBar';
import { FilterPanel } from './components/FilterPanel';
import { ResultsList } from './components/ResultsList';
import { LogoMark } from './components/Logo';
import { useSearch } from './hooks/useSearch';
import { API_URL, logClick } from './api';
import './App.css';

export default function App() {
  const [query, setQuery] = useState('');
  const [image, setImage] = useState(null);
  const [page, setPage] = useState(1);
  const [filters, setFilters] = useState({
    category: null,
    maxPrice: null,
    minRating: null,
    inStockOnly: true,
    size: null
  });

  // Which catalog is indexed (live crawl or demo), its stores, categories, sizes and features.
  // server: 'connecting' until /api/stats answers, 'offline' after a few failed tries.
  // Keeps retrying, so a backend that's still starting (it loads its models first)
  // or is started after the page was opened is picked up without a reload.
  const [catalog, setCatalog] = useState(null);
  const [features, setFeatures] = useState(null);
  const [server, setServer] = useState('connecting');
  useEffect(() => {
    let cancelled = false;
    let attempts = 0;
    let timer;
    const load = () => {
      fetch(`${API_URL}/api/stats`)
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((data) => {
          if (cancelled) return;
          setCatalog(data.catalog || null);
          setFeatures(data.features || null);
          setServer('ready');
        })
        .catch(() => {
          if (cancelled) return;
          attempts += 1;
          if (attempts >= 3) setServer('offline');
          timer = setTimeout(load, attempts < 10 ? 2000 : 5000);
        });
    };
    load();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);
  const serverReady = server === 'ready';
  const isLive = serverReady && catalog?.mode === 'live';
  const serverMessage =
    server === 'offline'
      ? `Can't reach the search server${API_URL ? ` at ${API_URL}` : ''}. Start the backend and this page will connect by itself.`
      : 'Connecting to the search server…';
  const storeCount = catalog?.stores?.length || 0;
  const storeNames = (catalog?.stores || []).map((s) => s.name).join(', ');
  const updated = catalog?.last_scraped_at
    ? new Date(catalog.last_scraped_at).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
    : null;

  const { loading, error, notice, results, total, intent, hasMore } = useSearch(
    query,
    image,
    page,
    filters,
    serverReady
  );

  // Stable callback (the search bar debounces on it); a ref avoids re-running for the same text.
  const queryRef = useRef(query);
  const handleSearch = useCallback((newQuery) => {
    if (newQuery !== queryRef.current) {
      queryRef.current = newQuery;
      setQuery(newQuery);
      setPage(1);
    }
  }, []);

  const handleImageChange = (file) => {
    setImage(file);
    setPage(1);
  };

  // Log which result was opened (and where it was shown) so the re-ranker can learn from it.
  const handleProductOpen = (product) => {
    logClick({
      query: image ? query.trim() || '[photo]' : query,
      product,
      shownIds: results.map((r) => r._id),
    });
  };

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
    setPage(1);
  };

  const handleLoadMore = () => {
    setPage((prev) => prev + 1);
  };

  const totalResults = typeof total === 'number' ? total : 0;
  const resultLabel = totalResults === 1 ? 'item' : 'items';
  const updatedDate = catalog?.last_scraped_at
    ? new Date(catalog.last_scraped_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })
    : null;

  return (
    <div className="app-container">
      <header className="glass header-panel">
        <div className="logo">
          <LogoMark size={46} animate title="" />
          <h1 className="wordmark">dhaaga</h1>
        </div>
        <div
          className="sys-status"
          title={isLive ? `Prices from ${storeNames}` : serverReady ? undefined : serverMessage}
          role="status"
        >
          <span className={`status-indicator ${isLive ? 'online' : server === 'offline' ? 'offline' : ''}`}></span>
          <span className="status-text">
            {isLive
              ? `${catalog.products.toLocaleString('en-IN')} products from ${storeCount} ${storeCount === 1 ? 'store' : 'stores'}`
              : serverReady
                ? 'Demo catalog'
                : server === 'offline'
                  ? 'Search server offline'
                  : 'Connecting…'}
          </span>
        </div>
      </header>

      <section className="hero-panel glass">
        <div className="hero-copy">
          <h2>Search India’s fashion brands in one place.</h2>
          <p>
            Describe what you want, like “linen shirt size L under 2000”, or search
            with a photo. Every result links to the brand’s own site at today’s price.
          </p>

          <div className="hero-features">
            <div className="hero-feature-card">
              <strong>Search by photo</strong>
              <span>Upload a picture and find pieces that look like it.</span>
            </div>
            <div className="hero-feature-card">
              <strong>Your size, in stock</strong>
              <span>Pick a size and see only what you can buy right now.</span>
            </div>
            <div className="hero-feature-card">
              <strong>Price drop alerts</strong>
              <span>Get an email when something you like gets cheaper.</span>
            </div>
          </div>
        </div>

        <aside className="hero-metric-card glass">
          <div className="metric-top">
            <span className="metric-label">Products tracked</span>
            <span className="metric-value">{serverReady && catalog ? catalog.products.toLocaleString('en-IN') : '…'}</span>
          </div>
          <p>
            {isLive
              ? `Prices and stock from ${storeNames}, checked every day.`
              : serverReady
                ? 'A demo catalog with synthetic prices. Run the crawler to track real stores.'
                : serverMessage}
          </p>
          <div className="metric-row">
            <div className="metric-item">
              <span className="metric-small">Stores</span>
              <strong>{isLive ? storeCount : '–'}</strong>
            </div>
            <div className="metric-item">
              <span className="metric-small">Prices updated</span>
              <strong>{isLive && updatedDate ? updatedDate : '–'}</strong>
            </div>
          </div>
        </aside>
      </section>

      <SearchBar
        onSearch={handleSearch}
        intent={intent}
        totalResults={totalResults}
        loading={loading && page === 1}
        image={image}
        onImageChange={handleImageChange}
        photoEnabled={features ? features.visual_search_ready : true}
        photoError={features?.visual_search_error}
      />

      <div className="main-content">
        <FilterPanel
          filters={filters}
          onChange={handleFilterChange}
          categories={(catalog?.top_categories || []).map((c) => c.name)}
          sizes={catalog?.top_sizes || []}
          showRating={catalog ? catalog.has_ratings !== false : true}
        />

        <div className="results-wrapper">
          {notice && (
            <div className="error-message glass notice-message">
              <span className="error-icon">ℹ</span>
              <div>{notice}</div>
            </div>
          )}

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
                {image ? (query.trim() ? `${query.trim()} + your photo` : 'your photo') : query || 'all products'}”
              </p>
            </div>
          </div>

          {serverReady && (
            <p className="catalog-disclaimer">
              {isLive
                ? `Live prices from ${storeNames || 'brand websites'}${updated ? `, updated ${updated}` : ''}. Click a product to open it on the store's site.`
                : 'Demo catalog with synthetic prices and stores. Run the crawler to index real products.'}
            </p>
          )}

          <ResultsList
            products={results}
            loading={loading}
            onLoadMore={handleLoadMore}
            hasMore={hasMore}
            page={page}
            onProductOpen={handleProductOpen}
            alertsEnabled={Boolean(isLive && features?.alerts_enabled)}
          />
        </div>
      </div>
    </div>
  );
}
