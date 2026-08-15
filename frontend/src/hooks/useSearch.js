import { useState, useEffect, useCallback } from 'react';

// In production, point to Render backend via VITE_GATEWAY_URL
// Locally, point directly to FastAPI on port 8000
const GATEWAY_URL = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000';

export function useSearch(query, page, filters) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState([]);
  const [total, setTotal] = useState(0);
  const [intent, setIntent] = useState(null);
  const [hasMore, setHasMore] = useState(true);

  // Clear results whenever query or filters change (new search context)
  useEffect(() => {
    setResults([]);
    setHasMore(true);
    setTotal(0);
  }, [query, filters.category, filters.minPrice, filters.maxPrice, filters.minRating, filters.inStockOnly]);

  const executeSearch = useCallback(async () => {
    if (!query || query.trim() === '') {
      setResults([]);
      setTotal(0);
      setIntent(null);
      setHasMore(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        q: query,
        page: page.toString(),
        limit: '20',
        in_stock_only: filters.inStockOnly.toString()
      });
      if (filters.category) params.set('category', filters.category);
      if (filters.minPrice) params.set('min_price', filters.minPrice.toString());
      if (filters.maxPrice) params.set('max_price', filters.maxPrice.toString());
      if (filters.minRating) params.set('min_rating', filters.minRating.toString());

      const response = await fetch(`${GATEWAY_URL}/api/search?${params}`);
      if (!response.ok) {
        throw new Error(`Search failed with status ${response.status}`);
      }

      const data = await response.json();
      setIntent(data.intent);
      setTotal(data.total);
      
      const newItems = data.results || [];
      
      setResults(prev => {
        if (page === 1) return newItems;
        // Avoid duplicate items by checking IDs
        const existingIds = new Set(prev.map(item => item._id));
        const filteredNewItems = newItems.filter(item => !existingIds.has(item._id));
        return [...prev, ...filteredNewItems];
      });

      // If we got fewer items than our page limit (20), we have reached the end
      setHasMore(newItems.length === 20);
    } catch (err) {
      console.error('Search request failed:', err);
      setError(err.message || 'Failed to retrieve search results');
    } finally {
      setLoading(false);
    }
  }, [query, page, filters]);

  useEffect(() => {
    executeSearch();
  }, [executeSearch]);

  return { loading, error, results, total, intent, hasMore };
}
