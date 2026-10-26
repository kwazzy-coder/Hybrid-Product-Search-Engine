import { useState, useEffect, useCallback } from 'react';
import { API_URL } from '../api';

const PAGE_SIZE = 20;

// Text search (GET /api/search) or photo search (POST /api/search/image, with optional text).
// serverReady: when the backend comes up, a search that failed while it was down runs again.
export function useSearch(query, image, page, filters, serverReady = true) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [results, setResults] = useState([]);
  const [total, setTotal] = useState(0);
  const [intent, setIntent] = useState(null);
  const [hasMore, setHasMore] = useState(true);

  // Clear results whenever the query, photo or filters change (new search context)
  useEffect(() => {
    setResults([]);
    setHasMore(true);
    setTotal(0);
  }, [query, image, filters.category, filters.minPrice, filters.maxPrice, filters.minRating, filters.inStockOnly, filters.size]);

  const executeSearch = useCallback(async () => {
    const text = (query || '').trim();
    if (!text && !image) {
      setResults([]);
      setTotal(0);
      setIntent(null);
      setNotice(null);
      setHasMore(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        page: page.toString(),
        limit: String(PAGE_SIZE),
        in_stock_only: filters.inStockOnly.toString(),
      });
      if (filters.category) params.set('category', filters.category);
      if (filters.minPrice) params.set('min_price', filters.minPrice.toString());
      if (filters.maxPrice) params.set('max_price', filters.maxPrice.toString());
      if (filters.minRating) params.set('min_rating', filters.minRating.toString());
      if (filters.size) params.set('size', filters.size);

      let response;
      if (image) {
        const form = new FormData();
        form.append('file', image);
        if (text) form.append('q', text);
        response = await fetch(`${API_URL}/api/search/image?${params}`, { method: 'POST', body: form });
      } else {
        params.set('q', text);
        response = await fetch(`${API_URL}/api/search?${params}`);
      }
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Search failed with status ${response.status}`);
      }

      const data = await response.json();
      setIntent(data.intent);
      setTotal(data.total);
      setNotice(data.notice || null);

      const newItems = data.results || [];
      setResults((prev) => {
        if (page === 1) return newItems;
        const existingIds = new Set(prev.map((item) => item._id));
        return [...prev, ...newItems.filter((item) => !existingIds.has(item._id))];
      });
      setHasMore(newItems.length === PAGE_SIZE);
    } catch (err) {
      console.error('Search request failed:', err);
      // fetch() throws a TypeError when the server can't be reached at all.
      let message = err.message || 'Failed to retrieve search results';
      if (err instanceof TypeError) {
        message = serverReady
          ? 'Lost the connection to the search server. Try again in a moment.'
          : "Can't reach the search server yet. If you're running it locally, start the backend; this search runs again once it connects.";
      }
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [query, image, page, filters, serverReady]);

  useEffect(() => {
    executeSearch();
  }, [executeSearch]);

  return { loading, error, notice, results, total, intent, hasMore };
}
