// Where the API lives. An empty VITE_GATEWAY_URL means "same origin", which is how the
// single-container deployment serves the site and the API from one URL.
export const API_URL = import.meta.env.VITE_GATEWAY_URL ?? 'http://localhost:8000';

// Record which result a shopper opened, so the re-ranker can learn from real clicks.
// Fire-and-forget: it must never slow down or block opening the product.
export function logClick({ query, product, shownIds }) {
  if (!product?._query_id) return;
  try {
    fetch(`${API_URL}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      keepalive: true,
      body: JSON.stringify({
        query,
        clicked_id: product._id,
        query_id: product._query_id,
        position: product._position,
        shown_results: shownIds.slice(0, 50),
      }),
    }).catch(() => {});
  } catch {
    /* ignore */
  }
}

export async function createPriceAlert(productId, targetPrice, email) {
  const res = await fetch(`${API_URL}/api/alerts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ product_id: productId, target_price: targetPrice, email }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail[0]?.msg : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}
