import React, { useState } from 'react';
import { API_URL, createPriceAlert } from '../api';
import './ProductCard.css';

const SOURCE_COLORS = {
  amazon: '#FF9900',
  myntra: '#FF3F6C',
  ajio: '#3880FF',
  flipkart: '#2874F0',
};

// Built-in placeholder, so a missing product image never needs the network.
const PLACEHOLDER_IMAGE =
  'data:image/svg+xml;utf8,' +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400"><rect width="400" height="400" fill="#1e2638"/>' +
      '<path d="M150 120l50-25 50 25 45 40-32 34-23-17v128H160V177l-23 17-32-34z" fill="#334155"/></svg>'
  );

// Stable colour per store for crawled sources we don't have a brand colour for.
function storeColor(source) {
  if (SOURCE_COLORS[source]) return SOURCE_COLORS[source];
  let hash = 0;
  for (const ch of source || '') hash = (hash * 31 + ch.charCodeAt(0)) % 360;
  return `hsl(${hash}, 55%, 42%)`;
}

const formatPrice = (value) => Number(value).toLocaleString('en-IN', { maximumFractionDigits: 0 });

function readSavedEmail() {
  try {
    return localStorage.getItem('priceAlertEmail') || '';
  } catch {
    return '';
  }
}

function saveEmail(email) {
  try {
    localStorage.setItem('priceAlertEmail', email);
  } catch {
    /* private mode etc. */
  }
}

// Small line chart of the prices recorded by the daily crawl.
function PriceTrend({ history, price }) {
  const series = history?.series || [];
  if (series.length < 2) return null;
  const prices = series.map(([, p]) => p);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const width = 96;
  const height = 26;
  const points = prices
    .map((p, i) => {
      const x = (i / (prices.length - 1)) * width;
      const y = max === min ? height / 2 : height - 3 - ((p - min) / (max - min)) * (height - 6);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  const lowest = history.lowest_30d ?? min;
  const label = history.is_lowest_30d ? 'Lowest price in 30 days' : `30-day low ₹${formatPrice(lowest)}`;
  const trend = price < prices[0] ? 'down' : price > prices[0] ? 'up' : 'flat';

  return (
    <div className={`price-trend ${trend}`} title={`Price on ${series.length} checks since ${series[0][0]}`}>
      <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} aria-hidden="true">
        <polyline points={points} fill="none" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      </svg>
      <span>{label}</span>
    </div>
  );
}

function AlertForm({ product, onDone }) {
  const [email, setEmail] = useState(readSavedEmail);
  const [target, setTarget] = useState(() => Math.max(1, Math.floor((product.price * 0.9) / 10) * 10));
  const [state, setState] = useState({ status: 'idle', message: '' });

  const submit = async (e) => {
    e.preventDefault();
    setState({ status: 'saving', message: '' });
    try {
      const data = await createPriceAlert(product._id, Number(target), email.trim());
      saveEmail(email.trim());
      setState({ status: 'done', message: data.message });
    } catch (err) {
      setState({ status: 'error', message: err.message });
    }
  };

  if (state.status === 'done') {
    return (
      <div className="alert-form done" role="status">
        <span>✓ {state.message}</span>
        <button type="button" className="link-btn" onClick={onDone}>Close</button>
      </div>
    );
  }

  return (
    <form className="alert-form" onSubmit={submit}>
      <label className="alert-row">
        <span>Email me when it's ₹</span>
        <input
          type="number"
          min="1"
          max={Math.max(1, product.price - 1)}
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          required
          aria-label="Target price in rupees"
        />
        <span>or less</span>
      </label>
      <input
        type="email"
        placeholder="you@example.com"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        required
        aria-label="Email address"
      />
      <div className="alert-actions">
        <button type="submit" className="alert-submit" disabled={state.status === 'saving'}>
          {state.status === 'saving' ? 'Saving…' : 'Set alert'}
        </button>
        <button type="button" className="link-btn" onClick={onDone}>Cancel</button>
      </div>
      {state.status === 'error' && <p className="alert-error" role="alert">{state.message}</p>}
      <p className="alert-note">Prices are checked after each daily crawl. One email, then the alert switches off.</p>
    </form>
  );
}

export function ProductCard({ product, onOpen, alertsEnabled = false }) {
  const {
    name,
    brand,
    price,
    mrp,
    discount_percent,
    rating,
    review_count,
    in_stock,
    image_url,
    source,
    source_name,
    source_url,
    scraped_at,
    available_sizes,
    price_history,
    _score,
    _cross_platform
  } = product;
  const [alertOpen, setAlertOpen] = useState(false);

  // Crawled products link to the real product page; the demo catalog's URLs are placeholders.
  const isLive = Boolean(scraped_at && source_url && source_url.startsWith('http'));
  const hasRating = Number(review_count) > 0 && Number(rating) > 0;
  const priceDrop = price_history?.price_dropped ? price_history.previous_price : null;

  // Photo results show visual similarity; text results show the re-ranker's score.
  const isVisual = product._score_kind === 'visual';
  const formattedScore = isVisual
    ? `Match ${Math.round((product._similarity ?? _score) * 100)}%`
    : _score !== undefined ? `LTR: ${_score.toFixed(3)}` : null;

  // Demo catalog images are served by the API ("/static/images/1234.jpg"); crawled ones are full URLs.
  const fullImageUrl = image_url
    ? (image_url.startsWith('http') ? image_url : `${API_URL}${image_url}`)
    : PLACEHOLDER_IMAGE;

  const sourceColor = storeColor(source);
  const sourceName = source_name || (source ? source.charAt(0).toUpperCase() + source.slice(1) : 'Unknown');
  const linkProps = isLive
    ? { href: source_url, target: '_blank', rel: 'noopener noreferrer', onClick: () => onOpen?.(product) }
    : null;

  const image = (
    <>
      <img
        src={fullImageUrl}
        alt={name}
        className="product-image"
        loading="lazy"
        onError={(e) => {
          if (e.currentTarget.src !== PLACEHOLDER_IMAGE) e.currentTarget.src = PLACEHOLDER_IMAGE;
        }}
      />
      <span className="source-badge" style={{ backgroundColor: sourceColor }}>
        {sourceName}
      </span>
      {!in_stock && <span className="out-of-stock-badge">Out of Stock</span>}
      {discount_percent > 0 && in_stock && (
        <span className="discount-badge">{discount_percent}% OFF</span>
      )}
      {priceDrop && in_stock && (
        <span className="price-drop-badge" title={`Was ₹${formatPrice(priceDrop)} before the last price change`}>
          ↓ Price drop
        </span>
      )}
      {formattedScore && (
        <span
          className="score-badge"
          title={isVisual ? 'How similar this looks to your photo' : 'Learning to Rank prediction score'}
        >
          {formattedScore}
        </span>
      )}
    </>
  );

  return (
    <div className={`product-card glass ${!in_stock ? 'out-of-stock-card' : ''}`}>
      {linkProps ? (
        <a {...linkProps} className="image-container" aria-label={`${name} on ${sourceName}`}>{image}</a>
      ) : (
        <div className="image-container">{image}</div>
      )}

      <div className="card-info">
        <span className="product-brand">{brand || 'Generic'}</span>
        <h4 className="product-name" title={name}>
          {linkProps ? <a {...linkProps} className="product-name-link">{name}</a> : name}
        </h4>

        {hasRating && (
          <div className="rating-row">
            <span className="rating-star-badge">
              <span className="star-icon-filled">★</span> {Number(rating).toFixed(1)}
            </span>
            <span className="reviews-count">({review_count} reviews)</span>
          </div>
        )}

        <div className="price-row">
          <span className="current-price">₹{formatPrice(price)}</span>
          {mrp > price && <span className="mrp-price">₹{formatPrice(mrp)}</span>}
          {priceDrop && <span className="was-price">was ₹{formatPrice(priceDrop)}</span>}
        </div>

        <PriceTrend history={price_history} price={price} />

        {isLive && available_sizes?.length > 0 && (
          <div className="sizes-row" title="Sizes in stock">
            {available_sizes.slice(0, 6).join(' · ')}
            {available_sizes.length > 6 ? ' …' : ''}
          </div>
        )}

        {/* Cross-platform pricing info */}
        {_cross_platform && (
          <div className="cross-platform-info">
            {_cross_platform.price_rank === 1 ? (
              <span className="best-price-badge">✓ Best Price</span>
            ) : (
              <span className="cheaper-elsewhere">
                ₹{formatPrice(_cross_platform.cheapest_price)} on {_cross_platform.cheapest_source}
              </span>
            )}
            <span className="available-on">
              {_cross_platform.available_on} sources
            </span>
          </div>
        )}

        {isLive && (
          <div className="card-actions">
            <a {...linkProps} className="view-on-store">View on {sourceName} ↗</a>
            {alertsEnabled && in_stock && !alertOpen && (
              <button type="button" className="alert-btn" onClick={() => setAlertOpen(true)} title="Get an email when the price drops">
                🔔 Price alert
              </button>
            )}
          </div>
        )}
        {alertOpen && <AlertForm product={product} onDone={() => setAlertOpen(false)} />}
      </div>
    </div>
  );
}
