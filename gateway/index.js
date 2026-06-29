const express = require('express');
const rateLimit = require('express-rate-limit');
const axios = require('axios');
const cors = require('cors');
const morgan = require('morgan');

const app = express();
const PORT = process.env.PORT || 3001;
const SEARCH_SERVICE_URL = process.env.SEARCH_SERVICE_URL || 'http://localhost:8000';

app.use(cors());
app.use(morgan('dev'));
app.use(express.json());

// Rate limiting middleware: 60 requests per minute per IP
const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 60, // Limit each IP to 60 requests per minute
  message: { error: 'Too many requests, please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});

app.use(limiter);

// Proxy search endpoint
app.get('/api/search', async (req, res) => {
  try {
    const { q, page, limit, category, min_price, max_price, min_rating, in_stock_only } = req.query;
    
    if (!q) {
      return res.status(400).json({ error: 'Query parameter "q" is required' });
    }

    const response = await axios.get(`${SEARCH_SERVICE_URL}/search`, {
      params: { q, page, limit, category, min_price, max_price, min_rating, in_stock_only }
    });

    return res.json(response.data);
  } catch (error) {
    console.error('Error forwarding search request:', error.message);
    if (error.response) {
      console.error('Search service error status:', error.response.status);
      console.error('Search service error data:', error.response.data);
      return res.status(error.response.status).json(error.response.data);
    }
    if (error.code === 'ECONNREFUSED') {
      console.error('Cannot connect to search service at', SEARCH_SERVICE_URL);
      return res.status(503).json({ error: 'Search service is not running', service_url: SEARCH_SERVICE_URL });
    }
    console.error('Full error:', error);
    return res.status(500).json({ error: 'Search service is currently unavailable', details: error.message });
  }
});

// Admin trigger endpoints
app.post('/api/admin/rebuild_indices', async (req, res) => {
  try {
    const response = await axios.post(`${SEARCH_SERVICE_URL}/admin/rebuild_indices`);
    return res.json(response.data);
  } catch (error) {
    console.error('Rebuild indices error:', error.message);
    return res.status(500).json({ error: 'Failed to rebuild indices' });
  }
});

app.post('/api/admin/reload_ltr', async (req, res) => {
  try {
    const response = await axios.post(`${SEARCH_SERVICE_URL}/admin/reload_ltr`);
    return res.json(response.data);
  } catch (error) {
    console.error('Reload LTR model error:', error.message);
    return res.status(500).json({ error: 'Failed to reload LTR model' });
  }
});

app.get('/api/stats', async (req, res) => {
  try {
    const response = await axios.get(`${SEARCH_SERVICE_URL}/stats`);
    return res.json(response.data);
  } catch (error) {
    return res.status(500).json({ error: 'Failed to fetch search stats' });
  }
});

app.listen(PORT, () => {
  console.log(`API Gateway is listening on port ${PORT}`);
  console.log(`Proxying requests to search service at ${SEARCH_SERVICE_URL}`);
});
