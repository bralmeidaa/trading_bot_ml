const API_BASE_URL = '/api';

class ApiService {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // Convenience methods for different HTTP methods
  async get(endpoint) {
    return this.request(endpoint, { method: 'GET' });
  }

  async post(endpoint, data) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async put(endpoint, data) {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async delete(endpoint) {
    return this.request(endpoint, { method: 'DELETE' });
  }

  // System Status
  async getSystemStatus() {
    return this.get('/status');
  }

  async startSystem() {
    return this.post('/start');
  }

  async stopSystem() {
    return this.post('/stop');
  }

  async emergencyStop() {
    return this.post('/emergency-stop');
  }

  // Performance Metrics
  async getMetrics() {
    return this.get('/metrics');
  }

  async getEquity() {
    return this.get('/equity');
  }

  // Bot Management
  async getBots() {
    return this.get('/bots');
  }

  async updateBotConfig(botId, config) {
    return this.put(`/bots/${botId}/config`, config);
  }

  async toggleBot(botId, enabled) {
    return this.post(`/bots/${botId}/toggle`, { enabled });
  }

  // New Bot Management Endpoints
  async getAvailableSymbols() {
    return this.get('/bots/available-symbols');
  }

  async getAvailableTimeframes() {
    return this.get('/bots/available-timeframes');
  }

  async addBot(botConfig) {
    return this.post('/bots/add', botConfig);
  }

  async updateBot(botIndex, botConfig) {
    return this.put(`/bots/update/${botIndex}`, botConfig);
  }

  async removeBot(botIndex) {
    return this.delete(`/bots/remove/${botIndex}`);
  }

  async getBotCount() {
    return this.get('/bots/count');
  }

  // Trades and Logs
  async getRecentTrades() {
    return this.get('/trades/recent');
  }

  async getLogs() {
    return this.get('/logs');
  }

  // Configuration
  async getConfig() {
    return this.get('/config/full');
  }

  async updateConfig(config) {
    return this.post('/config', config);
  }

  // Backtest
  async runBacktest() {
    return this.post('/backtest');
  }

  // Health Check
  async healthCheck() {
    return this.get('/health');
  }

  // Performance Metrics
  async getPerformanceMetrics() {
    return this.get('/performance-metrics');
  }
}

export const apiService = new ApiService();