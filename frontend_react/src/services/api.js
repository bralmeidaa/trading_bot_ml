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
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return { success: true, data };
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      return { success: false, error: error.message };
    }
  }

  // Convenience method for GET requests
  async get(endpoint) {
    return this.request(endpoint, { method: 'GET' });
  }

  // System Status
  async getSystemStatus() {
    return this.request('/status');
  }

  async startSystem() {
    return this.request('/start', { method: 'POST' });
  }

  async stopSystem() {
    return this.request('/stop', { method: 'POST' });
  }

  async emergencyStop() {
    return this.request('/emergency-stop', { method: 'POST' });
  }

  // Performance Metrics
  async getMetrics() {
    return this.request('/metrics');
  }

  async getEquity() {
    return this.request('/equity');
  }

  // Bot Management
  async getBots() {
    return this.request('/bots');
  }

  async updateBotConfig(botId, config) {
    return this.request(`/bots/${botId}/config`, {
      method: 'PUT',
      body: JSON.stringify(config),
    });
  }

  async toggleBot(botId, enabled) {
    return this.request(`/bots/${botId}/toggle`, {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    });
  }

  // Trades and Logs
  async getRecentTrades() {
    return this.request('/trades/recent');
  }

  async getLogs() {
    return this.request('/logs');
  }

  // Configuration
  async getConfig() {
    return this.request('/config/full');
  }

  async updateConfig(config) {
    return this.request('/config', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  // Backtest
  async runBacktest() {
    return this.request('/backtest', {
      method: 'POST',
    });
  }

  // Health Check
  async healthCheck() {
    return this.request('/health');
  }
}

export const apiService = new ApiService();