import React, { useState, useEffect } from 'react';
import { Settings, Save, RotateCcw, Plus, Trash2, Download, Upload, AlertTriangle } from 'lucide-react';
import { apiService } from '../services/api';
import { showNotification } from '../utils/notifications';

const EnhancedConfigPanel = () => {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState('global');
  const [backups, setBackups] = useState([]);
  const [showAddBot, setShowAddBot] = useState(false);

  // New bot form state
  const [newBot, setNewBot] = useState({
    symbol: 'BTC/USDT',
    timeframe: '5m',
    capital_allocation: 0.25,
    max_risk_per_trade: 0.025,
    confidence_threshold: 0.60,
    stop_loss_pct: 0.020,
    take_profit_pct: 0.035,
    enabled: true
  });

  useEffect(() => {
    loadConfiguration();
    loadBackups();
  }, []);

  const loadConfiguration = async () => {
    try {
      setLoading(true);
      const response = await apiService.get('/api/config/full');
      setConfig(response);
    } catch (error) {
      showNotification('Failed to load configuration', 'error');
      console.error('Error loading configuration:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadBackups = async () => {
    try {
      const response = await apiService.get('/api/config/backups');
      setBackups(response.backups || []);
    } catch (error) {
      console.error('Error loading backups:', error);
    }
  };

  const updateGlobalConfig = async (updates) => {
    try {
      setSaving(true);
      await apiService.put('/api/config/global', updates);
      showNotification('Global configuration updated successfully', 'success');
      await loadConfiguration();
    } catch (error) {
      showNotification('Failed to update global configuration', 'error');
      console.error('Error updating global config:', error);
    } finally {
      setSaving(false);
    }
  };

  const updateBotConfig = async (botId, updates) => {
    try {
      setSaving(true);
      await apiService.put(`/api/config/bot/${botId}`, updates);
      showNotification(`Bot ${botId} configuration updated successfully`, 'success');
      await loadConfiguration();
    } catch (error) {
      showNotification(`Failed to update bot ${botId} configuration`, 'error');
      console.error('Error updating bot config:', error);
    } finally {
      setSaving(false);
    }
  };

  const addBot = async () => {
    try {
      setSaving(true);
      await apiService.post('/api/config/bot', newBot);
      showNotification('Bot added successfully', 'success');
      setShowAddBot(false);
      setNewBot({
        symbol: 'BTC/USDT',
        timeframe: '5m',
        capital_allocation: 0.25,
        max_risk_per_trade: 0.025,
        confidence_threshold: 0.60,
        stop_loss_pct: 0.020,
        take_profit_pct: 0.035,
        enabled: true
      });
      await loadConfiguration();
    } catch (error) {
      showNotification('Failed to add bot', 'error');
      console.error('Error adding bot:', error);
    } finally {
      setSaving(false);
    }
  };

  const removeBot = async (botId) => {
    if (!confirm(`Are you sure you want to remove bot ${botId}?`)) return;

    try {
      setSaving(true);
      await apiService.delete(`/api/config/bot/${botId}`);
      showNotification(`Bot ${botId} removed successfully`, 'success');
      await loadConfiguration();
    } catch (error) {
      showNotification(`Failed to remove bot ${botId}`, 'error');
      console.error('Error removing bot:', error);
    } finally {
      setSaving(false);
    }
  };

  const restoreBackup = async (backupFilename) => {
    if (!confirm(`Are you sure you want to restore configuration from ${backupFilename}?`)) return;

    try {
      setSaving(true);
      await apiService.post(`/api/config/restore/${backupFilename}`);
      showNotification('Configuration restored successfully', 'success');
      await loadConfiguration();
    } catch (error) {
      showNotification('Failed to restore configuration', 'error');
      console.error('Error restoring backup:', error);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="text-center text-red-600">
          <AlertTriangle className="h-12 w-12 mx-auto mb-4" />
          <p>Failed to load configuration</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Settings className="h-6 w-6 text-blue-600 mr-3" />
            <h2 className="text-xl font-semibold text-gray-900">Enhanced Configuration</h2>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={loadConfiguration}
              disabled={saving}
              className="px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 disabled:opacity-50"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8 px-6">
          {['global', 'bots', 'backups'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)} Configuration
            </button>
          ))}
        </nav>
      </div>

      {/* Content */}
      <div className="p-6">
        {activeTab === 'global' && (
          <GlobalConfigTab 
            config={config.global_config} 
            onUpdate={updateGlobalConfig}
            saving={saving}
          />
        )}
        
        {activeTab === 'bots' && (
          <BotsConfigTab 
            bots={config.bot_configs}
            onUpdate={updateBotConfig}
            onRemove={removeBot}
            onAdd={addBot}
            newBot={newBot}
            setNewBot={setNewBot}
            showAddBot={showAddBot}
            setShowAddBot={setShowAddBot}
            saving={saving}
          />
        )}
        
        {activeTab === 'backups' && (
          <BackupsTab 
            backups={backups}
            onRestore={restoreBackup}
            onRefresh={loadBackups}
            saving={saving}
          />
        )}
      </div>
    </div>
  );
};

// Global Configuration Tab
const GlobalConfigTab = ({ config, onUpdate, saving }) => {
  const [formData, setFormData] = useState(config);

  useEffect(() => {
    setFormData(config);
  }, [config]);

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Only send changed values
    const changes = {};
    Object.keys(formData).forEach(key => {
      if (formData[key] !== config[key]) {
        changes[key] = formData[key];
      }
    });

    if (Object.keys(changes).length > 0) {
      onUpdate(changes);
    }
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Total Capital ($)
          </label>
          <input
            type="number"
            step="0.01"
            min="100"
            max="100000"
            value={formData.total_capital}
            onChange={(e) => handleChange('total_capital', parseFloat(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Max Concurrent Trades
          </label>
          <input
            type="number"
            min="1"
            max="20"
            value={formData.max_concurrent_trades}
            onChange={(e) => handleChange('max_concurrent_trades', parseInt(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Daily Loss Limit (%)
          </label>
          <input
            type="number"
            step="0.1"
            min="1"
            max="20"
            value={(formData.daily_loss_limit * 100).toFixed(1)}
            onChange={(e) => handleChange('daily_loss_limit', parseFloat(e.target.value) / 100)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Daily Profit Target (%)
          </label>
          <input
            type="number"
            step="0.1"
            min="0.5"
            max="10"
            value={(formData.daily_profit_target * 100).toFixed(1)}
            onChange={(e) => handleChange('daily_profit_target', parseFloat(e.target.value) / 100)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Emergency Stop Drawdown (%)
          </label>
          <input
            type="number"
            step="0.1"
            min="5"
            max="30"
            value={(formData.emergency_stop_drawdown * 100).toFixed(1)}
            onChange={(e) => handleChange('emergency_stop_drawdown', parseFloat(e.target.value) / 100)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Trading Mode
          </label>
          <select
            value={formData.paper_trading ? 'paper' : 'live'}
            onChange={(e) => handleChange('paper_trading', e.target.value === 'paper')}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="paper">Paper Trading</option>
            <option value="live">Live Trading</option>
          </select>
        </div>
      </div>

      <div className="flex justify-end">
        <button
          type="submit"
          disabled={saving}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center"
        >
          <Save className="h-4 w-4 mr-2" />
          {saving ? 'Saving...' : 'Save Changes'}
        </button>
      </div>
    </form>
  );
};

// Bots Configuration Tab
const BotsConfigTab = ({ bots, onUpdate, onRemove, onAdd, newBot, setNewBot, showAddBot, setShowAddBot, saving }) => {
  return (
    <div className="space-y-6">
      {/* Add Bot Button */}
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900">Trading Bots</h3>
        <button
          onClick={() => setShowAddBot(!showAddBot)}
          className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center"
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Bot
        </button>
      </div>

      {/* Add Bot Form */}
      {showAddBot && (
        <div className="bg-gray-50 p-4 rounded-lg border">
          <h4 className="text-md font-medium text-gray-900 mb-4">Add New Bot</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Symbol</label>
              <select
                value={newBot.symbol}
                onChange={(e) => setNewBot(prev => ({ ...prev, symbol: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="BTC/USDT">BTC/USDT</option>
                <option value="ETH/USDT">ETH/USDT</option>
                <option value="LINK/USDT">LINK/USDT</option>
                <option value="ADA/USDT">ADA/USDT</option>
                <option value="SOL/USDT">SOL/USDT</option>
                <option value="MATIC/USDT">MATIC/USDT</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Timeframe</label>
              <select
                value={newBot.timeframe}
                onChange={(e) => setNewBot(prev => ({ ...prev, timeframe: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="1m">1 minute</option>
                <option value="3m">3 minutes</option>
                <option value="5m">5 minutes</option>
                <option value="15m">15 minutes</option>
                <option value="30m">30 minutes</option>
                <option value="1h">1 hour</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Capital Allocation (%)</label>
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="1"
                value={(newBot.capital_allocation * 100).toFixed(0)}
                onChange={(e) => setNewBot(prev => ({ ...prev, capital_allocation: parseFloat(e.target.value) / 100 }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          <div className="flex justify-end mt-4 space-x-2">
            <button
              onClick={() => setShowAddBot(false)}
              className="px-4 py-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400"
            >
              Cancel
            </button>
            <button
              onClick={onAdd}
              disabled={saving}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
            >
              {saving ? 'Adding...' : 'Add Bot'}
            </button>
          </div>
        </div>
      )}

      {/* Existing Bots */}
      <div className="space-y-4">
        {bots.map((bot, index) => (
          <BotConfigCard
            key={`${bot.symbol}_${bot.timeframe}`}
            bot={bot}
            onUpdate={onUpdate}
            onRemove={onRemove}
            saving={saving}
          />
        ))}
      </div>
    </div>
  );
};

// Individual Bot Configuration Card
const BotConfigCard = ({ bot, onUpdate, onRemove, saving }) => {
  const [expanded, setExpanded] = useState(false);
  const [formData, setFormData] = useState(bot);

  const botId = `${bot.symbol}_${bot.timeframe}`;

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Only send changed values
    const changes = {};
    Object.keys(formData).forEach(key => {
      if (formData[key] !== bot[key]) {
        changes[key] = formData[key];
      }
    });

    if (Object.keys(changes).length > 0) {
      onUpdate(botId, changes);
    }
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="border border-gray-200 rounded-lg">
      <div className="p-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className={`w-3 h-3 rounded-full ${bot.enabled ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <div>
            <h4 className="font-medium text-gray-900">{bot.symbol} - {bot.timeframe}</h4>
            <p className="text-sm text-gray-500">
              {(bot.capital_allocation * 100).toFixed(0)}% allocation • 
              {(bot.confidence_threshold * 100).toFixed(0)}% confidence
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200"
          >
            {expanded ? 'Collapse' : 'Configure'}
          </button>
          <button
            onClick={() => onRemove(botId)}
            disabled={saving}
            className="px-3 py-1 text-sm bg-red-100 text-red-700 rounded-md hover:bg-red-200 disabled:opacity-50"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-gray-200 p-4">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Capital Allocation (%)
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0.01"
                  max="1"
                  value={(formData.capital_allocation * 100).toFixed(0)}
                  onChange={(e) => handleChange('capital_allocation', parseFloat(e.target.value) / 100)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Risk per Trade (%)
                </label>
                <input
                  type="number"
                  step="0.001"
                  min="0.005"
                  max="0.1"
                  value={(formData.max_risk_per_trade * 100).toFixed(1)}
                  onChange={(e) => handleChange('max_risk_per_trade', parseFloat(e.target.value) / 100)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Confidence Threshold (%)
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0.4"
                  max="0.9"
                  value={(formData.confidence_threshold * 100).toFixed(0)}
                  onChange={(e) => handleChange('confidence_threshold', parseFloat(e.target.value) / 100)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Stop Loss (%)
                </label>
                <input
                  type="number"
                  step="0.001"
                  min="0.005"
                  max="0.05"
                  value={(formData.stop_loss_pct * 100).toFixed(1)}
                  onChange={(e) => handleChange('stop_loss_pct', parseFloat(e.target.value) / 100)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Take Profit (%)
                </label>
                <input
                  type="number"
                  step="0.001"
                  min="0.01"
                  max="0.15"
                  value={(formData.take_profit_pct * 100).toFixed(1)}
                  onChange={(e) => handleChange('take_profit_pct', parseFloat(e.target.value) / 100)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Status
                </label>
                <select
                  value={formData.enabled ? 'enabled' : 'disabled'}
                  onChange={(e) => handleChange('enabled', e.target.value === 'enabled')}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="enabled">Enabled</option>
                  <option value="disabled">Disabled</option>
                </select>
              </div>
            </div>

            <div className="flex justify-end">
              <button
                type="submit"
                disabled={saving}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center"
              >
                <Save className="h-4 w-4 mr-2" />
                {saving ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};

// Backups Tab
const BackupsTab = ({ backups, onRestore, onRefresh, saving }) => {
  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900">Configuration Backups</h3>
        <button
          onClick={onRefresh}
          disabled={saving}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center"
        >
          <RotateCcw className="h-4 w-4 mr-2" />
          Refresh
        </button>
      </div>

      {backups.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No backups available
        </div>
      ) : (
        <div className="space-y-2">
          {backups.map((backup, index) => (
            <div key={index} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
              <div>
                <h4 className="font-medium text-gray-900">{backup.filename}</h4>
                <p className="text-sm text-gray-500">
                  Created: {(() => {
                    try {
                      if (!backup.created) return 'N/A';
                      const date = new Date(backup.created);
                      if (isNaN(date.getTime())) return 'Invalid Date';
                      return date.toLocaleString();
                    } catch (e) {
                      console.warn('Error formatting backup timestamp:', backup.created, e);
                      return 'Invalid Date';
                    }
                  })()} • 
                  Size: {(backup.size / 1024).toFixed(1)} KB
                </p>
              </div>
              <button
                onClick={() => onRestore(backup.filename)}
                disabled={saving}
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 flex items-center"
              >
                <Upload className="h-4 w-4 mr-2" />
                Restore
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default EnhancedConfigPanel;