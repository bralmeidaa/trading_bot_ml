import React, { useState, useEffect } from 'react';
import { Settings, Save, RotateCcw, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';
import { useConfig } from '../hooks/useApi';
import { apiService } from '../services/api';
import { notificationService } from '../utils/notifications';
import BotManagement from './BotManagement';

const FormField = ({ label, name, type = 'text', value, onChange, placeholder, min, max, step, required = false, disabled = false }) => (
  <div className="mb-4">
    <label htmlFor={name} className="block text-sm font-medium text-gray-700 mb-2">
      {label} {required && <span className="text-danger-500">*</span>}
    </label>
    <input
      type={type}
      id={name}
      name={name}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
    />
  </div>
);

const SelectField = ({ label, name, value, onChange, options, required = false, disabled = false }) => (
  <div className="mb-4">
    <label htmlFor={name} className="block text-sm font-medium text-gray-700 mb-2">
      {label} {required && <span className="text-danger-500">*</span>}
    </label>
    <select
      id={name}
      name={name}
      value={value}
      onChange={onChange}
      disabled={disabled}
      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
    >
      {options.map(option => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  </div>
);

const AdvancedBotConfig = ({ botConfig, onChange }) => {
  return (
    <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
      <h4 className="font-medium text-gray-900">{botConfig.symbol} - {botConfig.timeframe}</h4>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FormField
          label="Capital Allocation (%)"
          name="capital_allocation"
          type="number"
          value={botConfig.capital_allocation * 100}
          onChange={(e) => onChange(botConfig.id, 'capital_allocation', parseFloat(e.target.value) / 100)}
          placeholder="70"
          min="0"
          max="100"
          step="1"
        />
        
        <FormField
          label="Max Risk Per Trade (%)"
          name="max_risk_per_trade"
          type="number"
          value={botConfig.max_risk_per_trade * 100}
          onChange={(e) => onChange(botConfig.id, 'max_risk_per_trade', parseFloat(e.target.value) / 100)}
          placeholder="2.5"
          min="0.1"
          max="10"
          step="0.1"
        />
        
        <FormField
          label="Confidence Threshold"
          name="confidence_threshold"
          type="number"
          value={botConfig.confidence_threshold}
          onChange={(e) => onChange(botConfig.id, 'confidence_threshold', parseFloat(e.target.value))}
          placeholder="0.65"
          min="0.1"
          max="1"
          step="0.01"
        />
        
        <FormField
          label="Stop Loss (%)"
          name="stop_loss_pct"
          type="number"
          value={botConfig.stop_loss_pct * 100}
          onChange={(e) => onChange(botConfig.id, 'stop_loss_pct', parseFloat(e.target.value) / 100)}
          placeholder="1.8"
          min="0.1"
          max="10"
          step="0.1"
        />
        
        <FormField
          label="Take Profit (%)"
          name="take_profit_pct"
          type="number"
          value={botConfig.take_profit_pct * 100}
          onChange={(e) => onChange(botConfig.id, 'take_profit_pct', parseFloat(e.target.value) / 100)}
          placeholder="3.5"
          min="0.1"
          max="20"
          step="0.1"
        />
        
        <div className="flex items-center">
          <input
            type="checkbox"
            id={`enabled_${botConfig.id}`}
            checked={botConfig.enabled}
            onChange={(e) => onChange(botConfig.id, 'enabled', e.target.checked)}
            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
          />
          <label htmlFor={`enabled_${botConfig.id}`} className="ml-2 block text-sm text-gray-900">
            Bot Enabled
          </label>
        </div>
      </div>
    </div>
  );
};

export default function ConfigurationPanel() {
  const { data: config, loading, error, refetch } = useConfig();
  const [formData, setFormData] = useState({
    trading_mode: 'paper',
    total_capital: 1200,
    daily_loss_limit: 4,
    daily_profit_target: 2.5,
    max_concurrent_trades: 2,
    emergency_stop_drawdown: 8,
  });
  const [botConfigs, setBotConfigs] = useState([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (config && config.global_config) {
      const globalConfig = config.global_config;
      setFormData({
        trading_mode: globalConfig.paper_trading ? 'paper' : 'live',
        total_capital: globalConfig.total_capital || 1200,
        daily_loss_limit: (globalConfig.daily_loss_limit || 0.04) * 100,
        daily_profit_target: (globalConfig.daily_profit_target || 0.025) * 100,
        max_concurrent_trades: globalConfig.max_concurrent_trades || 2,
        emergency_stop_drawdown: (globalConfig.emergency_stop_drawdown || 0.08) * 100,
      });
      
      if (config.bot_configs) {
        setBotConfigs(config.bot_configs.map((bot, index) => ({
          ...bot,
          id: bot.id || `${bot.symbol}_${bot.timeframe}_${index}`
        })));
      }
    }
  }, [config]);

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) || 0 : value
    }));
  };

  const handleBotConfigChange = (botId, field, value) => {
    setBotConfigs(prev => prev.map(bot => 
      bot.id === botId ? { ...bot, [field]: value } : bot
    ));
  };

  const handleSave = async () => {
    setSaving(true);
    
    try {
      const configToSave = {
        trading_mode: formData.trading_mode,
        total_capital: formData.total_capital,
        daily_loss_limit: formData.daily_loss_limit / 100,
        daily_profit_target: formData.daily_profit_target / 100,
        max_concurrent_trades: formData.max_concurrent_trades,
        emergency_stop_drawdown: formData.emergency_stop_drawdown / 100,
        bot_configs: botConfigs,
      };

      const result = await apiService.updateConfig(configToSave);
      
      if (result.success) {
        notificationService.success('Configuration saved successfully');
        refetch();
      } else {
        notificationService.error(`Failed to save configuration: ${result.error}`);
      }
    } catch (error) {
      notificationService.error(`Error saving configuration: ${error.message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    if (window.confirm('Are you sure you want to reset to default configuration?')) {
      setFormData({
        trading_mode: 'paper',
        total_capital: 1200,
        daily_loss_limit: 4,
        daily_profit_target: 2.5,
        max_concurrent_trades: 2,
        emergency_stop_drawdown: 8,
      });
      notificationService.info('Configuration reset to defaults');
    }
  };

  if (error) {
    return (
      <div className="card mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">System Configuration</h2>
        </div>
        <div className="text-center py-8">
          <AlertCircle className="h-12 w-12 text-danger-500 mx-auto mb-4" />
          <p className="text-danger-600">Failed to load configuration: {error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card mb-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">System Configuration</h2>
          <p className="text-sm text-gray-600 mt-1">
            Adjust global settings and bot parameters
          </p>
        </div>
        
        <div className="flex space-x-2">
          <button
            onClick={handleReset}
            className="btn-secondary flex items-center space-x-2"
          >
            <RotateCcw className="h-4 w-4" />
            <span>Reset</span>
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="btn-primary flex items-center space-x-2"
          >
            {saving ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
            ) : (
              <Save className="h-4 w-4" />
            )}
            <span>Save Configuration</span>
          </button>
        </div>
      </div>

      {loading ? (
        <div className="animate-pulse space-y-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-16 bg-gray-200 rounded"></div>
          ))}
        </div>
      ) : (
        <div className="space-y-6">
          {/* Global Configuration */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">Global Settings</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <SelectField
                label="Trading Mode"
                name="trading_mode"
                value={formData.trading_mode}
                onChange={handleInputChange}
                options={[
                  { value: 'paper', label: 'Paper Trading (Recommended)' },
                  { value: 'live', label: 'Live Trading' }
                ]}
                required
              />
              
              <FormField
                label="Total Capital (USD)"
                name="total_capital"
                type="number"
                value={formData.total_capital}
                onChange={handleInputChange}
                placeholder="1200 (Recommended minimum)"
                min="100"
                step="100"
                required
              />
              
              <FormField
                label="Daily Loss Limit (%)"
                name="daily_loss_limit"
                type="number"
                value={formData.daily_loss_limit}
                onChange={handleInputChange}
                placeholder="4 (Recommended)"
                min="1"
                max="20"
                step="0.5"
                required
              />
              
              <FormField
                label="Daily Profit Target (%)"
                name="daily_profit_target"
                type="number"
                value={formData.daily_profit_target}
                onChange={handleInputChange}
                placeholder="2.5 (Recommended)"
                min="0.5"
                max="10"
                step="0.5"
                required
              />
              
              <FormField
                label="Max Concurrent Trades"
                name="max_concurrent_trades"
                type="number"
                value={formData.max_concurrent_trades}
                onChange={handleInputChange}
                placeholder="2 (Recommended for minimum capital)"
                min="1"
                max="10"
                step="1"
                required
              />
              
              <FormField
                label="Emergency Stop Drawdown (%)"
                name="emergency_stop_drawdown"
                type="number"
                value={formData.emergency_stop_drawdown}
                onChange={handleInputChange}
                placeholder="8 (Recommended)"
                min="5"
                max="25"
                step="1"
                required
              />
            </div>
          </div>

          {/* Bot Management */}
          <div>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center space-x-2 text-lg font-medium text-gray-900 hover:text-primary-600 transition-colors duration-200"
            >
              <Settings className="h-5 w-5" />
              <span>Bot Configuration & Management</span>
              {showAdvanced ? (
                <ChevronUp className="h-5 w-5" />
              ) : (
                <ChevronDown className="h-5 w-5" />
              )}
            </button>
            
            {showAdvanced && (
              <div className="mt-4">
                <p className="text-sm text-gray-600 mb-6">
                  Create and manage up to 5 trading bots with custom symbols, timeframes, and parameters.
                  Each bot can be configured independently for optimal performance.
                </p>
                
                <BotManagement 
                  bots={botConfigs} 
                  onBotsChange={refetch}
                />
              </div>
            )}
          </div>

          {/* Configuration Tips */}
          <div className="bg-primary-50 border border-primary-200 rounded-lg p-4">
            <h4 className="font-medium text-primary-900 mb-2">💡 Configuration Tips</h4>
            <ul className="text-sm text-primary-800 space-y-1">
              <li>• Start with Paper Trading to test your configuration safely</li>
              <li>• Minimum capital of $1,200 is recommended for optimal performance</li>
              <li>• Keep daily loss limit at 4% or lower to preserve capital</li>
              <li>• You can create up to 5 bots with different symbols and timeframes</li>
              <li>• LINK/USDT 5m timeframe has shown the best backtesting results</li>
              <li>• Diversify across different timeframes (1m, 5m, 15m, 1h) for better risk management</li>
              <li>• Lower confidence thresholds generate more trades but may reduce accuracy</li>
              <li>• Ensure total capital allocation across all bots doesn't exceed 100%</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}