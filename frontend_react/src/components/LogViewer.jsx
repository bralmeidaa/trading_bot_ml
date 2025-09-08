import React, { useState, useEffect } from 'react';
import { FileText, Download, Filter, RefreshCw, Search, Calendar, AlertCircle, Info, AlertTriangle, XCircle, CheckCircle } from 'lucide-react';
import { apiService } from '../services/api';
import { showNotification } from '../utils/notifications';

const LogViewer = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [statistics, setStatistics] = useState(null);
  const [categories, setCategories] = useState({ levels: [], categories: [] });
  
  // Filter state
  const [filters, setFilters] = useState({
    start_date: '',
    end_date: '',
    level: '',
    category: '',
    bot_id: '',
    symbol: '',
    limit: 1000
  });

  // UI state
  const [showFilters, setShowFilters] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedExportFormat, setSelectedExportFormat] = useState('json');

  useEffect(() => {
    loadLogs();
    loadStatistics();
    loadCategories();
    
    // Set default date range (last 7 days)
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 7);
    
    setFilters(prev => ({
      ...prev,
      start_date: startDate.toISOString().split('T')[0],
      end_date: endDate.toISOString().split('T')[0]
    }));
  }, []);

  const loadLogs = async (customFilters = null) => {
    try {
      setLoading(true);
      const queryFilters = customFilters || filters;
      
      // Build query parameters
      const params = new URLSearchParams();
      Object.entries(queryFilters).forEach(([key, value]) => {
        if (value && value !== '') {
          if (key.includes('date') && value) {
            // Convert date to ISO format
            params.append(key, new Date(value).toISOString());
          } else {
            params.append(key, value);
          }
        }
      });

      const response = await apiService.get(`/logs/enhanced?${params.toString()}`);
      if (response.success) {
        setLogs(response.data.logs || []);
      } else {
        throw new Error(response.error);
      }
    } catch (error) {
      showNotification('Failed to load logs', 'error');
      console.error('Error loading logs:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadStatistics = async () => {
    try {
      const response = await apiService.get('/logs/statistics');
      if (response.success) {
        setStatistics(response.data);
      }
    } catch (error) {
      console.error('Error loading log statistics:', error);
    }
  };

  const loadCategories = async () => {
    try {
      const response = await apiService.get('/logs/categories');
      if (response.success) {
        setCategories(response.data);
      }
    } catch (error) {
      console.error('Error loading log categories:', error);
    }
  };

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const applyFilters = () => {
    loadLogs();
  };

  const clearFilters = () => {
    setFilters({
      start_date: '',
      end_date: '',
      level: '',
      category: '',
      bot_id: '',
      symbol: '',
      limit: 1000
    });
    setSearchTerm('');
  };

  const exportLogs = async () => {
    try {
      setExporting(true);
      
      const params = new URLSearchParams();
      if (filters.start_date) params.append('start_date', new Date(filters.start_date).toISOString());
      if (filters.end_date) params.append('end_date', new Date(filters.end_date).toISOString());
      params.append('format', selectedExportFormat);

      const response = await fetch(`/api/logs/export?${params.toString()}`, {
        method: 'POST'
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `trading_logs_${new Date().toISOString().split('T')[0]}.${selectedExportFormat}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showNotification('Logs exported successfully', 'success');
      } else {
        throw new Error('Export failed');
      }
    } catch (error) {
      showNotification('Failed to export logs', 'error');
      console.error('Error exporting logs:', error);
    } finally {
      setExporting(false);
    }
  };

  // Filter logs by search term
  const filteredLogs = (Array.isArray(logs) ? logs : []).filter(log => {
    if (!searchTerm) return true;
    const searchLower = searchTerm.toLowerCase();
    return (
      log.message?.toLowerCase().includes(searchLower) ||
      (log.bot_id && log.bot_id.toLowerCase().includes(searchLower)) ||
      (log.symbol && log.symbol.toLowerCase().includes(searchLower)) ||
      (log.trade_id && log.trade_id.toLowerCase().includes(searchLower))
    );
  });

  const getLevelIcon = (level) => {
    switch (level) {
      case 'ERROR':
      case 'CRITICAL':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'WARNING':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'INFO':
        return <Info className="h-4 w-4 text-blue-500" />;
      case 'DEBUG':
        return <CheckCircle className="h-4 w-4 text-gray-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getLevelColor = (level) => {
    switch (level) {
      case 'ERROR':
      case 'CRITICAL':
        return 'text-red-600 bg-red-50';
      case 'WARNING':
        return 'text-yellow-600 bg-yellow-50';
      case 'INFO':
        return 'text-blue-600 bg-blue-50';
      case 'DEBUG':
        return 'text-gray-600 bg-gray-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const getCategoryColor = (category) => {
    switch (category) {
      case 'TRADING':
        return 'text-green-600 bg-green-50';
      case 'SIGNAL':
        return 'text-purple-600 bg-purple-50';
      case 'RISK':
        return 'text-red-600 bg-red-50';
      case 'CONFIG':
        return 'text-blue-600 bg-blue-50';
      case 'API':
        return 'text-indigo-600 bg-indigo-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <FileText className="h-6 w-6 text-blue-600 mr-3" />
            <h2 className="text-xl font-semibold text-gray-900">System Logs</h2>
            {statistics && (
              <span className="ml-4 px-2 py-1 bg-blue-100 text-blue-800 text-sm rounded-full">
                {statistics.total_logs_in_buffer} logs in buffer
              </span>
            )}
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`px-3 py-2 text-sm rounded-md flex items-center ${
                showFilters ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700'
              } hover:bg-blue-200`}
            >
              <Filter className="h-4 w-4 mr-1" />
              Filters
            </button>
            <button
              onClick={loadLogs}
              disabled={loading}
              className="px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 disabled:opacity-50 flex items-center"
            >
              <RefreshCw className={`h-4 w-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Statistics Bar */}
      {statistics && (
        <div className="border-b border-gray-200 px-6 py-3 bg-gray-50">
          <div className="flex items-center justify-between text-sm">
            <div className="flex space-x-6">
              <span>Disk Usage: {statistics.disk_usage?.total_mb || 0} MB</span>
              <span>Recent Activity: {Object.values(statistics.recent_activity || {}).reduce((a, b) => a + b, 0)} logs/hour</span>
            </div>
            <div className="flex space-x-4">
              {Object.entries(statistics.log_levels || {}).map(([level, count]) => (
                <span key={level} className={`px-2 py-1 rounded text-xs ${getLevelColor(level)}`}>
                  {level}: {count}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Filters Panel */}
      {showFilters && (
        <div className="border-b border-gray-200 px-6 py-4 bg-gray-50">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Start Date</label>
              <input
                type="date"
                value={filters.start_date}
                onChange={(e) => handleFilterChange('start_date', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">End Date</label>
              <input
                type="date"
                value={filters.end_date}
                onChange={(e) => handleFilterChange('end_date', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Level</label>
              <select
                value={filters.level}
                onChange={(e) => handleFilterChange('level', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Levels</option>
                {categories.levels.map(level => (
                  <option key={level} value={level}>{level}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
              <select
                value={filters.category}
                onChange={(e) => handleFilterChange('category', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Categories</option>
                {categories.categories.map(category => (
                  <option key={category} value={category}>{category}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Bot ID</label>
              <input
                type="text"
                value={filters.bot_id}
                onChange={(e) => handleFilterChange('bot_id', e.target.value)}
                placeholder="e.g., LINK/USDT_5m"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Symbol</label>
              <input
                type="text"
                value={filters.symbol}
                onChange={(e) => handleFilterChange('symbol', e.target.value)}
                placeholder="e.g., BTC/USDT"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Limit</label>
              <select
                value={filters.limit}
                onChange={(e) => handleFilterChange('limit', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value={100}>100</option>
                <option value={500}>500</option>
                <option value={1000}>1000</option>
                <option value={5000}>5000</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Export Format</label>
              <select
                value={selectedExportFormat}
                onChange={(e) => setSelectedExportFormat(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="json">JSON</option>
                <option value="csv">CSV</option>
                <option value="txt">Text</option>
              </select>
            </div>
          </div>
          <div className="flex justify-between">
            <div className="flex space-x-2">
              <button
                onClick={applyFilters}
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                Apply Filters
              </button>
              <button
                onClick={clearFilters}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400"
              >
                Clear
              </button>
            </div>
            <button
              onClick={exportLogs}
              disabled={exporting}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 flex items-center"
            >
              <Download className="h-4 w-4 mr-2" />
              {exporting ? 'Exporting...' : 'Export Logs'}
            </button>
          </div>
        </div>
      )}

      {/* Search Bar */}
      <div className="px-6 py-3 border-b border-gray-200">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search logs by message, bot ID, symbol, or trade ID..."
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Logs Content */}
      <div className="p-6">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        ) : filteredLogs.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <FileText className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p>No logs found matching your criteria</p>
          </div>
        ) : (
          <div className="space-y-2">
            {filteredLogs.map((log, index) => (
              <LogEntry key={index} log={log} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Individual Log Entry Component
const LogEntry = ({ log }) => {
  const [expanded, setExpanded] = useState(false);

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getLevelColor = (level) => {
    switch (level) {
      case 'ERROR':
      case 'CRITICAL':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'WARNING':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'INFO':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'DEBUG':
        return 'text-gray-600 bg-gray-50 border-gray-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getCategoryColor = (category) => {
    switch (category) {
      case 'TRADING':
        return 'text-green-600 bg-green-50';
      case 'SIGNAL':
        return 'text-purple-600 bg-purple-50';
      case 'RISK':
        return 'text-red-600 bg-red-50';
      case 'CONFIG':
        return 'text-blue-600 bg-blue-50';
      case 'API':
        return 'text-indigo-600 bg-indigo-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className={`border rounded-lg p-3 ${getLevelColor(log.level)}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-1">
            <span className="text-xs font-mono text-gray-500">
              {formatTimestamp(log.timestamp)}
            </span>
            <span className={`px-2 py-1 rounded text-xs font-medium ${getLevelColor(log.level)}`}>
              {log.level}
            </span>
            <span className={`px-2 py-1 rounded text-xs font-medium ${getCategoryColor(log.category)}`}>
              {log.category}
            </span>
            {log.bot_id && (
              <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs">
                {log.bot_id}
              </span>
            )}
            {log.symbol && (
              <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs">
                {log.symbol}
              </span>
            )}
            {log.trade_id && (
              <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs">
                Trade: {log.trade_id}
              </span>
            )}
          </div>
          <p className="text-sm text-gray-900 mb-2">{log.message}</p>
          
          {log.data && (
            <div className="mt-2">
              <button
                onClick={() => setExpanded(!expanded)}
                className="text-xs text-blue-600 hover:text-blue-800"
              >
                {expanded ? 'Hide' : 'Show'} Details
              </button>
              {expanded && (
                <pre className="mt-2 p-2 bg-gray-100 rounded text-xs overflow-x-auto">
                  {JSON.stringify(log.data, null, 2)}
                </pre>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LogViewer;