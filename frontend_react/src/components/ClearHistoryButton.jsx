/**
 * Clear History Button Component
 * Implements the "Zerar Histórico de Trades e Logs" functionality
 */

import React, { useState } from 'react';
import { Trash2, AlertTriangle, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import { useClearTradingHistory } from '../services/apiClient';

const ClearHistoryButton = ({ className = "" }) => {
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [confirmText, setConfirmText] = useState('');
  const clearHistoryMutation = useClearTradingHistory();

  const handleClearHistory = async () => {
    if (confirmText !== 'CLEAR ALL DATA') {
      toast.error('Please type "CLEAR ALL DATA" to confirm');
      return;
    }

    try {
      await clearHistoryMutation.mutateAsync();
      setShowConfirmation(false);
      setConfirmText('');
    } catch (error) {
      // Error is handled by the mutation hook
      console.error('Clear history error:', error);
    }
  };

  const handleCancel = () => {
    setShowConfirmation(false);
    setConfirmText('');
  };

  return (
    <>
      {/* Main Button */}
      <button
        onClick={() => setShowConfirmation(true)}
        disabled={clearHistoryMutation.isPending}
        className={`
          flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 
          text-white rounded-lg transition-colors duration-200
          disabled:opacity-50 disabled:cursor-not-allowed
          ${className}
        `}
      >
        {clearHistoryMutation.isPending ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <Trash2 className="w-4 h-4" />
        )}
        <span className="text-sm font-medium">
          {clearHistoryMutation.isPending ? 'Clearing...' : 'Clear History'}
        </span>
      </button>

      {/* Confirmation Modal */}
      <AnimatePresence>
        {showConfirmation && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full p-6"
            >
              {/* Header */}
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Clear Trading History
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    This action cannot be undone
                  </p>
                </div>
              </div>

              {/* Warning Message */}
              <div className="bg-red-50 dark:bg-red-900/10 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
                  <div className="text-sm">
                    <p className="font-medium text-red-800 dark:text-red-200 mb-2">
                      ⚠️ DANGER: This will permanently delete:
                    </p>
                    <ul className="text-red-700 dark:text-red-300 space-y-1 list-disc list-inside">
                      <li>All trading history and closed trades</li>
                      <li>All system logs and activity records</li>
                      <li>All trading signals and performance data</li>
                      <li>Equity curve and daily performance metrics</li>
                    </ul>
                    <p className="font-medium text-red-800 dark:text-red-200 mt-3">
                      Bot configurations and settings will be preserved.
                    </p>
                  </div>
                </div>
              </div>

              {/* Confirmation Input */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Type <span className="font-mono bg-gray-100 dark:bg-gray-700 px-1 rounded">CLEAR ALL DATA</span> to confirm:
                </label>
                <input
                  type="text"
                  value={confirmText}
                  onChange={(e) => setConfirmText(e.target.value)}
                  placeholder="CLEAR ALL DATA"
                  className="
                    w-full px-3 py-2 border border-gray-300 dark:border-gray-600 
                    rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                    focus:ring-2 focus:ring-red-500 focus:border-red-500
                    placeholder-gray-400 dark:placeholder-gray-500
                  "
                  autoComplete="off"
                />
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 justify-end">
                <button
                  onClick={handleCancel}
                  disabled={clearHistoryMutation.isPending}
                  className="
                    px-4 py-2 text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700
                    hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg transition-colors
                    disabled:opacity-50 disabled:cursor-not-allowed
                  "
                >
                  Cancel
                </button>
                <button
                  onClick={handleClearHistory}
                  disabled={clearHistoryMutation.isPending || confirmText !== 'CLEAR ALL DATA'}
                  className="
                    flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 
                    text-white rounded-lg transition-colors
                    disabled:opacity-50 disabled:cursor-not-allowed
                  "
                >
                  {clearHistoryMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Clearing...</span>
                    </>
                  ) : (
                    <>
                      <Trash2 className="w-4 h-4" />
                      <span>Clear All Data</span>
                    </>
                  )}
                </button>
              </div>

              {/* Additional Info */}
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
                  💡 Tip: You can backup your data before clearing by exporting reports
                </p>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
};

// Compact version for smaller spaces
export const ClearHistoryButtonCompact = ({ className = "" }) => {
  const [showConfirmation, setShowConfirmation] = useState(false);
  const clearHistoryMutation = useClearTradingHistory();

  const handleQuickClear = async () => {
    const confirmed = window.confirm(
      '⚠️ WARNING: This will permanently delete ALL trading history, logs, and performance data.\n\nThis action cannot be undone. Are you sure?'
    );

    if (confirmed) {
      try {
        await clearHistoryMutation.mutateAsync();
      } catch (error) {
        console.error('Clear history error:', error);
      }
    }
  };

  return (
    <button
      onClick={handleQuickClear}
      disabled={clearHistoryMutation.isPending}
      title="Clear all trading history and logs"
      className={`
        p-2 text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20
        rounded-lg transition-colors duration-200
        disabled:opacity-50 disabled:cursor-not-allowed
        ${className}
      `}
    >
      {clearHistoryMutation.isPending ? (
        <Loader2 className="w-4 h-4 animate-spin" />
      ) : (
        <Trash2 className="w-4 h-4" />
      )}
    </button>
  );
};

// Admin panel version with additional options
export const AdminClearHistoryPanel = ({ className = "" }) => {
  const [selectedOptions, setSelectedOptions] = useState({
    trades: true,
    logs: true,
    signals: true,
    performance: true,
    equity: true
  });

  const clearHistoryMutation = useClearTradingHistory();

  const handleOptionChange = (option) => {
    setSelectedOptions(prev => ({
      ...prev,
      [option]: !prev[option]
    }));
  };

  const hasAnySelection = Object.values(selectedOptions).some(Boolean);

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-red-100 dark:bg-red-900/20 rounded-lg flex items-center justify-center">
          <Trash2 className="w-5 h-5 text-red-600 dark:text-red-400" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Data Management
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Clear specific types of historical data
          </p>
        </div>
      </div>

      {/* Options */}
      <div className="space-y-3 mb-6">
        {[
          { key: 'trades', label: 'Trading History', desc: 'All closed and cancelled trades' },
          { key: 'logs', label: 'System Logs', desc: 'Application logs and activity records' },
          { key: 'signals', label: 'Trading Signals', desc: 'Generated trading signals and analysis' },
          { key: 'performance', label: 'Performance Metrics', desc: 'Daily performance and statistics' },
          { key: 'equity', label: 'Equity Curve', desc: 'Historical equity and drawdown data' }
        ].map(({ key, label, desc }) => (
          <label key={key} className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={selectedOptions[key]}
              onChange={() => handleOptionChange(key)}
              className="mt-1 w-4 h-4 text-red-600 border-gray-300 rounded focus:ring-red-500"
            />
            <div className="flex-1">
              <div className="font-medium text-gray-900 dark:text-white">{label}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">{desc}</div>
            </div>
          </label>
        ))}
      </div>

      {/* Action */}
      <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {hasAnySelection ? 'Selected data will be permanently deleted' : 'Select data types to clear'}
        </p>
        <ClearHistoryButton 
          className={!hasAnySelection ? 'opacity-50 cursor-not-allowed' : ''}
        />
      </div>
    </div>
  );
};

export default ClearHistoryButton;