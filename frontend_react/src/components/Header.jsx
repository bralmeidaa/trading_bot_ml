import React, { useState } from 'react';
import { Play, Square, AlertTriangle, Bot, Activity } from 'lucide-react';
import { useSystemStatus } from '../hooks/useApi';
import { apiService } from '../services/api';
import { notificationService } from '../utils/notifications';
import { formatDuration, getStatusColor } from '../utils/formatters';

export default function Header() {
  const { data: status, loading, refetch } = useSystemStatus();
  const [actionLoading, setActionLoading] = useState(null);

  const handleSystemAction = async (action, actionName) => {
    setActionLoading(action);
    
    try {
      let result;
      switch (action) {
        case 'start':
          result = await apiService.startSystem();
          break;
        case 'stop':
          result = await apiService.stopSystem();
          break;
        case 'emergency':
          if (window.confirm('Are you sure you want to perform an emergency stop? This will immediately close all positions.')) {
            result = await apiService.emergencyStop();
          } else {
            setActionLoading(null);
            return;
          }
          break;
        default:
          setActionLoading(null);
          return;
      }

      if (result.success) {
        notificationService.success(`System ${actionName} successful`);
        // Wait a moment for the system to update its state
        setTimeout(async () => {
          await refetch();
        }, 1000);
      } else {
        notificationService.error(`Failed to ${actionName} system: ${result.error}`);
      }
    } catch (error) {
      notificationService.error(`Error during ${actionName}: ${error.message}`);
    } finally {
      setActionLoading(null);
    }
  };

  if (loading) {
    return (
      <header className="bg-gradient-to-r from-primary-600 to-primary-700 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Bot className="h-8 w-8 mr-3" />
              <h1 className="text-2xl font-bold">Trading Bot ML</h1>
            </div>
            <div className="animate-pulse">
              <div className="h-6 bg-white bg-opacity-20 rounded w-24"></div>
            </div>
          </div>
        </div>
      </header>
    );
  }

  const isRunning = status?.running;
  const statusColor = isRunning ? 'text-green-400' : 'text-gray-400';

  return (
    <header className="bg-gradient-to-r from-primary-600 to-primary-700 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center">
            <Bot className="h-8 w-8 mr-3" />
            <h1 className="text-2xl font-bold">Trading Bot ML</h1>
          </div>

          {/* System Status and Info */}
          <div className="flex items-center space-x-6">
            {/* System Status */}
            <div className="flex items-center space-x-2">
              <Activity className={`h-5 w-5 ${statusColor}`} />
              <span className="font-medium">
                {isRunning ? 'Running' : 'Stopped'}
              </span>
            </div>

            {/* Key Info */}
            <div className="hidden md:flex items-center space-x-6 text-sm">
              <div>
                <span className="text-primary-200">Uptime:</span>
                <span className="ml-1 font-medium">
                  {status?.uptime || '0:00:00'}
                </span>
              </div>
              <div>
                <span className="text-primary-200">Capital:</span>
                <span className="ml-1 font-medium">
                  ${status?.total_capital?.toLocaleString() || '0'}
                </span>
              </div>
              <div>
                <span className="text-primary-200">Mode:</span>
                <span className="ml-1 font-medium">
                  {status?.paper_trading ? 'Paper' : 'Live'}
                </span>
              </div>
            </div>

            {/* Control Buttons */}
            <div className="flex items-center space-x-2">
              {!isRunning ? (
                <button
                  onClick={() => handleSystemAction('start', 'start')}
                  disabled={actionLoading !== null}
                  className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                >
                  {actionLoading === 'start' ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                  <span>Start</span>
                </button>
              ) : (
                <button
                  onClick={() => handleSystemAction('stop', 'stop')}
                  disabled={actionLoading !== null}
                  className="bg-gray-600 hover:bg-gray-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                >
                  {actionLoading === 'stop' ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  ) : (
                    <Square className="h-4 w-4" />
                  )}
                  <span>Stop</span>
                </button>
              )}

              <button
                onClick={() => handleSystemAction('emergency', 'emergency stop')}
                disabled={actionLoading !== null || !isRunning}
                className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              >
                {actionLoading === 'emergency' ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                ) : (
                  <AlertTriangle className="h-4 w-4" />
                )}
                <span className="hidden sm:inline">Emergency</span>
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Info */}
        <div className="md:hidden mt-3 flex items-center justify-between text-sm">
          <div className="flex items-center space-x-4">
            <span>Uptime: {status?.uptime || '0:00:00'}</span>
            <span>Capital: ${status?.total_capital?.toLocaleString() || '0'}</span>
          </div>
          <span className="text-primary-200">
            {status?.paper_trading ? 'Paper Trading' : 'Live Trading'}
          </span>
        </div>
      </div>
    </header>
  );
}