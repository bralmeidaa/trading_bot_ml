import React, { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import PerformanceMetrics from './components/PerformanceMetrics';
import EquityChart from './components/EquityChart';
import BotStatusTable from './components/BotStatusTable';
import TradesAndLogs from './components/TradesAndLogs';
import ConfigurationPanel from './components/ConfigurationPanel';
import LogViewer from './components/LogViewer';
import Notifications from './components/Notifications';
import SignalQualityMonitor from './components/SignalQualityMonitor';
import MarketSentimentPanel from './components/MarketSentimentPanel';
import MarketRegimeMonitor from './components/MarketRegimeMonitor';
import TradingStatusIndicator from './components/TradingStatusIndicator';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
    },
  },
});

function App() {
  const [activeView, setActiveView] = useState('dashboard');

  const renderView = () => {
    switch (activeView) {
      case 'dashboard':
        return (
          <>
            {/* Trading Status Indicator */}
            <TradingStatusIndicator className="mb-6" />
            
            <PerformanceMetrics />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <EquityChart />
              <BotStatusTable />
            </div>
            <TradesAndLogs />
          </>
        );
      case 'quality':
        return <SignalQualityMonitor />;
      case 'sentiment':
        return <MarketSentimentPanel />;
      case 'regime':
        return <MarketRegimeMonitor />;
      case 'config':
        return <ConfigurationPanel />;
      case 'logs':
        return <LogViewer />;
      default:
        return null;
    }
  };

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        {/* Header */}
        <Header />

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Navigation Tabs */}
          <div className="mb-6">
            <nav className="flex space-x-8 overflow-x-auto">
              {[
                { id: 'dashboard', label: 'Dashboard', icon: '📊' },
                { id: 'quality', label: 'Signal Quality', icon: '🎯' },
                { id: 'sentiment', label: 'Market Sentiment', icon: '📈' },
                { id: 'regime', label: 'Market Regime', icon: '🔄' },
                { id: 'config', label: 'Configuration', icon: '⚙️' },
                { id: 'logs', label: 'Logs & Export', icon: '📋' }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveView(tab.id)}
                  className={`flex items-center px-4 py-2 rounded-lg font-medium transition-colors duration-200 whitespace-nowrap ${
                    activeView === tab.id
                      ? 'bg-blue-600 text-white shadow-lg'
                      : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
                  }`}
                >
                  <span className="mr-2">{tab.icon}</span>
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          {/* View Content */}
          {renderView()}

          {/* Footer */}
          <footer className="mt-12 py-8 border-t border-gray-200 dark:border-gray-700">
            <div className="text-center text-gray-500 dark:text-gray-400">
              <p className="text-sm">
                Trading Bot ML Dashboard v2.0 - Built with React, Tailwind CSS & MySQL HeatWave
              </p>
              <p className="text-xs mt-2">
                ⚠️ Trading involves risk. Past performance does not guarantee future results.
              </p>
            </div>
          </footer>
        </main>

        {/* Notifications */}
        <Notifications />
        
        {/* Toast notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
            success: {
              duration: 3000,
              iconTheme: {
                primary: '#10B981',
                secondary: '#fff',
              },
            },
            error: {
              duration: 5000,
              iconTheme: {
                primary: '#EF4444',
                secondary: '#fff',
              },
            },
          }}
        />
      </div>
      
      {/* React Query DevTools - only in development */}
      {import.meta.env.DEV && <ReactQueryDevtools initialIsOpen={false} />}
    </QueryClientProvider>
  );
}

export default App;