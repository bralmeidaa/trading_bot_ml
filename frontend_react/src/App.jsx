import React, { useState } from 'react';
import Header from './components/Header';
import PerformanceMetrics from './components/PerformanceMetrics';
import EquityChart from './components/EquityChart';
import BotStatusTable from './components/BotStatusTable';
import TradesAndLogs from './components/TradesAndLogs';
import ConfigurationPanel from './components/ConfigurationPanel';
import Notifications from './components/Notifications';

function App() {
  const [showConfig, setShowConfig] = useState(false);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <Header />

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Configuration Toggle */}
        <div className="mb-6 flex justify-end">
          <button
            onClick={() => setShowConfig(!showConfig)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors duration-200 ${
              showConfig
                ? 'bg-primary-600 text-white hover:bg-primary-700'
                : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
            }`}
          >
            {showConfig ? 'Hide Configuration' : 'Show Configuration'}
          </button>
        </div>

        {/* Configuration Panel */}
        {showConfig && <ConfigurationPanel />}

        {/* Performance Metrics */}
        <PerformanceMetrics />

        {/* Equity Chart */}
        <EquityChart />

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mb-8">
          {/* Bot Status Table */}
          <div className="xl:col-span-2">
            <BotStatusTable />
          </div>
        </div>

        {/* Trades and Logs */}
        <TradesAndLogs />

        {/* Footer */}
        <footer className="mt-12 py-8 border-t border-gray-200">
          <div className="text-center text-gray-500">
            <p className="text-sm">
              Trading Bot ML Dashboard - Built with React & Tailwind CSS
            </p>
            <p className="text-xs mt-2">
              ⚠️ Trading involves risk. Past performance does not guarantee future results.
            </p>
          </div>
        </footer>
      </main>

      {/* Notifications */}
      <Notifications />
    </div>
  );
}

export default App;