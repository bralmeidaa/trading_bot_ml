import React, { useState } from 'react';
import Header from './components/Header';
import PerformanceMetrics from './components/PerformanceMetrics';
import EquityChart from './components/EquityChart';
import BotStatusTable from './components/BotStatusTable';
import TradesAndLogs from './components/TradesAndLogs';
import ConfigurationPanel from './components/ConfigurationPanel';
import LogViewer from './components/LogViewer';
import Notifications from './components/Notifications';

function App() {
  const [activeView, setActiveView] = useState('dashboard');

  const renderView = () => {
    switch (activeView) {
      case 'dashboard':
        return (
          <>
            <PerformanceMetrics />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <EquityChart />
              <BotStatusTable />
            </div>
            <TradesAndLogs />
          </>
        );
      case 'config':
        return <ConfigurationPanel />;
      case 'logs':
        return <LogViewer />;
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <Header />

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Navigation Tabs */}
        <div className="mb-6">
          <nav className="flex space-x-8">
            {[
              { id: 'dashboard', label: 'Dashboard', icon: '📊' },
              { id: 'config', label: 'Configuration', icon: '⚙️' },
              { id: 'logs', label: 'Logs & Export', icon: '📋' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveView(tab.id)}
                className={`flex items-center px-4 py-2 rounded-lg font-medium transition-colors duration-200 ${
                  activeView === tab.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
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