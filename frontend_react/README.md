# Trading Bot ML Dashboard

A modern React frontend dashboard for the Trading Bot ML system, providing real-time monitoring, system controls, performance metrics, and configuration management.

## Features

### 🎯 Core Dashboard Components

- **System Control Panel**: Start/stop/emergency controls with real-time status
- **Performance Metrics**: KPI cards showing PnL, ROI, win rate, drawdown
- **Equity Curve Chart**: Interactive line chart of account equity over time
- **Bot Status Table**: Individual bot monitoring with symbol, timeframe, status, PnL
- **Recent Trades & Logs**: Tabbed interface for trades and system logs
- **Configuration Panel**: Advanced system settings and bot parameters

### 🚀 Technical Features

- **Real-time Updates**: Automatic polling of backend APIs every 5 seconds
- **Responsive Design**: Mobile-first design with Tailwind CSS
- **Error Handling**: Graceful error handling with user-friendly messages
- **Loading States**: Visual feedback during API calls and actions
- **Interactive Charts**: Chart.js integration for data visualization
- **Modern UI**: Clean, professional interface with consistent styling

## Technology Stack

- **React 18**: Modern React with hooks and functional components
- **Vite**: Fast development server and build tool
- **Tailwind CSS**: Utility-first CSS framework for styling
- **Chart.js**: Interactive charts and data visualization
- **Lucide React**: Modern icon library
- **Axios**: HTTP client for API communication

## Project Structure

```
frontend_react/
├── src/
│   ├── components/           # React components
│   │   ├── Header.jsx       # System control panel
│   │   ├── PerformanceMetrics.jsx
│   │   ├── EquityChart.jsx
│   │   ├── BotStatusTable.jsx
│   │   ├── TradesAndLogs.jsx
│   │   ├── ConfigurationPanel.jsx
│   │   └── Notifications.jsx
│   ├── services/            # API service layer
│   │   └── api.js          # Backend API integration
│   ├── hooks/              # Custom React hooks
│   │   └── usePolling.js   # Data fetching with polling
│   ├── utils/              # Utility functions
│   │   ├── formatters.js   # Data formatting helpers
│   │   └── notifications.js # Notification service
│   ├── App.jsx             # Main application component
│   ├── main.jsx           # Application entry point
│   └── index.css          # Global styles and Tailwind
├── public/                 # Static assets
├── package.json           # Dependencies and scripts
├── vite.config.js        # Vite configuration
├── tailwind.config.js    # Tailwind CSS configuration
└── postcss.config.js     # PostCSS configuration
```

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Trading Bot ML backend running on port 8000

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend_react
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser to `http://localhost:44261`

### Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## API Integration

The dashboard connects to the Trading Bot ML backend via REST APIs:

- `GET /api/status` - System status and basic info
- `GET /api/metrics` - Performance metrics and KPIs
- `GET /api/equity` - Equity curve data for charts
- `GET /api/bots` - Individual bot status and performance
- `GET /api/trades/recent` - Recent trades data
- `GET /api/logs` - System logs
- `POST /api/system/start` - Start the trading system
- `POST /api/system/stop` - Stop the trading system
- `POST /api/system/emergency` - Emergency stop
- `POST /api/config` - Update system configuration

## Configuration

### Environment Variables

The dashboard uses Vite's proxy configuration to connect to the backend. Update `vite.config.js` to change the backend URL:

```javascript
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',  // Backend URL
        changeOrigin: true
      }
    }
  }
})
```

### Polling Intervals

Data refresh intervals can be configured in the components:

- System status: 5 seconds
- Performance metrics: 5 seconds  
- Bot status: 5 seconds
- Recent trades: 10 seconds

## Component Details

### Header Component
- System status indicator with color coding
- Start/Stop/Emergency control buttons
- Key metrics display (uptime, active bots, total PnL)
- Loading states and error handling

### Performance Metrics
- Real-time KPI cards with color-coded values
- Total PnL, Daily PnL, ROI, Win Rate, Max Drawdown
- Success rate visualization with wins/losses
- Error handling with user-friendly messages

### Equity Chart
- Interactive line chart using Chart.js
- Real-time equity curve updates
- Responsive design with proper scaling
- Loading states and error handling

### Bot Status Table
- Individual bot monitoring
- Symbol, timeframe, status, PnL, trade count
- Enable/disable bot controls
- Status indicators with color coding

### Trades & Logs
- Tabbed interface for trades and logs
- Recent trades table with detailed information
- System logs with timestamp and severity
- Auto-refresh functionality

### Configuration Panel
- System-wide settings form
- Advanced bot parameters
- Form validation and error handling
- Save/reset functionality

## Styling

The dashboard uses Tailwind CSS with custom utility classes:

- `.card` - Standard card container
- `.btn-primary`, `.btn-success`, `.btn-danger` - Button styles
- `.status-running`, `.status-stopped`, `.status-error` - Status indicators
- `.metric-positive`, `.metric-negative` - Value color coding

## Error Handling

The dashboard includes comprehensive error handling:

- API connection errors with retry logic
- Loading states during data fetching
- User-friendly error messages
- Graceful degradation when backend is unavailable

## Performance

- Efficient re-rendering with React hooks
- Optimized API polling with cleanup
- Responsive design for all screen sizes
- Fast development server with Vite

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Follow the existing code style and structure
2. Add proper error handling for new features
3. Include loading states for async operations
4. Test on multiple screen sizes
5. Update this README for significant changes

## License

This project is part of the Trading Bot ML system.