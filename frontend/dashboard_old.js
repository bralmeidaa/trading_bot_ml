// Dashboard JavaScript for Trading Bot ML
class TradingDashboard {
    constructor() {
        this.apiUrl = '/api';
        this.equityChart = null;
        this.systemRunning = true;
        this.refreshInterval = null;
        
        this.init();
    }
    
    init() {
        this.initializeCharts();
        this.loadInitialData();
        this.startAutoRefresh();
        this.setupEventListeners();
    }
    
    initializeCharts() {
        const ctx = document.getElementById('equityChart').getContext('2d');
        this.equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Equity ($)',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    },
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return 'Equity: $' + context.parsed.y.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    }
    
    async loadInitialData() {
        try {
            await this.updateSystemStatus();
            await this.updatePerformanceMetrics();
            await this.updateBotStatus();
            await this.updateRecentTrades();
            await this.updateEquityChart();
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showNotification('Erro ao carregar dados iniciais', 'error');
        }
    }
    
    async updateSystemStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/status`);
            const data = await response.json();
            
            const statusElement = document.getElementById('system-status');
            if (data.running) {
                statusElement.innerHTML = '<i class="fas fa-circle status-running"></i> Online';
                this.systemRunning = true;
            } else {
                statusElement.innerHTML = '<i class="fas fa-circle status-stopped"></i> Offline';
                this.systemRunning = false;
            }
        } catch (error) {
            // Simulate data for demo
            this.simulateSystemStatus();
        }
    }
    
    async updatePerformanceMetrics() {
        try {
            const response = await fetch(`${this.apiUrl}/metrics`);
            const data = await response.json();
            
            document.getElementById('total-pnl').textContent = `$${data.total_pnl.toFixed(2)}`;
            document.getElementById('total-roi').textContent = `${(data.total_roi * 100).toFixed(2)}%`;
            document.getElementById('daily-pnl').textContent = `$${data.daily_pnl.toFixed(2)}`;
            document.getElementById('active-trades').textContent = data.active_trades;
            document.getElementById('win-rate').textContent = `${(data.win_rate * 100).toFixed(1)}%`;
            
            // Update colors based on performance
            this.updateMetricColors(data);
        } catch (error) {
            // Simulate data for demo
            this.simulatePerformanceMetrics();
        }
    }
    
    async updateBotStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/bots`);
            const data = await response.json();
            
            const botStatusContainer = document.getElementById('bot-status');
            botStatusContainer.innerHTML = '';
            
            data.bots.forEach(bot => {
                const botElement = this.createBotStatusElement(bot);
                botStatusContainer.appendChild(botElement);
            });
        } catch (error) {
            // Simulate data for demo
            this.simulateBotStatus();
        }
    }
    
    async updateRecentTrades() {
        try {
            const response = await fetch(`${this.apiUrl}/trades/recent`);
            const data = await response.json();
            
            const tradesTable = document.getElementById('trades-table');
            tradesTable.innerHTML = '';
            
            data.trades.forEach(trade => {
                const row = this.createTradeRow(trade);
                tradesTable.appendChild(row);
            });
        } catch (error) {
            // Simulate data for demo
            this.simulateRecentTrades();
        }
    }
    
    async updateEquityChart() {
        try {
            const response = await fetch(`${this.apiUrl}/equity`);
            const data = await response.json();
            
            const labels = data.equity_curve.map(point => new Date(point.timestamp));
            const values = data.equity_curve.map(point => point.equity);
            
            this.equityChart.data.labels = labels;
            this.equityChart.data.datasets[0].data = values;
            this.equityChart.update();
        } catch (error) {
            // Simulate data for demo
            this.simulateEquityChart();
        }
    }
    
    // Simulation methods for demo purposes
    simulateSystemStatus() {
        const statusElement = document.getElementById('system-status');
        statusElement.innerHTML = '<i class="fas fa-circle status-running"></i> Online (Demo)';
    }
    
    simulatePerformanceMetrics() {
        // Generate realistic demo data
        const totalPnl = Math.random() * 1000 - 200; // -200 to 800
        const dailyPnl = Math.random() * 200 - 50; // -50 to 150
        const activeTrades = Math.floor(Math.random() * 4);
        const winRate = 0.6 + Math.random() * 0.2; // 60-80%
        
        document.getElementById('total-pnl').textContent = `$${totalPnl.toFixed(2)}`;
        document.getElementById('total-roi').textContent = `${(totalPnl / 10000 * 100).toFixed(2)}%`;
        document.getElementById('daily-pnl').textContent = `$${dailyPnl.toFixed(2)}`;
        document.getElementById('active-trades').textContent = activeTrades;
        document.getElementById('win-rate').textContent = `${(winRate * 100).toFixed(1)}%`;
        
        this.updateMetricColors({
            total_pnl: totalPnl,
            daily_pnl: dailyPnl,
            win_rate: winRate
        });
    }
    
    simulateBotStatus() {
        const bots = [
            { symbol: 'LINK/USDT', timeframe: '5m', status: 'running', pnl: 156.78, trades: 3 },
            { symbol: 'LINK/USDT', timeframe: '1m', status: 'running', pnl: 89.45, trades: 8 },
            { symbol: 'ADA/USDT', timeframe: '1m', status: 'running', pnl: 234.12, trades: 5 },
            { symbol: 'ADA/USDT', timeframe: '5m', status: 'paused', pnl: -45.67, trades: 2 }
        ];
        
        const botStatusContainer = document.getElementById('bot-status');
        botStatusContainer.innerHTML = '';
        
        bots.forEach(bot => {
            const botElement = this.createBotStatusElement(bot);
            botStatusContainer.appendChild(botElement);
        });
    }
    
    simulateRecentTrades() {
        const trades = [
            { symbol: 'LINK/USDT', direction: 'LONG', pnl: 45.67, status: 'closed', time: '10:30' },
            { symbol: 'ADA/USDT', direction: 'SHORT', pnl: -23.45, status: 'closed', time: '10:15' },
            { symbol: 'LINK/USDT', direction: 'LONG', pnl: 78.90, status: 'open', time: '10:00' },
            { symbol: 'ADA/USDT', direction: 'LONG', pnl: 34.56, status: 'closed', time: '09:45' },
            { symbol: 'LINK/USDT', direction: 'SHORT', pnl: -12.34, status: 'closed', time: '09:30' }
        ];
        
        const tradesTable = document.getElementById('trades-table');
        tradesTable.innerHTML = '';
        
        trades.forEach(trade => {
            const row = this.createTradeRow(trade);
            tradesTable.appendChild(row);
        });
    }
    
    simulateEquityChart() {
        const now = new Date();
        const labels = [];
        const values = [];
        let equity = 10000;
        
        for (let i = 100; i >= 0; i--) {
            const time = new Date(now.getTime() - i * 5 * 60 * 1000); // 5 minutes intervals
            labels.push(time);
            
            // Simulate equity changes
            equity += (Math.random() - 0.4) * 50; // Slight upward bias
            values.push(Math.max(equity, 9000)); // Don't go below 9000
        }
        
        this.equityChart.data.labels = labels;
        this.equityChart.data.datasets[0].data = values;
        this.equityChart.update();
    }
    
    createBotStatusElement(bot) {
        const div = document.createElement('div');
        div.className = 'flex items-center justify-between p-3 bg-gray-50 rounded-lg';
        
        const statusColor = bot.status === 'running' ? 'text-green-600' : 
                           bot.status === 'paused' ? 'text-yellow-600' : 'text-red-600';
        const pnlColor = bot.pnl >= 0 ? 'text-green-600' : 'text-red-600';
        
        div.innerHTML = `
            <div class="flex items-center">
                <div class="w-3 h-3 rounded-full ${bot.status === 'running' ? 'bg-green-500' : 'bg-gray-400'} mr-3"></div>
                <div>
                    <div class="font-medium">${bot.symbol} ${bot.timeframe}</div>
                    <div class="text-sm text-gray-500">${bot.trades} trades</div>
                </div>
            </div>
            <div class="text-right">
                <div class="font-medium ${pnlColor}">$${bot.pnl.toFixed(2)}</div>
                <div class="text-sm ${statusColor}">${bot.status}</div>
            </div>
        `;
        
        return div;
    }
    
    createTradeRow(trade) {
        const row = document.createElement('tr');
        row.className = 'border-b border-gray-100';
        
        const directionColor = trade.direction === 'LONG' ? 'text-green-600' : 'text-red-600';
        const pnlColor = trade.pnl >= 0 ? 'text-green-600' : 'text-red-600';
        const statusColor = trade.status === 'open' ? 'text-blue-600' : 'text-gray-600';
        
        row.innerHTML = `
            <td class="px-3 py-2">${trade.symbol}</td>
            <td class="px-3 py-2 ${directionColor}">${trade.direction}</td>
            <td class="px-3 py-2 ${pnlColor}">$${trade.pnl.toFixed(2)}</td>
            <td class="px-3 py-2 ${statusColor}">${trade.status}</td>
        `;
        
        return row;
    }
    
    updateMetricColors(data) {
        const totalPnlElement = document.getElementById('total-pnl');
        const dailyPnlElement = document.getElementById('daily-pnl');
        const winRateElement = document.getElementById('win-rate');
        
        // Update total PnL color
        totalPnlElement.className = data.total_pnl >= 0 ? 
            'text-2xl font-bold text-green-600' : 'text-2xl font-bold text-red-600';
        
        // Update daily PnL color
        dailyPnlElement.className = data.daily_pnl >= 0 ? 
            'text-2xl font-bold text-blue-600' : 'text-2xl font-bold text-red-600';
        
        // Update win rate color
        winRateElement.className = data.win_rate >= 0.6 ? 
            'text-2xl font-bold text-orange-600' : 'text-2xl font-bold text-red-600';
    }
    
    setupEventListeners() {
        // Auto-refresh toggle
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.stopAutoRefresh();
            } else {
                this.startAutoRefresh();
            }
        });
    }
    
    startAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        this.refreshInterval = setInterval(() => {
            this.loadInitialData();
        }, 30000); // Refresh every 30 seconds
    }
    
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
            type === 'error' ? 'bg-red-500 text-white' : 
            type === 'success' ? 'bg-green-500 text-white' : 
            'bg-blue-500 text-white'
        }`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 3000);
    }
}

// Global functions for UI interactions
function toggleSystem() {
    const modal = document.getElementById('emergency-modal');
    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

function closeEmergencyModal() {
    const modal = document.getElementById('emergency-modal');
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

async function emergencyStop() {
    try {
        const response = await fetch('/api/emergency-stop', { method: 'POST' });
        if (response.ok) {
            dashboard.showNotification('Sistema parado com sucesso', 'success');
            document.getElementById('system-status').innerHTML = '<i class="fas fa-circle status-stopped"></i> Offline';
        } else {
            throw new Error('Falha ao parar o sistema');
        }
    } catch (error) {
        dashboard.showNotification('Erro ao parar o sistema', 'error');
    }
    closeEmergencyModal();
}

async function saveConfiguration() {
    const config = {
        trading_mode: document.getElementById('trading-mode').value,
        total_capital: parseFloat(document.getElementById('total-capital').value),
        daily_loss_limit: parseFloat(document.getElementById('daily-loss-limit').value) / 100,
        daily_profit_target: parseFloat(document.getElementById('daily-profit-target').value) / 100
    };
    
    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (response.ok) {
            dashboard.showNotification('Configurações salvas com sucesso', 'success');
        } else {
            throw new Error('Falha ao salvar configurações');
        }
    } catch (error) {
        dashboard.showNotification('Erro ao salvar configurações', 'error');
    }
}

function refreshTrades() {
    dashboard.updateRecentTrades();
    dashboard.showNotification('Trades atualizados', 'success');
}

// Initialize dashboard when page loads
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new TradingDashboard();
});