import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { TrendingUp, AlertCircle } from 'lucide-react';
import { useEquity } from '../hooks/useApi';
import { formatCurrency, formatDateTime } from '../utils/formatters';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function EquityChart() {
  const { data: equityData, loading, error } = useEquity();

  const chartData = useMemo(() => {
    if (!equityData || !Array.isArray(equityData)) {
      return {
        labels: [],
        datasets: [],
      };
    }

    const labels = equityData.map(point => {
      try {
        if (!point.timestamp || point.timestamp <= 0) {
          return '--:--';
        }
        
        // Handle both seconds and milliseconds timestamps
        const timestamp = point.timestamp > 1e12 ? point.timestamp : point.timestamp * 1000;
        const date = new Date(timestamp);
        
        if (isNaN(date.getTime())) {
          return '--:--';
        }
        
        return date.toLocaleTimeString('en-US', {
          hour: '2-digit',
          minute: '2-digit',
        });
      } catch (error) {
        console.warn('Invalid timestamp in equity chart:', point.timestamp, error);
        return '--:--';
      }
    });

    const equityValues = equityData.map(point => point.equity);
    const minEquity = Math.min(...equityValues);
    const maxEquity = Math.max(...equityValues);
    const isPositive = equityValues[equityValues.length - 1] >= equityValues[0];

    return {
      labels,
      datasets: [
        {
          label: 'Account Equity',
          data: equityValues,
          borderColor: isPositive ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)',
          backgroundColor: isPositive 
            ? 'rgba(34, 197, 94, 0.1)' 
            : 'rgba(239, 68, 68, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: isPositive ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)',
          pointHoverBorderColor: 'white',
          pointHoverBorderWidth: 2,
        },
      ],
    };
  }, [equityData]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: false,
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        callbacks: {
          label: function(context) {
            return `Equity: ${formatCurrency(context.parsed.y)}`;
          },
          title: function(context) {
            if (context.length > 0 && equityData) {
              const dataPoint = equityData[context[0].dataIndex];
              return formatDateTime(dataPoint.timestamp);
            }
            return '';
          },
        },
      },
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: false,
        },
        ticks: {
          maxTicksLimit: 8,
          color: 'rgb(107, 114, 128)',
        },
      },
      y: {
        display: true,
        grid: {
          color: 'rgba(107, 114, 128, 0.1)',
        },
        ticks: {
          color: 'rgb(107, 114, 128)',
          callback: function(value) {
            return formatCurrency(value);
          },
        },
      },
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false,
    },
    elements: {
      point: {
        hoverRadius: 8,
      },
    },
  };

  if (error) {
    return (
      <div className="card mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Equity Curve</h2>
        </div>
        <div className="h-80 flex items-center justify-center">
          <div className="text-center text-danger-600">
            <AlertCircle className="h-12 w-12 mx-auto mb-2" />
            <p>Failed to load equity data: {error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="card mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Equity Curve</h2>
        </div>
        <div className="h-80 flex items-center justify-center">
          <div className="animate-pulse text-center">
            <div className="h-4 bg-gray-200 rounded w-32 mx-auto mb-4"></div>
            <div className="h-64 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  const currentEquity = equityData && equityData.length > 0 
    ? equityData[equityData.length - 1].equity 
    : 0;
  const initialEquity = equityData && equityData.length > 0 
    ? equityData[0].equity 
    : 0;
  const totalReturn = initialEquity !== 0 
    ? ((currentEquity - initialEquity) / initialEquity) * 100 
    : 0;

  return (
    <div className="card mb-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Equity Curve</h2>
          <p className="text-sm text-gray-600 mt-1">
            Account balance over time
          </p>
        </div>
        <div className="text-right">
          <div className="flex items-center space-x-2">
            <TrendingUp className={`h-5 w-5 ${
              totalReturn >= 0 ? 'text-success-600' : 'text-danger-600'
            }`} />
            <div>
              <p className="text-sm text-gray-600">Current Equity</p>
              <p className="text-lg font-semibold">
                {formatCurrency(currentEquity)}
              </p>
            </div>
          </div>
          <div className="mt-2">
            <p className="text-sm text-gray-600">Total Return</p>
            <p className={`text-lg font-semibold ${
              totalReturn >= 0 ? 'text-success-600' : 'text-danger-600'
            }`}>
              {totalReturn >= 0 ? '+' : ''}{totalReturn.toFixed(2)}%
            </p>
          </div>
        </div>
      </div>

      <div className="h-80">
        {equityData && equityData.length > 0 ? (
          <Line data={chartData} options={chartOptions} />
        ) : (
          <div className="h-full flex items-center justify-center text-gray-500">
            <div className="text-center">
              <TrendingUp className="h-12 w-12 mx-auto mb-2 opacity-50" />
              <p>Waiting for equity data...</p>
              <p className="text-sm">Your equity curve will appear here once trading begins</p>
            </div>
          </div>
        )}
      </div>

      {/* Chart Statistics */}
      {equityData && equityData.length > 0 && (
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-gray-200">
          <div className="text-center">
            <p className="text-sm text-gray-600">Starting Equity</p>
            <p className="font-semibold">{formatCurrency(initialEquity)}</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600">Current Equity</p>
            <p className="font-semibold">{formatCurrency(currentEquity)}</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600">Total P&L</p>
            <p className={`font-semibold ${
              (currentEquity - initialEquity) >= 0 ? 'text-success-600' : 'text-danger-600'
            }`}>
              {formatCurrency(currentEquity - initialEquity)}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600">Data Points</p>
            <p className="font-semibold">{equityData.length}</p>
          </div>
        </div>
      )}
    </div>
  );
}