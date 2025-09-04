export function formatCurrency(value, currency = 'USD') {
  if (value === null || value === undefined) return '--';
  
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatPercentage(value, decimals = 2) {
  if (value === null || value === undefined) return '--';
  
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value / 100);
}

export function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined) return '--';
  
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

export function formatDateTime(timestamp) {
  if (!timestamp) return '--';
  
  const date = new Date(timestamp);
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(date);
}

export function formatDuration(seconds) {
  if (!seconds) return '0s';
  
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  
  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  } else {
    return `${secs}s`;
  }
}

export function getMetricColor(value, isPercentage = false) {
  if (value === null || value === undefined) return 'metric-neutral';
  
  const numValue = isPercentage ? value : parseFloat(value);
  
  if (numValue > 0) return 'metric-positive';
  if (numValue < 0) return 'metric-negative';
  return 'metric-neutral';
}

export function getStatusColor(status) {
  switch (status?.toLowerCase()) {
    case 'running':
    case 'active':
    case 'open':
      return 'status-running';
    case 'stopped':
    case 'inactive':
    case 'closed':
      return 'status-stopped';
    case 'error':
    case 'failed':
      return 'status-error';
    default:
      return 'status-stopped';
  }
}