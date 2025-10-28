/**
 * Date utilities using date-fns for safe date parsing and formatting
 * Fixes "Invalid Date" bugs throughout the application
 */

import { 
  format, 
  parseISO, 
  isValid, 
  formatDistanceToNow, 
  subDays, 
  subWeeks, 
  subMonths, 
  startOfYear,
  isAfter,
  isBefore
} from 'date-fns';

/**
 * Safely parse and format a date string or timestamp
 * @param {string|number|Date} dateInput - Date input in various formats
 * @param {string} formatString - Format string for output
 * @returns {string} Formatted date string or 'Invalid Date'
 */
export const safeFormatDate = (dateInput, formatString = 'MMM dd, yyyy HH:mm:ss') => {
  try {
    if (!dateInput) return 'N/A';
    
    let date;
    
    // Handle different input types
    if (typeof dateInput === 'string') {
      // Try to parse ISO string first
      if (dateInput.includes('T') || dateInput.includes('-')) {
        date = parseISO(dateInput);
      } else {
        // Try to parse as timestamp
        const timestamp = parseInt(dateInput);
        if (!isNaN(timestamp)) {
          date = new Date(timestamp);
        } else {
          date = new Date(dateInput);
        }
      }
    } else if (typeof dateInput === 'number') {
      // Handle timestamp (both seconds and milliseconds)
      const timestamp = dateInput < 10000000000 ? dateInput * 1000 : dateInput;
      date = new Date(timestamp);
    } else if (dateInput instanceof Date) {
      date = dateInput;
    } else {
      return 'Invalid Date';
    }
    
    // Check if date is valid
    if (!isValid(date)) {
      console.warn('Invalid date input:', dateInput);
      return 'Invalid Date';
    }
    
    return format(date, formatString);
  } catch (error) {
    console.error('Error formatting date:', error, 'Input:', dateInput);
    return 'Invalid Date';
  }
};

/**
 * Format date for display in tables
 * @param {string|number|Date} dateInput 
 * @returns {string}
 */
export const formatTableDate = (dateInput) => {
  return safeFormatDate(dateInput, 'MMM dd, HH:mm');
};

/**
 * Format date for detailed view
 * @param {string|number|Date} dateInput 
 * @returns {string}
 */
export const formatDetailedDate = (dateInput) => {
  return safeFormatDate(dateInput, 'MMMM dd, yyyy \'at\' HH:mm:ss');
};

/**
 * Format relative time (e.g., "2 hours ago")
 * @param {string|number|Date} dateInput 
 * @returns {string}
 */
export const formatRelativeTime = (dateInput) => {
  try {
    if (!dateInput) return 'N/A';
    
    let date;
    if (typeof dateInput === 'string') {
      date = parseISO(dateInput);
    } else if (typeof dateInput === 'number') {
      const timestamp = dateInput < 10000000000 ? dateInput * 1000 : dateInput;
      date = new Date(timestamp);
    } else {
      date = dateInput;
    }
    
    if (!isValid(date)) return 'Invalid Date';
    
    return formatDistanceToNow(date, { addSuffix: true });
  } catch (error) {
    console.error('Error formatting relative time:', error);
    return 'Invalid Date';
  }
};

/**
 * Get date range for time period selector
 * @param {string} period - '1D', '7D', '1M', 'YTD', 'All'
 * @returns {Object} { startDate, endDate }
 */
export const getDateRangeForPeriod = (period) => {
  const now = new Date();
  let startDate;
  
  switch (period) {
    case '1D':
      startDate = subDays(now, 1);
      break;
    case '7D':
      startDate = subDays(now, 7);
      break;
    case '1M':
      startDate = subMonths(now, 1);
      break;
    case '3M':
      startDate = subMonths(now, 3);
      break;
    case '6M':
      startDate = subMonths(now, 6);
      break;
    case 'YTD':
      startDate = startOfYear(now);
      break;
    case 'All':
      startDate = new Date('2020-01-01'); // Arbitrary old date
      break;
    default:
      startDate = subDays(now, 7);
  }
  
  return {
    startDate: startDate.toISOString(),
    endDate: now.toISOString()
  };
};

/**
 * Validate if a date string is valid
 * @param {string} dateString 
 * @returns {boolean}
 */
export const isValidDateString = (dateString) => {
  try {
    if (!dateString) return false;
    const date = parseISO(dateString);
    return isValid(date);
  } catch {
    return false;
  }
};

/**
 * Format duration in minutes to human readable format
 * @param {number} minutes 
 * @returns {string}
 */
export const formatDuration = (minutes) => {
  if (!minutes || minutes < 0) return '0m';
  
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  
  if (hours > 0) {
    return `${hours}h ${mins}m`;
  } else {
    return `${mins}m`;
  }
};

/**
 * Check if date is within a specific range
 * @param {Date} date 
 * @param {Date} startDate 
 * @param {Date} endDate 
 * @returns {boolean}
 */
export const isDateInRange = (date, startDate, endDate) => {
  try {
    return isAfter(date, startDate) && isBefore(date, endDate);
  } catch {
    return false;
  }
};

/**
 * Format timestamp for API requests
 * @param {Date} date 
 * @returns {string}
 */
export const formatForAPI = (date) => {
  try {
    if (!date || !isValid(date)) return '';
    return date.toISOString();
  } catch {
    return '';
  }
};

/**
 * Parse various date formats safely
 * @param {any} input 
 * @returns {Date|null}
 */
export const safeParseDateInput = (input) => {
  try {
    if (!input) return null;
    
    if (input instanceof Date) {
      return isValid(input) ? input : null;
    }
    
    if (typeof input === 'string') {
      const date = parseISO(input);
      return isValid(date) ? date : null;
    }
    
    if (typeof input === 'number') {
      const timestamp = input < 10000000000 ? input * 1000 : input;
      const date = new Date(timestamp);
      return isValid(date) ? date : null;
    }
    
    return null;
  } catch {
    return null;
  }
};

// Time period options for selectors
export const TIME_PERIODS = [
  { value: '1D', label: '1 Day' },
  { value: '7D', label: '7 Days' },
  { value: '1M', label: '1 Month' },
  { value: '3M', label: '3 Months' },
  { value: '6M', label: '6 Months' },
  { value: 'YTD', label: 'Year to Date' },
  { value: 'All', label: 'All Time' }
];

export default {
  safeFormatDate,
  formatTableDate,
  formatDetailedDate,
  formatRelativeTime,
  getDateRangeForPeriod,
  isValidDateString,
  formatDuration,
  isDateInRange,
  formatForAPI,
  safeParseDateInput,
  TIME_PERIODS
};