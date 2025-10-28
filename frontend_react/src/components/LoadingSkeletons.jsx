/**
 * Loading Skeleton Components
 * Provides smooth loading states for better UX
 */

import React from 'react';
import { motion } from 'framer-motion';

// Base skeleton animation
const skeletonAnimation = {
  animate: {
    opacity: [0.5, 1, 0.5],
  },
  transition: {
    duration: 1.5,
    repeat: Infinity,
    ease: "easeInOut"
  }
};

// Generic skeleton component
export const Skeleton = ({ className = "", width = "100%", height = "20px", ...props }) => (
  <motion.div
    className={`bg-gray-200 dark:bg-gray-700 rounded ${className}`}
    style={{ width, height }}
    {...skeletonAnimation}
    {...props}
  />
);

// Card skeleton for dashboard cards
export const CardSkeleton = () => (
  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
    <div className="flex items-center justify-between mb-4">
      <Skeleton width="120px" height="16px" />
      <Skeleton width="24px" height="24px" className="rounded-full" />
    </div>
    <Skeleton width="80px" height="32px" className="mb-2" />
    <Skeleton width="100px" height="14px" />
  </div>
);

// Table skeleton
export const TableSkeleton = ({ rows = 5, columns = 4 }) => (
  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
    {/* Table header */}
    <div className="border-b border-gray-200 dark:border-gray-700 p-4">
      <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
        {Array.from({ length: columns }).map((_, i) => (
          <Skeleton key={i} width="80px" height="16px" />
        ))}
      </div>
    </div>
    
    {/* Table rows */}
    {Array.from({ length: rows }).map((_, rowIndex) => (
      <div key={rowIndex} className="border-b border-gray-200 dark:border-gray-700 p-4 last:border-b-0">
        <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
          {Array.from({ length: columns }).map((_, colIndex) => (
            <Skeleton key={colIndex} width="100%" height="16px" />
          ))}
        </div>
      </div>
    ))}
  </div>
);

// Chart skeleton
export const ChartSkeleton = ({ height = "300px" }) => (
  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
    <div className="flex items-center justify-between mb-6">
      <Skeleton width="150px" height="20px" />
      <div className="flex gap-2">
        <Skeleton width="60px" height="32px" className="rounded" />
        <Skeleton width="60px" height="32px" className="rounded" />
        <Skeleton width="60px" height="32px" className="rounded" />
      </div>
    </div>
    <Skeleton width="100%" height={height} className="rounded" />
  </div>
);

// Bot status skeleton
export const BotStatusSkeleton = ({ count = 4 }) => (
  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
    <div className="flex items-center justify-between mb-4">
      <Skeleton width="100px" height="20px" />
      <Skeleton width="80px" height="16px" />
    </div>
    
    <div className="space-y-3 max-h-64 overflow-y-auto">
      {Array.from({ length: count }).map((_, index) => (
        <div key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded">
          <div className="flex items-center gap-3">
            <Skeleton width="12px" height="12px" className="rounded-full" />
            <div>
              <Skeleton width="80px" height="16px" className="mb-1" />
              <Skeleton width="60px" height="12px" />
            </div>
          </div>
          <div className="text-right">
            <Skeleton width="40px" height="16px" className="mb-1" />
            <Skeleton width="60px" height="12px" />
          </div>
        </div>
      ))}
    </div>
  </div>
);

// Log entry skeleton
export const LogSkeleton = ({ count = 10 }) => (
  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
    <div className="p-4 border-b border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between">
        <Skeleton width="120px" height="20px" />
        <div className="flex gap-2">
          <Skeleton width="60px" height="32px" className="rounded" />
          <Skeleton width="80px" height="32px" className="rounded" />
        </div>
      </div>
    </div>
    
    <div className="max-h-96 overflow-y-auto">
      {Array.from({ length: count }).map((_, index) => (
        <div key={index} className="p-4 border-b border-gray-200 dark:border-gray-700 last:border-b-0">
          <div className="flex items-start gap-3">
            <Skeleton width="60px" height="20px" className="rounded-full" />
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <Skeleton width="80px" height="14px" />
                <Skeleton width="100px" height="14px" />
              </div>
              <Skeleton width="100%" height="16px" className="mb-1" />
              <Skeleton width="80%" height="16px" />
            </div>
          </div>
        </div>
      ))}
    </div>
  </div>
);

// Metrics skeleton for performance cards
export const MetricsSkeleton = () => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
    {Array.from({ length: 4 }).map((_, index) => (
      <div key={index} className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <Skeleton width="100px" height="16px" />
          <Skeleton width="20px" height="20px" className="rounded-full" />
        </div>
        <Skeleton width="60px" height="28px" className="mb-2" />
        <div className="flex items-center gap-2">
          <Skeleton width="40px" height="14px" />
          <Skeleton width="60px" height="14px" />
        </div>
      </div>
    ))}
  </div>
);

// Full dashboard skeleton
export const DashboardSkeleton = () => (
  <div className="space-y-6">
    {/* Header */}
    <div className="flex items-center justify-between">
      <Skeleton width="200px" height="32px" />
      <div className="flex gap-2">
        <Skeleton width="100px" height="36px" className="rounded" />
        <Skeleton width="80px" height="36px" className="rounded" />
      </div>
    </div>
    
    {/* Metrics cards */}
    <MetricsSkeleton />
    
    {/* Charts and tables */}
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <ChartSkeleton />
      <BotStatusSkeleton />
    </div>
    
    {/* Tables */}
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <TableSkeleton rows={6} columns={5} />
      <LogSkeleton count={8} />
    </div>
  </div>
);

// Loading overlay for buttons and forms
export const LoadingOverlay = ({ isLoading, children, className = "" }) => (
  <div className={`relative ${className}`}>
    {children}
    {isLoading && (
      <div className="absolute inset-0 bg-white/50 dark:bg-gray-900/50 flex items-center justify-center rounded">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-gray-600 dark:text-gray-400">Loading...</span>
        </div>
      </div>
    )}
  </div>
);

// Spinner component
export const Spinner = ({ size = "md", className = "" }) => {
  const sizeClasses = {
    sm: "w-4 h-4",
    md: "w-6 h-6",
    lg: "w-8 h-8",
    xl: "w-12 h-12"
  };

  return (
    <div className={`${sizeClasses[size]} border-2 border-blue-600 border-t-transparent rounded-full animate-spin ${className}`} />
  );
};

// Error state component
export const ErrorState = ({ message = "Something went wrong", onRetry, className = "" }) => (
  <div className={`flex flex-col items-center justify-center p-8 text-center ${className}`}>
    <div className="w-16 h-16 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center mb-4">
      <svg className="w-8 h-8 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    </div>
    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">Error</h3>
    <p className="text-gray-600 dark:text-gray-400 mb-4">{message}</p>
    {onRetry && (
      <button
        onClick={onRetry}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
      >
        Try Again
      </button>
    )}
  </div>
);

// Empty state component
export const EmptyState = ({ 
  title = "No data available", 
  description = "There's nothing to show here yet.", 
  icon,
  action,
  className = "" 
}) => (
  <div className={`flex flex-col items-center justify-center p-8 text-center ${className}`}>
    <div className="w-16 h-16 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mb-4">
      {icon || (
        <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
        </svg>
      )}
    </div>
    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">{title}</h3>
    <p className="text-gray-600 dark:text-gray-400 mb-4">{description}</p>
    {action}
  </div>
);

export default {
  Skeleton,
  CardSkeleton,
  TableSkeleton,
  ChartSkeleton,
  BotStatusSkeleton,
  LogSkeleton,
  MetricsSkeleton,
  DashboardSkeleton,
  LoadingOverlay,
  Spinner,
  ErrorState,
  EmptyState
};