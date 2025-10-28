import React from 'react';
import { motion } from 'framer-motion';
import { Shield, CheckCircle, XCircle, Clock, AlertTriangle } from 'lucide-react';
import { useCheckTradingAllowed } from '../services/apiClient';

const TradingStatusIndicator = ({ className = '' }) => {
  const { data: tradingStatus, isLoading, error } = useCheckTradingAllowed();

  if (isLoading) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 ${className}`}>
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gray-100 dark:bg-gray-700 rounded-lg animate-pulse" />
          <div className="flex-1">
            <div className="h-4 bg-gray-200 dark:bg-gray-600 rounded animate-pulse mb-2" />
            <div className="h-3 bg-gray-100 dark:bg-gray-700 rounded animate-pulse w-2/3" />
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg border border-red-200 dark:border-red-800 p-4 ${className}`}>
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-red-100 dark:bg-red-900/20 rounded-lg">
            <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400" />
          </div>
          <div>
            <p className="text-sm font-medium text-red-900 dark:text-red-100">
              Erro ao verificar status
            </p>
            <p className="text-xs text-red-600 dark:text-red-400">
              Não foi possível verificar as restrições
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (!tradingStatus) {
    return null;
  }

  const isAllowed = tradingStatus.is_trading_allowed;
  const blockedBy = tradingStatus.blocked_by || [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-white dark:bg-gray-800 rounded-lg border ${
        isAllowed 
          ? 'border-green-200 dark:border-green-800' 
          : 'border-red-200 dark:border-red-800'
      } p-4 ${className}`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${
            isAllowed 
              ? 'bg-green-100 dark:bg-green-900/20' 
              : 'bg-red-100 dark:bg-red-900/20'
          }`}>
            {isAllowed ? (
              <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
            ) : (
              <Shield className="w-5 h-5 text-red-600 dark:text-red-400" />
            )}
          </div>
          
          <div className="flex-1">
            <div className="flex items-center space-x-2">
              <h3 className={`font-semibold ${
                isAllowed 
                  ? 'text-green-900 dark:text-green-100' 
                  : 'text-red-900 dark:text-red-100'
              }`}>
                Trading {isAllowed ? 'Permitido' : 'Bloqueado'}
              </h3>
              
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                isAllowed
                  ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
                  : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
              }`}>
                {isAllowed ? 'Ativo' : 'Restrito'}
              </span>
            </div>
            
            <div className="flex items-center space-x-4 mt-2 text-xs text-gray-600 dark:text-gray-400">
              <div className="flex items-center space-x-1">
                <Clock className="w-3 h-3" />
                <span>{tradingStatus.time_of_day}</span>
              </div>
              
              {tradingStatus.total_active_restrictions > 0 && (
                <div className="flex items-center space-x-1">
                  <Shield className="w-3 h-3" />
                  <span>{tradingStatus.total_active_restrictions} restrições ativas</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Status Indicator */}
        <div className={`w-3 h-3 rounded-full ${
          isAllowed ? 'bg-green-500' : 'bg-red-500'
        } animate-pulse`} />
      </div>

      {/* Blocked By Details */}
      {!isAllowed && blockedBy.length > 0 && (
        <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
          <p className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
            Bloqueado por:
          </p>
          <div className="space-y-1">
            {blockedBy.map((restriction, index) => (
              <div key={index} className="flex items-center space-x-2 text-xs">
                <div className="w-1.5 h-1.5 bg-red-500 rounded-full" />
                <span className="text-gray-600 dark:text-gray-400">
                  {restriction.name}: {restriction.formatted_period}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Additional Info */}
      {isAllowed && tradingStatus.total_active_restrictions > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
          <p className="text-xs text-gray-600 dark:text-gray-400">
            {tradingStatus.total_active_restrictions} restrições configuradas, mas nenhuma ativa no momento
          </p>
        </div>
      )}
    </motion.div>
  );
};

export default TradingStatusIndicator;