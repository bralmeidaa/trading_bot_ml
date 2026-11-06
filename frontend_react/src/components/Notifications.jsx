import React, { useState, useEffect } from 'react';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react';
import { notificationService } from '../utils/notifications';

const NotificationIcon = ({ type }) => {
  switch (type) {
    case 'success':
      return <CheckCircle className="h-5 w-5 text-success-600" />;
    case 'error':
      return <AlertCircle className="h-5 w-5 text-danger-600" />;
    case 'warning':
      return <AlertTriangle className="h-5 w-5 text-warning-600" />;
    case 'info':
    default:
      return <Info className="h-5 w-5 text-primary-600" />;
  }
};

const NotificationItem = ({ notification, onClose }) => {
  const getNotificationStyles = (type) => {
    switch (type) {
      case 'success':
        return 'bg-success-50 border-success-200 text-success-800';
      case 'error':
        return 'bg-danger-50 border-danger-200 text-danger-800';
      case 'warning':
        return 'bg-warning-50 border-warning-200 text-warning-800';
      case 'info':
      default:
        return 'bg-primary-50 border-primary-200 text-primary-800';
    }
  };

  return (
    <div
      className={`max-w-sm w-full shadow-lg rounded-lg pointer-events-auto border ${getNotificationStyles(
        notification.type
      )} transform transition-all duration-300 ease-in-out`}
    >
      <div className="p-4">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <NotificationIcon type={notification.type} />
          </div>
          <div className="ml-3 w-0 flex-1 pt-0.5">
            <p className="text-sm font-medium">{notification.message}</p>
          </div>
          <div className="ml-4 flex-shrink-0 flex">
            <button
              className="inline-flex text-gray-400 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              onClick={() => onClose(notification.id)}
            >
              <span className="sr-only">Close</span>
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default function Notifications() {
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    const unsubscribe = notificationService.subscribe(setNotifications);
    return unsubscribe;
  }, []);

  const handleClose = (id) => {
    notificationService.remove(id);
  };

  if (notifications.length === 0) {
    return null;
  }

  return (
    <div
      aria-live="assertive"
      className="fixed inset-0 flex items-end justify-center px-4 py-6 pointer-events-none sm:p-6 sm:items-start sm:justify-end z-50"
    >
      <div className="w-full flex flex-col items-center space-y-4 sm:items-end">
        {notifications.map((notification) => (
          <NotificationItem
            key={notification.id}
            notification={notification}
            onClose={handleClose}
          />
        ))}
      </div>
    </div>
  );
}