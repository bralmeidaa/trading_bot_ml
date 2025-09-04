class NotificationService {
  constructor() {
    this.notifications = [];
    this.listeners = [];
  }

  subscribe(callback) {
    this.listeners.push(callback);
    return () => {
      this.listeners = this.listeners.filter(listener => listener !== callback);
    };
  }

  notify(message, type = 'info', duration = 5000) {
    const notification = {
      id: Date.now() + Math.random(),
      message,
      type,
      timestamp: Date.now(),
      duration,
    };

    this.notifications.push(notification);
    this.listeners.forEach(listener => listener(this.notifications));

    if (duration > 0) {
      setTimeout(() => {
        this.remove(notification.id);
      }, duration);
    }

    return notification.id;
  }

  remove(id) {
    this.notifications = this.notifications.filter(n => n.id !== id);
    this.listeners.forEach(listener => listener(this.notifications));
  }

  success(message, duration = 5000) {
    return this.notify(message, 'success', duration);
  }

  error(message, duration = 8000) {
    return this.notify(message, 'error', duration);
  }

  warning(message, duration = 6000) {
    return this.notify(message, 'warning', duration);
  }

  info(message, duration = 5000) {
    return this.notify(message, 'info', duration);
  }

  clear() {
    this.notifications = [];
    this.listeners.forEach(listener => listener(this.notifications));
  }
}

export const notificationService = new NotificationService();