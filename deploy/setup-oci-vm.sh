#!/bin/bash
# Setup script for OCI VM - Run this once on your OCI Ubuntu VM

set -e

echo "🚀 Setting up OCI VM for Trading Bot ML..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
echo "🔧 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
echo "👤 Adding user to docker group..."
sudo usermod -aG docker $USER

# Install additional tools
echo "🛠️ Installing additional tools..."
sudo apt-get install -y htop curl wget git unzip

# Create application directory
echo "📁 Creating application directory..."
sudo mkdir -p /opt/trading-bot-ml
sudo chown $USER:$USER /opt/trading-bot-ml
cd /opt/trading-bot-ml

# Create necessary subdirectories
mkdir -p logs data config ssl

# Setup firewall (UFW)
echo "🔥 Configuring firewall..."
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 8000/tcp  # Trading Bot API
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS

# Create systemd service for auto-start
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/trading-bot-ml.service > /dev/null << EOF
[Unit]
Description=Trading Bot ML
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/trading-bot-ml
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable trading-bot-ml.service

# Setup log rotation
echo "📝 Setting up log rotation..."
sudo tee /etc/logrotate.d/trading-bot-ml > /dev/null << EOF
/opt/trading-bot-ml/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
}
EOF

# Create monitoring script
echo "📊 Creating monitoring script..."
tee /opt/trading-bot-ml/monitor.sh > /dev/null << 'EOF'
#!/bin/bash
# Simple monitoring script for Trading Bot ML

echo "=== Trading Bot ML Status ==="
echo "Date: $(date)"
echo ""

echo "🐳 Docker Containers:"
docker-compose ps
echo ""

echo "💾 Disk Usage:"
df -h /opt/trading-bot-ml
echo ""

echo "🔍 Recent Logs (last 10 lines):"
tail -n 10 logs/trading_system.log 2>/dev/null || echo "No logs found"
echo ""

echo "🌐 API Health Check:"
curl -s http://localhost:8000/api/health | jq . 2>/dev/null || echo "API not responding"
echo ""

echo "📈 System Resources:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)% used"
echo "Memory: $(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')"
echo "Load: $(uptime | awk -F'load average:' '{print $2}')"
EOF

chmod +x /opt/trading-bot-ml/monitor.sh

# Create backup script
echo "💾 Creating backup script..."
tee /opt/trading-bot-ml/backup.sh > /dev/null << 'EOF'
#!/bin/bash
# Backup script for Trading Bot ML

BACKUP_DIR="/opt/trading-bot-ml/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

echo "📦 Creating backup: $DATE"

# Backup configuration and data
tar -czf "$BACKUP_DIR/trading-bot-backup-$DATE.tar.gz" \
    config/ \
    data/ \
    logs/ \
    .env \
    docker-compose.yml

# Keep only last 7 backups
find $BACKUP_DIR -name "trading-bot-backup-*.tar.gz" -mtime +7 -delete

echo "✅ Backup completed: $BACKUP_DIR/trading-bot-backup-$DATE.tar.gz"
EOF

chmod +x /opt/trading-bot-ml/backup.sh

# Setup cron job for daily backup
echo "⏰ Setting up daily backup cron job..."
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/trading-bot-ml/backup.sh >> /opt/trading-bot-ml/logs/backup.log 2>&1") | crontab -

# Create environment template
echo "📝 Creating environment template..."
tee /opt/trading-bot-ml/.env.template > /dev/null << 'EOF'
# Trading Bot ML Environment Configuration
# Copy this file to .env and fill in your values

# Binance API Configuration (for live trading)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# System Configuration
ENVIRONMENT=production
PYTHONPATH=/app
PYTHONUNBUFFERED=1

# Security (change these!)
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here

# Optional: Database Configuration
# DATABASE_URL=postgresql://user:password@localhost:5432/trading_bot

# Optional: Redis Configuration
# REDIS_URL=redis://localhost:6379/0
EOF

echo ""
echo "✅ OCI VM setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Copy .env.template to .env and configure your API keys"
echo "2. Deploy your application using Azure DevOps pipeline"
echo "3. Access dashboard at: http://$(curl -s ifconfig.me):8000"
echo ""
echo "🔧 Useful commands:"
echo "  Monitor status: /opt/trading-bot-ml/monitor.sh"
echo "  Create backup: /opt/trading-bot-ml/backup.sh"
echo "  View logs: docker-compose logs -f"
echo "  Restart service: sudo systemctl restart trading-bot-ml"
echo ""
echo "⚠️  Important: Configure your firewall to allow access only from your IP!"
echo "   sudo ufw allow from YOUR_IP_ADDRESS to any port 8000"
echo ""

# Show final status
echo "🔍 Current system status:"
docker --version
docker-compose --version
sudo systemctl status docker