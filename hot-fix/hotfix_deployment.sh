#!/bin/bash
# Hot-fix Deployment Script
# Apply optimizations directly to running container

set -e

echo "🔥 TRADING BOT HOT-FIX DEPLOYMENT"
echo "=================================="
echo "This will apply optimizations directly to your running container"
echo "for immediate improvement in trading activity."
echo ""

# Configuration
CONTAINER_NAME="trading-bot-master"
REMOTE_PATH="/app"

# Check if container is running
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "❌ Container $CONTAINER_NAME is not running!"
    echo "Please start your container first."
    exit 1
fi

echo "📦 Container found: $CONTAINER_NAME"
echo ""

# Step 1: Backup current files in container
echo "1️⃣ Creating backup in container..."
docker exec $CONTAINER_NAME mkdir -p /app/backup_hotfix
docker exec $CONTAINER_NAME cp /app/production_trading_system.py /app/backup_hotfix/ 2>/dev/null || echo "   No existing production_trading_system.py to backup"
echo "   ✅ Backup created in container:/app/backup_hotfix/"

# Step 2: Copy optimization files to container
echo ""
echo "2️⃣ Copying optimization files to container..."

files_to_copy=(
    "enhanced_signal_generator.py"
    "optimized_config.py" 
    "diagnostic_tool.py"
    "apply_optimizations.py"
    "OPTIMIZATION_SOLUTION.md"
)

for file in "${files_to_copy[@]}"; do
    if [ -f "$file" ]; then
        docker cp "$file" $CONTAINER_NAME:$REMOTE_PATH/
        echo "   ✅ Copied $file"
    else
        echo "   ⚠️  $file not found, skipping"
    fi
done

# Step 3: Apply optimizations inside container
echo ""
echo "3️⃣ Applying optimizations inside container..."
docker exec $CONTAINER_NAME python /app/apply_optimizations.py

# Step 4: Restart the trading system
echo ""
echo "4️⃣ Restarting trading system with optimizations..."

# Stop current trading process (if running)
docker exec $CONTAINER_NAME pkill -f "production_trading_system.py" || echo "   No existing process to stop"
docker exec $CONTAINER_NAME pkill -f "api_server.py" || echo "   No existing API server to stop"

# Wait a moment
sleep 3

# Start optimized system in background
echo "   🚀 Starting optimized trading system..."
docker exec -d $CONTAINER_NAME python /app/start_optimized_bot.py

# Start API server in background
echo "   🌐 Starting API server..."
docker exec -d $CONTAINER_NAME python /app/api_server.py

# Step 5: Verify deployment
echo ""
echo "5️⃣ Verifying deployment..."
sleep 10

# Check if processes are running
if docker exec $CONTAINER_NAME pgrep -f "start_optimized_bot.py" > /dev/null; then
    echo "   ✅ Optimized trading bot is running"
else
    echo "   ❌ Trading bot failed to start"
fi

if docker exec $CONTAINER_NAME pgrep -f "api_server.py" > /dev/null; then
    echo "   ✅ API server is running"
else
    echo "   ❌ API server failed to start"
fi

# Check API health
sleep 5
if docker exec $CONTAINER_NAME curl -s http://localhost:8000/api/health > /dev/null; then
    echo "   ✅ API health check passed"
else
    echo "   ⚠️  API health check failed (may need more time)"
fi

echo ""
echo "🎉 HOT-FIX DEPLOYMENT COMPLETED!"
echo "=================================="
echo ""
echo "📊 WHAT TO EXPECT:"
echo "• Trading activity should increase from 2 trades/6 days to 5-20 trades/day"
echo "• Check logs for 'Signal Generated' messages"
echo "• Monitor dashboard for increased activity"
echo "• Signal statistics logged every 100 checks"
echo ""
echo "🔍 MONITORING COMMANDS:"
echo "• View logs: docker logs -f $CONTAINER_NAME"
echo "• Check processes: docker exec $CONTAINER_NAME ps aux | grep python"
echo "• Run diagnostic: docker exec $CONTAINER_NAME python /app/diagnostic_tool.py"
echo ""
echo "⚠️  IMPORTANT NOTES:"
echo "• This is a temporary fix - changes will be lost if container restarts"
echo "• Create a proper PR for permanent deployment"
echo "• Still in paper trading mode for safety"
echo ""
echo "🔄 TO ROLLBACK (if needed):"
echo "docker exec $CONTAINER_NAME cp /app/backup_hotfix/production_trading_system.py /app/"
echo "docker restart $CONTAINER_NAME"