#!/usr/bin/env python3
"""
Test script to verify that the frontend fix is working correctly.
This script starts the API server and tests both frontend and API endpoints.
"""

import asyncio
import httpx
import uvicorn
import threading
import time
from pathlib import Path

def test_endpoints():
    """Test both frontend and API endpoints."""
    time.sleep(3)  # Wait for server to start
    
    try:
        with httpx.Client() as client:
            # Test frontend (root path)
            print("🧪 Testing frontend at root path '/'...")
            response = client.get("http://localhost:12000/")
            if response.status_code == 200 and "Trading Bot ML Dashboard" in response.text:
                print("✅ Frontend is working! Status:", response.status_code)
                print("✅ HTML title found in response")
            else:
                print("❌ Frontend test failed. Status:", response.status_code)
            
            # Test API endpoint
            print("\n🧪 Testing API endpoint '/api/status'...")
            response = client.get("http://localhost:12000/api/status")
            if response.status_code == 200:
                print("✅ API is working! Status:", response.status_code)
                print("✅ Response:", response.json())
            else:
                print("❌ API test failed. Status:", response.status_code)
                
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Trading Bot API Server for testing...")
    print("📁 Frontend build directory exists:", Path("frontend_react/dist/index.html").exists())
    
    # Start test in a separate thread
    test_thread = threading.Thread(target=test_endpoints)
    test_thread.daemon = True
    test_thread.start()
    
    # Import and run the server
    from api_server import app
    
    # Run server with timeout
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=12000,
            log_level="info",
            access_log=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")