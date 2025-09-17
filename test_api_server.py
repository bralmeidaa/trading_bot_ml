#!/usr/bin/env python3
"""
Test script for the API server functionality.
"""

import asyncio
import sys
import os
import requests
import time
import json
from threading import Thread

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def start_api_server():
    """Start the API server in a separate thread."""
    import uvicorn
    from api_server import app
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def test_api_endpoints():
    """Test all API endpoints."""
    
    print("🌐 TESTING API SERVER ENDPOINTS")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    # Test endpoints
    endpoints = [
        ("/api/status", "GET", "System status"),
        ("/api/metrics", "GET", "Performance metrics"),
        ("/api/trades/recent", "GET", "Recent trades"),
        ("/api/logs", "GET", "System logs"),
        ("/api/config/full", "GET", "System configuration"),
        ("/api/bots", "GET", "Bot status")
    ]
    
    results = {}
    
    for endpoint, method, description in endpoints:
        print(f"\n🧪 Testing {method} {endpoint} - {description}")
        
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
            else:
                response = requests.post(f"{base_url}{endpoint}", timeout=10)
            
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("  ✅ SUCCESS")
                
                # Try to parse JSON response for API endpoints
                if endpoint.startswith("/api/"):
                    try:
                        data = response.json()
                        print(f"  📊 Response keys: {list(data.keys()) if isinstance(data, dict) else 'Non-dict response'}")
                    except:
                        print(f"  📄 Response length: {len(response.text)} chars")
                else:
                    print(f"  📄 HTML response length: {len(response.text)} chars")
                
                results[endpoint] = "PASS"
            else:
                print(f"  ❌ FAILED - {response.status_code}")
                print(f"  Error: {response.text[:200]}...")
                results[endpoint] = "FAIL"
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ CONNECTION ERROR: {e}")
            results[endpoint] = "ERROR"
        except Exception as e:
            print(f"  ❌ UNEXPECTED ERROR: {e}")
            results[endpoint] = "ERROR"
    
    # Test trading system control endpoints
    control_endpoints = [
        ("/api/start", "POST", "Start trading system"),
        ("/api/stop", "POST", "Stop trading system"),
    ]
    
    print(f"\n🎮 TESTING CONTROL ENDPOINTS")
    print("-" * 40)
    
    for endpoint, method, description in control_endpoints:
        print(f"\n🧪 Testing {method} {endpoint} - {description}")
        
        try:
            response = requests.post(f"{base_url}{endpoint}", timeout=10)
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code in [200, 202]:  # Accept both OK and Accepted
                print("  ✅ SUCCESS")
                try:
                    data = response.json()
                    print(f"  📊 Response: {data}")
                except:
                    print(f"  📄 Response: {response.text[:100]}...")
                results[endpoint] = "PASS"
            else:
                print(f"  ❌ FAILED - {response.status_code}")
                results[endpoint] = "FAIL"
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results[endpoint] = "ERROR"
    
    # Summary
    print(f"\n📊 API TEST SUMMARY")
    print("-" * 40)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result == "PASS")
    
    for endpoint, result in results.items():
        status_icon = "✅" if result == "PASS" else "❌"
        print(f"  {status_icon} {endpoint}: {result}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    return passed_tests >= (total_tests * 0.7)  # 70% pass rate

def test_frontend_files():
    """Test if frontend files exist and are accessible."""
    
    print(f"\n📁 TESTING FRONTEND FILES")
    print("-" * 40)
    
    # Check if frontend directory exists
    frontend_paths = [
        "/workspace/trading_bot_ml/frontend_react",
        "/workspace/trading_bot_ml/frontend",
        "/workspace/trading_bot_ml/static"
    ]
    
    frontend_dir = None
    for path in frontend_paths:
        if os.path.exists(path):
            frontend_dir = path
            break
    
    if frontend_dir:
        print(f"✅ Frontend directory found: {frontend_dir}")
        
        # List files
        try:
            files = os.listdir(frontend_dir)
            print(f"📄 Files found: {len(files)}")
            for file in files[:10]:  # Show first 10 files
                print(f"  - {file}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
        except Exception as e:
            print(f"❌ Error listing files: {e}")
    else:
        print("❌ No frontend directory found")
    
    return frontend_dir is not None

def main():
    """Main test function."""
    
    print("🚀 API SERVER TEST")
    print("=" * 80)
    
    # Test 1: Check frontend files
    frontend_ok = test_frontend_files()
    
    # Test 2: Start API server in background
    print(f"\n🚀 Starting API server...")
    server_thread = Thread(target=start_api_server, daemon=True)
    server_thread.start()
    
    # Test 3: Test API endpoints
    api_ok = test_api_endpoints()
    
    print(f"\n📊 FINAL RESULTS")
    print("=" * 40)
    print(f"Frontend files: {'✅ FOUND' if frontend_ok else '❌ MISSING'}")
    print(f"API endpoints: {'✅ WORKING' if api_ok else '❌ ISSUES'}")
    
    overall_success = frontend_ok and api_ok
    print(f"\nOverall: {'✅ SUCCESS' if overall_success else '❌ NEEDS ATTENTION'}")
    
    if overall_success:
        print(f"\n🎉 API server is ready for frontend integration!")
        print(f"   Access the dashboard at: http://localhost:8000")
    else:
        print(f"\n⚠️ Some issues found - check logs above")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    
    # Keep server running for a bit to allow manual testing
    if success:
        print(f"\n⏰ Server will run for 30 seconds for manual testing...")
        time.sleep(30)
    
    sys.exit(0 if success else 1)