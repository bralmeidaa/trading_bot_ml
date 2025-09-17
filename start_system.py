#!/usr/bin/env python3
"""
Quick start script for the Trading Bot ML system.
Launches both backend API and provides frontend instructions.
"""

import sys
import os
import time
import subprocess
import threading
from datetime import datetime

def print_banner():
    """Print startup banner."""
    print("=" * 80)
    print("🚀 TRADING BOT ML - SISTEMA DE INICIALIZAÇÃO RÁPIDA")
    print("=" * 80)
    print("🤖 Sistema Otimizado de Trading Automatizado")
    print("📊 Backend: Python + FastAPI | Frontend: React + Vite")
    print("💰 Paper Trading Ativo | Gestão Automática de Risco")
    print("=" * 80)

def check_dependencies():
    """Check if all dependencies are available."""
    print("\n🔍 Verificando dependências...")
    
    # Check Python dependencies
    try:
        import pandas as pd
        import numpy as np
        import ccxt
        import fastapi
        import uvicorn
        print("✅ Dependências Python: OK")
    except ImportError as e:
        print(f"❌ Dependência Python faltando: {e}")
        print("   Execute: pip install -r requirements.txt")
        return False
    
    # Check configuration file
    if os.path.exists("trading_config.json"):
        print("✅ Configuração otimizada: OK")
    else:
        print("❌ Arquivo de configuração não encontrado")
        return False
    
    # Check frontend
    if os.path.exists("frontend_react"):
        print("✅ Frontend React: OK")
        
        # Check if node_modules exists
        if os.path.exists("frontend_react/node_modules"):
            print("✅ Dependências Node.js: OK")
        else:
            print("⚠️ Dependências Node.js não instaladas")
            print("   Execute: cd frontend_react && npm install")
    else:
        print("❌ Frontend não encontrado")
        return False
    
    return True

def start_api_server():
    """Start the API server."""
    print("\n🌐 Iniciando servidor API...")
    
    try:
        import uvicorn
        from api_server import app
        
        print("✅ Servidor API iniciado na porta 8000")
        print("🔗 API disponível em: http://localhost:8000")
        print("📊 Status: http://localhost:8000/api/status")
        
        # Start server
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
        
    except Exception as e:
        print(f"❌ Erro ao iniciar API: {e}")
        return False

def show_frontend_instructions():
    """Show frontend startup instructions."""
    print("\n📱 INSTRUÇÕES PARA O FRONTEND")
    print("-" * 50)
    print("🚀 Para iniciar o dashboard web:")
    print()
    print("1️⃣ Abrir NOVO TERMINAL:")
    print("   cd /workspace/trading_bot_ml/frontend_react")
    print()
    print("2️⃣ Instalar dependências (se necessário):")
    print("   npm install")
    print()
    print("3️⃣ Iniciar servidor de desenvolvimento:")
    print("   npm run dev")
    print()
    print("4️⃣ Acessar dashboard:")
    print("   http://localhost:5173")
    print()
    print("📊 O dashboard oferece:")
    print("   • 📈 Métricas de performance em tempo real")
    print("   • 🤖 Controle dos bots de trading")
    print("   • 💰 Histórico de trades e PnL")
    print("   • ⚙️ Configuração do sistema")
    print("   • 📋 Logs e monitoramento")

def show_system_info():
    """Show system information."""
    print("\n📊 INFORMAÇÕES DO SISTEMA OTIMIZADO")
    print("-" * 50)
    print("🤖 Configuração dos Bots:")
    print("   • BTC/USDT 5m - 30% capital - Conservador")
    print("   • ETH/USDT 5m - 25% capital - Balanceado")
    print("   • LINK/USDT 3m - 20% capital - Ativo")
    print("   • LINK/USDT 1m - 15% capital - Scalping")
    print("   • SOL/USDT 5m - 10% capital - Especulativo")
    print()
    print("📈 Performance Esperada:")
    print("   • Win Rate: ~43%")
    print("   • Trades/dia: 15-25")
    print("   • Risk/Reward: 2.2:1")
    print("   • Max Drawdown: <5%")
    print()
    print("🛡️ Gestão de Risco:")
    print("   • Paper Trading ativo")
    print("   • Stop Loss automático")
    print("   • Limite perda diária: 3.5%")
    print("   • Emergency stop: 8%")

def show_usage_tips():
    """Show usage tips."""
    print("\n💡 DICAS DE USO")
    print("-" * 50)
    print("🎯 Fluxo Recomendado:")
    print("   1. Iniciar API (este script)")
    print("   2. Iniciar frontend (novo terminal)")
    print("   3. Acessar dashboard web")
    print("   4. Clicar 'Start System' no dashboard")
    print("   5. Monitorar performance")
    print()
    print("📊 Monitoramento:")
    print("   • Verificar métricas a cada hora")
    print("   • Acompanhar PnL diário")
    print("   • Observar alertas no dashboard")
    print("   • Revisar logs regularmente")
    print()
    print("⚠️ Importante:")
    print("   • Sistema em PAPER TRADING (simulação)")
    print("   • Teste por 1 semana antes de capital real")
    print("   • Mantenha backups da configuração")
    print("   • Pare o sistema se drawdown > 5%")

def main():
    """Main startup function."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Falha na verificação de dependências")
        print("   Corrija os problemas acima e tente novamente")
        return False
    
    # Show system information
    show_system_info()
    
    # Show frontend instructions
    show_frontend_instructions()
    
    # Show usage tips
    show_usage_tips()
    
    # Confirm startup
    print("\n🚀 INICIANDO SISTEMA")
    print("-" * 50)
    print("⏰ Aguarde alguns segundos para inicialização...")
    print("🌐 API será iniciada na porta 8000")
    print("📱 Frontend deve ser iniciado manualmente na porta 5173")
    print()
    print("🔄 Para parar o sistema: Ctrl+C")
    print("=" * 80)
    
    # Small delay
    time.sleep(2)
    
    # Start API server (this will block)
    try:
        start_api_server()
    except KeyboardInterrupt:
        print("\n\n🛑 Sistema interrompido pelo usuário")
        print("✅ Shutdown completo")
        return True
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)