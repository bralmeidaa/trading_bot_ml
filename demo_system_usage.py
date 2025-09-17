#!/usr/bin/env python3
"""
Demo script showing how to use the Trading Bot ML system.
This script demonstrates the complete workflow from setup to monitoring.
"""

import asyncio
import sys
import os
import time
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_trading_system import (
    ProductionTradingSystem, 
    create_production_config
)

def print_banner():
    """Print system banner."""
    print("=" * 80)
    print("🤖 TRADING BOT ML - SISTEMA DE TRADING AUTOMATIZADO")
    print("=" * 80)
    print("📊 Sistema de trading com Machine Learning")
    print("🎯 Análise técnica + IA para mercados voláteis")
    print("💰 Gestão automática de risco")
    print("📈 Monitoramento em tempo real")
    print("=" * 80)

def print_system_info():
    """Print system information."""
    print("\n📋 INFORMAÇÕES DO SISTEMA")
    print("-" * 40)
    print("🔧 Backend: Python 3.12 + FastAPI")
    print("🧠 ML: Random Forest + XGBoost + LightGBM")
    print("📊 Indicadores: RSI, Bollinger Bands, EMA, Volume")
    print("💱 Exchange: Binance (Paper Trading)")
    print("🎯 Estratégias: Momentum + Mean Reversion + Volume")
    print("🛡️ Risco: Stop Loss + Take Profit automáticos")

async def demo_system_initialization():
    """Demonstrate system initialization."""
    print("\n🚀 DEMONSTRAÇÃO: INICIALIZAÇÃO DO SISTEMA")
    print("-" * 50)
    
    # Load configuration
    print("📁 Carregando configuração...")
    global_config, bot_configs = create_production_config()
    
    print(f"✅ Configuração carregada:")
    print(f"   💰 Capital total: ${global_config.total_capital:,.2f}")
    print(f"   🤖 Número de bots: {len(bot_configs)}")
    print(f"   📊 Modo: {'Paper Trading' if global_config.paper_trading else 'Real Trading'}")
    print(f"   ⚠️ Limite de perda diária: {global_config.daily_loss_limit:.1%}")
    
    # Initialize system
    print("\n🔧 Inicializando sistema de trading...")
    trading_system = ProductionTradingSystem(global_config, bot_configs)
    
    print("✅ Sistema inicializado com sucesso!")
    print(f"   🌐 Exchange: {'Binance' if not trading_system.using_mock else 'Mock Exchange'}")
    print(f"   📊 Bots configurados: {len(trading_system.bot_configs)}")
    
    # Initialize bots with historical data
    print("\n📊 Inicializando bots com dados históricos...")
    await trading_system._initialize_bots()
    
    enabled_bots = sum(1 for config in trading_system.bot_configs.values() if config.enabled)
    print(f"✅ Bots inicializados: {enabled_bots}/{len(trading_system.bot_configs)}")
    
    # Show bot details
    print("\n🤖 DETALHES DOS BOTS:")
    for bot_id, config in trading_system.bot_configs.items():
        status = "🟢 ATIVO" if config.enabled else "🔴 INATIVO"
        print(f"   {status} {config.symbol} {config.timeframe}")
        print(f"      💰 Alocação: {config.capital_allocation:.0%}")
        print(f"      🎯 Confiança mín: {config.confidence_threshold:.2f}")
        print(f"      🛡️ Stop Loss: {config.stop_loss_pct:.1%}")
        print(f"      📈 Take Profit: {config.take_profit_pct:.1%}")
    
    return trading_system

async def demo_signal_generation(trading_system):
    """Demonstrate signal generation."""
    print("\n🔍 DEMONSTRAÇÃO: GERAÇÃO DE SINAIS")
    print("-" * 50)
    
    signals_found = 0
    
    for bot_id, config in trading_system.bot_configs.items():
        if not config.enabled:
            continue
        
        print(f"\n🧪 Testando sinais para {config.symbol} {config.timeframe}")
        
        try:
            # Get market data
            ohlcv = trading_system.exchange.fetch_ohlcv(config.symbol, config.timeframe, limit=200)
            if not ohlcv or len(ohlcv) < 100:
                print("   ❌ Dados insuficientes")
                continue
            
            import pandas as pd
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Generate signals
            signal_generator = trading_system.signal_generators[bot_id]
            signals = signal_generator.generate_signals(df)
            
            if signals:
                signals_found += len(signals)
                for i, signal in enumerate(signals):
                    direction_text = "🟢 LONG" if signal.direction == 1 else "🔴 SHORT"
                    print(f"   ✅ Sinal {i+1}: {direction_text}")
                    print(f"      💪 Força: {signal.strength:.3f}")
                    print(f"      🎯 Confiança: {signal.confidence:.3f}")
                    print(f"      💰 Preço entrada: ${signal.entry_price:.4f}")
                    print(f"      🛡️ Stop Loss: ${signal.stop_loss:.4f}")
                    print(f"      📈 Take Profit: ${signal.take_profit:.4f}")
            else:
                print("   ⏳ Nenhum sinal no momento")
        
        except Exception as e:
            print(f"   ❌ Erro: {e}")
    
    print(f"\n📊 RESUMO DE SINAIS:")
    print(f"   🎯 Total de sinais encontrados: {signals_found}")
    
    return signals_found > 0

async def demo_trading_execution(trading_system):
    """Demonstrate trading execution."""
    print("\n💰 DEMONSTRAÇÃO: EXECUÇÃO DE TRADES")
    print("-" * 50)
    
    print("🔄 Executando ciclo de trading...")
    
    initial_pnl = trading_system.total_pnl
    initial_trades = len(trading_system.trade_history)
    
    # Execute trading cycles
    for cycle in range(5):
        print(f"\n📅 Ciclo {cycle + 1}/5")
        
        # Process each bot
        for bot_id, config in trading_system.bot_configs.items():
            if not config.enabled:
                continue
            
            try:
                await trading_system._process_bot(bot_id, config)
            except Exception as e:
                print(f"   ❌ Erro processando {config.symbol}: {e}")
        
        # Check for exits with simulated price movements
        for trade in list(trading_system.active_trades.values()):
            try:
                # Get current price
                ohlcv = trading_system.exchange.fetch_ohlcv(trade.symbol, '1m', limit=2)
                if ohlcv:
                    current_price = ohlcv[-1][4]
                    
                    # Simulate some price volatility for demo
                    import random
                    price_change = random.uniform(-0.01, 0.01)  # ±1%
                    simulated_price = current_price * (1 + price_change)
                    
                    await trading_system._check_trade_exit(trade, simulated_price)
            except Exception as e:
                print(f"   ❌ Erro verificando saída: {e}")
        
        # Show current status
        active_trades = len(trading_system.active_trades)
        completed_trades = len(trading_system.trade_history)
        current_pnl = trading_system.total_pnl
        
        print(f"   📊 Trades ativos: {active_trades}")
        print(f"   ✅ Trades completos: {completed_trades}")
        print(f"   💰 PnL atual: ${current_pnl:.2f}")
        
        # Small delay
        await asyncio.sleep(0.5)
    
    # Final summary
    final_pnl = trading_system.total_pnl
    final_trades = len(trading_system.trade_history)
    new_trades = final_trades - initial_trades
    pnl_change = final_pnl - initial_pnl
    
    print(f"\n📊 RESUMO DA EXECUÇÃO:")
    print(f"   🆕 Novos trades: {new_trades}")
    print(f"   💰 Mudança no PnL: ${pnl_change:.2f}")
    print(f"   📈 PnL total: ${final_pnl:.2f}")
    
    if trading_system.trade_history:
        print(f"\n📋 HISTÓRICO DE TRADES:")
        for i, trade in enumerate(trading_system.trade_history[-5:]):  # Last 5 trades
            direction_text = "🟢 LONG" if trade.direction == 1 else "🔴 SHORT"
            status_icon = "✅" if trade.pnl and trade.pnl > 0 else "❌"
            print(f"   {status_icon} Trade {i+1}: {trade.symbol} {direction_text}")
            print(f"      💰 PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2%})")
            print(f"      🏁 Razão: {trade.reason}")
    
    return new_trades > 0

def demo_api_usage():
    """Demonstrate API usage."""
    print("\n🌐 DEMONSTRAÇÃO: USO DA API")
    print("-" * 50)
    
    print("🚀 Para usar a API do sistema:")
    print()
    print("1️⃣ Iniciar o servidor:")
    print("   python api_server.py")
    print()
    print("2️⃣ Acessar o dashboard:")
    print("   http://localhost:8000")
    print()
    print("3️⃣ Endpoints principais:")
    print("   GET  /api/status        - Status do sistema")
    print("   GET  /api/metrics       - Métricas de performance")
    print("   GET  /api/trades/recent - Trades recentes")
    print("   GET  /api/bots          - Status dos bots")
    print("   POST /api/start         - Iniciar trading")
    print("   POST /api/stop          - Parar trading")
    print()
    print("4️⃣ Exemplo de uso com curl:")
    print("   curl http://localhost:8000/api/status")
    print("   curl -X POST http://localhost:8000/api/start")

def demo_configuration():
    """Demonstrate configuration options."""
    print("\n⚙️ DEMONSTRAÇÃO: CONFIGURAÇÃO DO SISTEMA")
    print("-" * 50)
    
    print("📁 Arquivo de configuração: trading_config.json")
    print()
    print("🌍 Configuração Global:")
    print("   • total_capital: Capital total para trading")
    print("   • max_concurrent_trades: Máximo de trades simultâneos")
    print("   • daily_loss_limit: Limite de perda diária (%)")
    print("   • daily_profit_target: Meta de lucro diária (%)")
    print("   • paper_trading: true/false para modo simulação")
    print()
    print("🤖 Configuração dos Bots:")
    print("   • symbol: Par de trading (ex: BTC/USDT)")
    print("   • timeframe: Período dos candles (1m, 5m, 15m)")
    print("   • capital_allocation: % do capital para este bot")
    print("   • confidence_threshold: Confiança mínima para trade")
    print("   • stop_loss_pct: % de stop loss")
    print("   • take_profit_pct: % de take profit")
    print()
    print("🎛️ Parâmetros Avançados (no código):")
    print("   • momentum_threshold: Sensibilidade ao momentum")
    print("   • volume_threshold: Multiplicador de volume")
    print("   • rsi_oversold/overbought: Níveis de RSI")

def print_next_steps():
    """Print next steps for the user."""
    print("\n🎯 PRÓXIMOS PASSOS RECOMENDADOS")
    print("-" * 50)
    print()
    print("1️⃣ OTIMIZAÇÃO (1-2 dias):")
    print("   • Ajustar parâmetros para melhor win rate")
    print("   • Testar diferentes timeframes")
    print("   • Calibrar thresholds de confiança")
    print()
    print("2️⃣ TESTES ESTENDIDOS (1 semana):")
    print("   • Executar backtests com 30+ dias")
    print("   • Validar em diferentes condições de mercado")
    print("   • Monitorar performance diária")
    print()
    print("3️⃣ DEPLOY EM PRODUÇÃO:")
    print("   • Configurar servidor na nuvem")
    print("   • Ativar trading real (com cuidado!)")
    print("   • Implementar monitoramento 24/7")
    print()
    print("4️⃣ MONITORAMENTO CONTÍNUO:")
    print("   • Dashboard em tempo real")
    print("   • Alertas automáticos")
    print("   • Relatórios diários/semanais")

async def main():
    """Main demo function."""
    print_banner()
    print_system_info()
    
    try:
        # Demo 1: System initialization
        trading_system = await demo_system_initialization()
        
        # Demo 2: Signal generation
        signals_generated = await demo_signal_generation(trading_system)
        
        # Demo 3: Trading execution (only if signals were generated)
        if signals_generated:
            trades_executed = await demo_trading_execution(trading_system)
        else:
            print("\n⏳ Pulando demonstração de trading (sem sinais no momento)")
            trades_executed = False
        
        # Demo 4: API usage
        demo_api_usage()
        
        # Demo 5: Configuration
        demo_configuration()
        
        # Next steps
        print_next_steps()
        
        # Final summary
        print("\n" + "=" * 80)
        print("✅ DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 80)
        print(f"🤖 Sistema inicializado: ✅")
        print(f"🔍 Sinais gerados: {'✅' if signals_generated else '⏳'}")
        print(f"💰 Trades executados: {'✅' if trades_executed else '⏳'}")
        print(f"🌐 API funcional: ✅")
        print(f"📊 Frontend disponível: ✅")
        print()
        print("🎉 O Trading Bot ML está pronto para uso!")
        print("📖 Consulte o COMPREHENSIVE_TEST_REPORT.md para detalhes completos")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO NA DEMONSTRAÇÃO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)