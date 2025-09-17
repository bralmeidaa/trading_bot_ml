#!/usr/bin/env python3
"""
Complete Deployment Validation Script
Valida todos os componentes do sistema antes do deploy
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class DeploymentValidator:
    """Validador completo de deployment."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'components': {},
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
    
    def log_result(self, component: str, status: str, details: str = "", error: str = ""):
        """Log resultado de validação."""
        self.results['components'][component] = {
            'status': status,
            'details': details,
            'error': error
        }
        
        if status == 'FAIL':
            self.results['errors'].append(f"{component}: {error}")
        elif status == 'WARN':
            self.results['warnings'].append(f"{component}: {details}")
        
        print(f"{'✅' if status == 'PASS' else '⚠️' if status == 'WARN' else '❌'} {component}: {details or error}")
    
    def validate_dependencies(self):
        """Valida dependências Python."""
        print("\n🔍 VALIDANDO DEPENDÊNCIAS...")
        
        critical_deps = [
            'fastapi', 'uvicorn', 'pandas', 'numpy', 'sklearn',
            'xgboost', 'ta', 'ccxt', 'binance', 'loguru', 'aiohttp'
        ]
        
        missing_deps = []
        for dep in critical_deps:
            try:
                __import__(dep.replace('-', '_'))
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            self.log_result(
                'dependencies', 'FAIL',
                error=f"Missing dependencies: {', '.join(missing_deps)}"
            )
        else:
            self.log_result(
                'dependencies', 'PASS',
                details=f"All {len(critical_deps)} critical dependencies available"
            )
    
    def validate_core_imports(self):
        """Valida imports dos sistemas principais."""
        print("\n🔍 VALIDANDO IMPORTS PRINCIPAIS...")
        
        core_systems = {
            'ProductionTradingSystem': 'production_trading_system',
            'RegimeIntegratedTradingSystem': 'regime_integrated_trading_system',
            'UltimateTradingSystem': 'ultimate_trading_system',
            'MLEnsembleSystem': 'advanced_ml.ml_ensemble_system',
            'ContinuousLearningSystem': 'advanced_ml.continuous_learning_system',
            'APIServer': 'api_server'
        }
        
        for system_name, module_path in core_systems.items():
            try:
                if '.' in module_path:
                    module_parts = module_path.split('.')
                    module = __import__(module_path, fromlist=[module_parts[-1]])
                    getattr(module, system_name)
                else:
                    module = __import__(module_path)
                    if hasattr(module, system_name):
                        getattr(module, system_name)
                    else:
                        # For api_server, just check if it imports
                        pass
                
                self.log_result(
                    f'import_{system_name}', 'PASS',
                    details=f"{system_name} importado com sucesso"
                )
            except Exception as e:
                self.log_result(
                    f'import_{system_name}', 'FAIL',
                    error=f"Erro ao importar {system_name}: {str(e)}"
                )
    
    def validate_configuration_files(self):
        """Valida arquivos de configuração."""
        print("\n🔍 VALIDANDO ARQUIVOS DE CONFIGURAÇÃO...")
        
        config_files = {
            'trading_config.json': 'Configuração principal do sistema',
            'regime_strategy_config.json': 'Configuração de estratégias por regime',
            'ml_ensemble_config.json': 'Configuração do ensemble ML',
            'ultimate_config.json': 'Configuração do sistema ultimate',
            'requirements.txt': 'Dependências Python',
            'docker-compose.yml': 'Configuração Docker',
            'Dockerfile': 'Imagem Docker'
        }
        
        for file_path, description in config_files.items():
            if os.path.exists(file_path):
                try:
                    if file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            json.load(f)  # Validate JSON
                    
                    self.log_result(
                        f'config_{file_path}', 'PASS',
                        details=f"{description} válido"
                    )
                except Exception as e:
                    self.log_result(
                        f'config_{file_path}', 'FAIL',
                        error=f"Erro no arquivo {file_path}: {str(e)}"
                    )
            else:
                self.log_result(
                    f'config_{file_path}', 'WARN',
                    details=f"Arquivo {file_path} não encontrado"
                )
    
    def validate_frontend_build(self):
        """Valida build do frontend."""
        print("\n🔍 VALIDANDO FRONTEND...")
        
        frontend_path = Path('frontend_react')
        
        if not frontend_path.exists():
            self.log_result(
                'frontend_structure', 'FAIL',
                error="Diretório frontend_react não encontrado"
            )
            return
        
        # Check package.json
        package_json = frontend_path / 'package.json'
        if package_json.exists():
            self.log_result(
                'frontend_package', 'PASS',
                details="package.json encontrado"
            )
        else:
            self.log_result(
                'frontend_package', 'FAIL',
                error="package.json não encontrado"
            )
            return
        
        # Check if build exists
        dist_path = frontend_path / 'dist'
        if dist_path.exists():
            self.log_result(
                'frontend_build', 'PASS',
                details="Build do frontend encontrado"
            )
        else:
            self.log_result(
                'frontend_build', 'WARN',
                details="Build do frontend não encontrado - execute npm run build"
            )
    
    def validate_specialized_bots(self):
        """Valida bots especializados."""
        print("\n🔍 VALIDANDO BOTS ESPECIALIZADOS...")
        
        bots_path = Path('specialized_bots')
        
        if not bots_path.exists():
            self.log_result(
                'specialized_bots_dir', 'FAIL',
                error="Diretório specialized_bots não encontrado"
            )
            return
        
        expected_bots = [
            'base_specialized_bot.py',
            'btc_trend_bot.py',
            'eth_momentum_bot.py'
        ]
        
        for bot_file in expected_bots:
            bot_path = bots_path / bot_file
            if bot_path.exists():
                self.log_result(
                    f'bot_{bot_file}', 'PASS',
                    details=f"Bot {bot_file} encontrado"
                )
            else:
                self.log_result(
                    f'bot_{bot_file}', 'FAIL',
                    error=f"Bot {bot_file} não encontrado"
                )
    
    def validate_advanced_ml(self):
        """Valida sistema ML avançado."""
        print("\n🔍 VALIDANDO SISTEMA ML AVANÇADO...")
        
        ml_path = Path('advanced_ml')
        
        if not ml_path.exists():
            self.log_result(
                'advanced_ml_dir', 'FAIL',
                error="Diretório advanced_ml não encontrado"
            )
            return
        
        expected_files = [
            'feature_engineering.py',
            'ml_ensemble_system.py',
            'continuous_learning_system.py'
        ]
        
        for ml_file in expected_files:
            file_path = ml_path / ml_file
            if file_path.exists():
                self.log_result(
                    f'ml_{ml_file}', 'PASS',
                    details=f"Arquivo ML {ml_file} encontrado"
                )
            else:
                self.log_result(
                    f'ml_{ml_file}', 'FAIL',
                    error=f"Arquivo ML {ml_file} não encontrado"
                )
    
    def validate_docker_setup(self):
        """Valida configuração Docker."""
        print("\n🔍 VALIDANDO CONFIGURAÇÃO DOCKER...")
        
        # Check Dockerfile
        if os.path.exists('Dockerfile'):
            self.log_result(
                'dockerfile', 'PASS',
                details="Dockerfile encontrado"
            )
        else:
            self.log_result(
                'dockerfile', 'FAIL',
                error="Dockerfile não encontrado"
            )
        
        # Check docker-compose.yml
        if os.path.exists('docker-compose.yml'):
            self.log_result(
                'docker_compose', 'PASS',
                details="docker-compose.yml encontrado"
            )
        else:
            self.log_result(
                'docker_compose', 'FAIL',
                error="docker-compose.yml não encontrado"
            )
        
        # Check nginx.conf
        if os.path.exists('nginx.conf'):
            self.log_result(
                'nginx_config', 'PASS',
                details="nginx.conf encontrado"
            )
        else:
            self.log_result(
                'nginx_config', 'WARN',
                details="nginx.conf não encontrado - opcional para desenvolvimento"
            )
    
    async def validate_api_functionality(self):
        """Valida funcionalidade básica da API."""
        print("\n🔍 VALIDANDO FUNCIONALIDADE DA API...")
        
        try:
            from api_server import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            
            # Test health endpoint
            response = client.get("/api/health")
            if response.status_code == 200:
                self.log_result(
                    'api_health', 'PASS',
                    details="Endpoint /api/health funcionando"
                )
            else:
                self.log_result(
                    'api_health', 'FAIL',
                    error=f"Endpoint /api/health retornou {response.status_code}"
                )
            
            # Test system status endpoint
            response = client.get("/api/system/status")
            if response.status_code == 200:
                self.log_result(
                    'api_status', 'PASS',
                    details="Endpoint /api/system/status funcionando"
                )
            else:
                self.log_result(
                    'api_status', 'WARN',
                    details=f"Endpoint /api/system/status retornou {response.status_code}"
                )
                
        except Exception as e:
            self.log_result(
                'api_functionality', 'FAIL',
                error=f"Erro ao testar API: {str(e)}"
            )
    
    def validate_file_permissions(self):
        """Valida permissões de arquivos."""
        print("\n🔍 VALIDANDO PERMISSÕES...")
        
        critical_files = [
            'api_server.py',
            'production_trading_system.py',
            'ultimate_trading_system.py'
        ]
        
        for file_path in critical_files:
            if os.path.exists(file_path):
                if os.access(file_path, os.R_OK):
                    self.log_result(
                        f'permissions_{file_path}', 'PASS',
                        details=f"Arquivo {file_path} legível"
                    )
                else:
                    self.log_result(
                        f'permissions_{file_path}', 'FAIL',
                        error=f"Arquivo {file_path} não legível"
                    )
            else:
                self.log_result(
                    f'permissions_{file_path}', 'FAIL',
                    error=f"Arquivo {file_path} não encontrado"
                )
    
    def generate_recommendations(self):
        """Gera recomendações baseadas na validação."""
        print("\n💡 GERANDO RECOMENDAÇÕES...")
        
        # Check for common issues
        if any('frontend_build' in comp and self.results['components'][comp]['status'] == 'WARN' 
               for comp in self.results['components']):
            self.results['recommendations'].append(
                "Execute 'cd frontend_react && npm run build' para gerar o build do frontend"
            )
        
        if any('FAIL' in self.results['components'][comp]['status'] 
               for comp in self.results['components']):
            self.results['recommendations'].append(
                "Corrija os erros críticos antes do deploy em produção"
            )
        
        if len(self.results['warnings']) > 0:
            self.results['recommendations'].append(
                "Revise os warnings - alguns podem afetar funcionalidades específicas"
            )
        
        # Environment-specific recommendations
        self.results['recommendations'].extend([
            "Configure as variáveis de ambiente BINANCE_API_KEY e BINANCE_API_SECRET",
            "Para produção, use HTTPS e configure certificados SSL",
            "Configure backup automático dos logs e dados de trading",
            "Monitore o uso de CPU e memória em produção",
            "Configure alertas para falhas do sistema"
        ])
    
    def calculate_overall_status(self):
        """Calcula status geral do deployment."""
        
        total_components = len(self.results['components'])
        passed_components = sum(1 for comp in self.results['components'].values() 
                               if comp['status'] == 'PASS')
        failed_components = sum(1 for comp in self.results['components'].values() 
                               if comp['status'] == 'FAIL')
        
        if failed_components == 0:
            if len(self.results['warnings']) == 0:
                self.results['overall_status'] = 'READY_FOR_PRODUCTION'
            else:
                self.results['overall_status'] = 'READY_WITH_WARNINGS'
        elif failed_components <= 2:
            self.results['overall_status'] = 'NEEDS_MINOR_FIXES'
        else:
            self.results['overall_status'] = 'NEEDS_MAJOR_FIXES'
        
        self.results['summary'] = {
            'total_components': total_components,
            'passed': passed_components,
            'warnings': len([c for c in self.results['components'].values() if c['status'] == 'WARN']),
            'failed': failed_components,
            'pass_rate': (passed_components / total_components * 100) if total_components > 0 else 0
        }
    
    async def run_complete_validation(self):
        """Executa validação completa."""
        print("🚀 INICIANDO VALIDAÇÃO COMPLETA DO DEPLOYMENT")
        print("=" * 60)
        
        # Run all validations
        self.validate_dependencies()
        self.validate_core_imports()
        self.validate_configuration_files()
        self.validate_frontend_build()
        self.validate_specialized_bots()
        self.validate_advanced_ml()
        self.validate_docker_setup()
        await self.validate_api_functionality()
        self.validate_file_permissions()
        
        # Generate final report
        self.generate_recommendations()
        self.calculate_overall_status()
        
        # Print summary
        self.print_summary()
        
        # Save results
        with open('deployment_validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return self.results
    
    def print_summary(self):
        """Imprime resumo da validação."""
        print("\n" + "=" * 60)
        print("📊 RESUMO DA VALIDAÇÃO")
        print("=" * 60)
        
        summary = self.results['summary']
        status = self.results['overall_status']
        
        print(f"Status Geral: {self.get_status_emoji(status)} {status}")
        print(f"Componentes Testados: {summary['total_components']}")
        print(f"✅ Passou: {summary['passed']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        print(f"❌ Falhou: {summary['failed']}")
        print(f"Taxa de Sucesso: {summary['pass_rate']:.1f}%")
        
        if self.results['errors']:
            print(f"\n❌ ERROS CRÍTICOS ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                print(f"  • {error}")
        
        if self.results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                print(f"  • {warning}")
        
        if self.results['recommendations']:
            print(f"\n💡 RECOMENDAÇÕES ({len(self.results['recommendations'])}):")
            for rec in self.results['recommendations']:
                print(f"  • {rec}")
        
        print(f"\n📄 Relatório completo salvo em: deployment_validation_results.json")
        
        # Final verdict
        print("\n" + "=" * 60)
        if status == 'READY_FOR_PRODUCTION':
            print("🎉 SISTEMA PRONTO PARA PRODUÇÃO!")
        elif status == 'READY_WITH_WARNINGS':
            print("✅ SISTEMA PRONTO (com warnings menores)")
        elif status == 'NEEDS_MINOR_FIXES':
            print("🔧 SISTEMA PRECISA DE CORREÇÕES MENORES")
        else:
            print("⚠️  SISTEMA PRECISA DE CORREÇÕES IMPORTANTES")
        print("=" * 60)
    
    def get_status_emoji(self, status):
        """Retorna emoji para o status."""
        emojis = {
            'READY_FOR_PRODUCTION': '🎉',
            'READY_WITH_WARNINGS': '✅',
            'NEEDS_MINOR_FIXES': '🔧',
            'NEEDS_MAJOR_FIXES': '⚠️'
        }
        return emojis.get(status, '❓')

async def main():
    """Função principal."""
    validator = DeploymentValidator()
    results = await validator.run_complete_validation()
    
    # Exit with appropriate code
    if results['overall_status'] in ['READY_FOR_PRODUCTION', 'READY_WITH_WARNINGS']:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())