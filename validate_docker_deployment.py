#!/usr/bin/env python3
"""
Validate Docker and Nginx configuration for the optimized trading system.
Ensures all deployment configurations are compatible with the optimizations.
"""

import os
import json
import yaml
import subprocess
import sys
from pathlib import Path

def print_banner():
    """Print validation banner."""
    print("=" * 80)
    print("🐳 VALIDAÇÃO DOCKER & NGINX - SISTEMA OTIMIZADO")
    print("=" * 80)
    print("🔍 Verificando compatibilidade das otimizações com deploy")
    print("📦 Docker + Nginx + Sistema Otimizado")
    print("=" * 80)

def check_docker_files():
    """Check Docker configuration files."""
    print("\n🐳 VALIDANDO CONFIGURAÇÃO DOCKER")
    print("-" * 50)
    
    issues = []
    
    # Check Dockerfile
    dockerfile = Path("Dockerfile")
    if dockerfile.exists():
        print("✅ Dockerfile encontrado")
        
        with open(dockerfile, 'r') as f:
            content = f.read()
        
        # Check if it builds frontend
        if "npm run build" in content:
            print("✅ Build do frontend configurado")
        else:
            issues.append("❌ Build do frontend não configurado no Dockerfile")
        
        # Check if it copies optimized config
        if "COPY . ." in content or "COPY --from=builder /app /app" in content:
            print("✅ Cópia de arquivos configurada")
        else:
            issues.append("❌ Cópia de arquivos não configurada")
        
        # Check port configuration
        if "12000" in content:
            print("✅ Porta 12000 configurada corretamente")
        else:
            issues.append("❌ Porta incorreta no Dockerfile")
    else:
        issues.append("❌ Dockerfile não encontrado")
    
    # Check docker-compose.yml
    compose_file = Path("docker-compose.yml")
    if compose_file.exists():
        print("✅ docker-compose.yml encontrado")
        
        try:
            with open(compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            # Check trading-bot service
            if 'services' in compose_config and 'trading-bot' in compose_config['services']:
                service = compose_config['services']['trading-bot']
                
                # Check port mapping
                if 'ports' in service and '12000:12000' in service['ports']:
                    print("✅ Mapeamento de porta correto")
                else:
                    issues.append("❌ Mapeamento de porta incorreto")
                
                # Check environment variables
                if 'environment' in service:
                    print("✅ Variáveis de ambiente configuradas")
                else:
                    issues.append("⚠️ Variáveis de ambiente não configuradas")
                
                # Check health check
                if 'healthcheck' in service:
                    print("✅ Health check configurado")
                else:
                    issues.append("⚠️ Health check não configurado")
            else:
                issues.append("❌ Serviço trading-bot não encontrado")
        
        except yaml.YAMLError as e:
            issues.append(f"❌ Erro ao ler docker-compose.yml: {e}")
    else:
        issues.append("❌ docker-compose.yml não encontrado")
    
    # Check .dockerignore
    dockerignore = Path(".dockerignore")
    if dockerignore.exists():
        print("✅ .dockerignore encontrado")
        
        with open(dockerignore, 'r') as f:
            ignore_content = f.read()
        
        # Check if it ignores unnecessary files
        if "node_modules" in ignore_content and "*.log" in ignore_content:
            print("✅ Arquivos desnecessários ignorados")
        else:
            issues.append("⚠️ .dockerignore pode ser otimizado")
    else:
        issues.append("⚠️ .dockerignore não encontrado")
    
    return issues

def check_nginx_config():
    """Check Nginx configuration."""
    print("\n🌐 VALIDANDO CONFIGURAÇÃO NGINX")
    print("-" * 50)
    
    issues = []
    
    nginx_conf = Path("nginx.conf")
    if nginx_conf.exists():
        print("✅ nginx.conf encontrado")
        
        with open(nginx_conf, 'r') as f:
            content = f.read()
        
        # Check upstream configuration
        if "upstream trading_bot" in content and "trading-bot:12000" in content:
            print("✅ Upstream configurado corretamente")
        else:
            issues.append("❌ Upstream não configurado corretamente")
        
        # Check SSL configuration
        if "ssl_certificate" in content and "ssl_certificate_key" in content:
            print("✅ SSL configurado")
        else:
            issues.append("⚠️ SSL não configurado")
        
        # Check API routing
        if "location /api/" in content:
            print("✅ Roteamento de API configurado")
        else:
            issues.append("❌ Roteamento de API não configurado")
        
        # Check rate limiting
        if "limit_req_zone" in content:
            print("✅ Rate limiting configurado")
        else:
            issues.append("⚠️ Rate limiting não configurado")
        
        # Check security headers
        if "X-Frame-Options" in content and "X-Content-Type-Options" in content:
            print("✅ Headers de segurança configurados")
        else:
            issues.append("⚠️ Headers de segurança não configurados")
    else:
        issues.append("❌ nginx.conf não encontrado")
    
    return issues

def check_api_server_compatibility():
    """Check if API server is compatible with Docker deployment."""
    print("\n🔧 VALIDANDO COMPATIBILIDADE DO API SERVER")
    print("-" * 50)
    
    issues = []
    
    api_server = Path("api_server.py")
    if api_server.exists():
        print("✅ api_server.py encontrado")
        
        with open(api_server, 'r') as f:
            content = f.read()
        
        # Check port configuration
        if "port=12000" in content:
            print("✅ Porta 12000 configurada")
        else:
            issues.append("❌ Porta incorreta no api_server.py")
        
        # Check host binding
        if 'host="0.0.0.0"' in content:
            print("✅ Host binding configurado para Docker")
        else:
            issues.append("❌ Host binding não configurado para Docker")
        
        # Check static files serving
        if "StaticFiles" in content and "frontend_react/dist" in content:
            print("✅ Servir arquivos estáticos configurado")
        else:
            issues.append("❌ Servir arquivos estáticos não configurado")
        
        # Check health endpoint
        if "/api/health" in content or "/health" in content:
            print("✅ Health endpoint disponível")
        else:
            issues.append("⚠️ Health endpoint não encontrado")
    else:
        issues.append("❌ api_server.py não encontrado")
    
    return issues

def check_optimized_config_compatibility():
    """Check if optimized configuration is compatible with Docker."""
    print("\n⚙️ VALIDANDO CONFIGURAÇÃO OTIMIZADA")
    print("-" * 50)
    
    issues = []
    
    # Check optimized config file
    config_file = Path("trading_config.json")
    if config_file.exists():
        print("✅ trading_config.json encontrado")
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Check if it has the optimized structure
            if 'global_config' in config and 'bot_configs' in config:
                print("✅ Estrutura de configuração otimizada")
                
                # Check bot count
                bot_count = len(config['bot_configs'])
                if bot_count >= 3:
                    print(f"✅ {bot_count} bots configurados")
                else:
                    issues.append(f"⚠️ Apenas {bot_count} bots configurados")
                
                # Check capital allocation
                global_config = config['global_config']
                if 'total_capital' in global_config:
                    capital = global_config['total_capital']
                    print(f"✅ Capital configurado: ${capital:,.2f}")
                else:
                    issues.append("❌ Capital não configurado")
                
                # Check paper trading
                if global_config.get('paper_trading', False):
                    print("✅ Paper trading ativo (seguro)")
                else:
                    issues.append("⚠️ Paper trading não ativo")
            else:
                issues.append("❌ Estrutura de configuração inválida")
        
        except json.JSONDecodeError as e:
            issues.append(f"❌ Erro ao ler configuração: {e}")
    else:
        issues.append("❌ trading_config.json não encontrado")
    
    return issues

def check_frontend_build():
    """Check if frontend build is ready for Docker."""
    print("\n📱 VALIDANDO BUILD DO FRONTEND")
    print("-" * 50)
    
    issues = []
    
    # Check if frontend directory exists
    frontend_dir = Path("frontend_react")
    if frontend_dir.exists():
        print("✅ Diretório frontend_react encontrado")
        
        # Check package.json
        package_json = frontend_dir / "package.json"
        if package_json.exists():
            print("✅ package.json encontrado")
            
            try:
                with open(package_json, 'r') as f:
                    package_config = json.load(f)
                
                # Check build script
                if 'scripts' in package_config and 'build' in package_config['scripts']:
                    print("✅ Script de build configurado")
                else:
                    issues.append("❌ Script de build não configurado")
                
                # Check dependencies
                if 'dependencies' in package_config:
                    deps = package_config['dependencies']
                    if 'react' in deps and 'vite' in deps:
                        print("✅ Dependências principais encontradas")
                    else:
                        issues.append("⚠️ Dependências principais não encontradas")
            
            except json.JSONDecodeError as e:
                issues.append(f"❌ Erro ao ler package.json: {e}")
        else:
            issues.append("❌ package.json não encontrado")
        
        # Check if build directory exists (optional)
        dist_dir = frontend_dir / "dist"
        if dist_dir.exists():
            print("✅ Build do frontend já existe")
        else:
            print("⚠️ Build do frontend não existe (será criado no Docker)")
        
        # Check Vite config
        vite_config = frontend_dir / "vite.config.js"
        if vite_config.exists():
            print("✅ Configuração do Vite encontrada")
        else:
            issues.append("⚠️ vite.config.js não encontrado")
    else:
        issues.append("❌ Diretório frontend_react não encontrado")
    
    return issues

def test_docker_build():
    """Test if Docker image can be built."""
    print("\n🔨 TESTANDO BUILD DOCKER")
    print("-" * 50)
    
    issues = []
    
    # Check if Docker is available
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Docker disponível")
            print(f"   {result.stdout.strip()}")
        else:
            issues.append("❌ Docker não disponível")
            return issues
    except (subprocess.TimeoutExpired, FileNotFoundError):
        issues.append("❌ Docker não instalado ou não disponível")
        return issues
    
    # Test build (dry run)
    print("🔍 Testando sintaxe do Dockerfile...")
    try:
        # Just parse the Dockerfile, don't actually build
        result = subprocess.run(['docker', 'build', '--dry-run', '.'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Dockerfile válido")
        else:
            issues.append(f"❌ Erro no Dockerfile: {result.stderr}")
    except subprocess.TimeoutExpired:
        issues.append("⚠️ Timeout ao testar Dockerfile")
    except Exception as e:
        issues.append(f"⚠️ Não foi possível testar build: {e}")
    
    return issues

def generate_deployment_guide():
    """Generate deployment guide."""
    print("\n📖 GUIA DE DEPLOY")
    print("-" * 50)
    
    guide = """
🚀 DEPLOY DO SISTEMA OTIMIZADO

1️⃣ PREPARAÇÃO:
   • Certifique-se que Docker está instalado
   • Configure variáveis de ambiente (BINANCE_API_KEY, etc.)
   • Prepare certificados SSL (se usar HTTPS)

2️⃣ BUILD DA IMAGEM:
   docker build -t trading-bot-ml:optimized .

3️⃣ DEPLOY COM DOCKER COMPOSE:
   docker-compose up -d

4️⃣ VERIFICAÇÃO:
   • http://localhost:12000 (aplicação)
   • http://localhost:12000/api/health (health check)
   • docker logs trading-bot-master (logs)

5️⃣ MONITORAMENTO:
   • docker stats (recursos)
   • docker logs -f trading-bot-master (logs em tempo real)
   • curl http://localhost:12000/api/status (status da API)

🔧 COMANDOS ÚTEIS:
   • docker-compose logs -f trading-bot
   • docker-compose restart trading-bot
   • docker-compose down && docker-compose up -d

⚠️ IMPORTANTE:
   • Sistema está em paper trading por padrão
   • Configure SSL para produção
   • Monitore logs regularmente
   • Faça backup da configuração
"""
    
    print(guide)

def main():
    """Main validation function."""
    print_banner()
    
    all_issues = []
    
    # Run all validations
    all_issues.extend(check_docker_files())
    all_issues.extend(check_nginx_config())
    all_issues.extend(check_api_server_compatibility())
    all_issues.extend(check_optimized_config_compatibility())
    all_issues.extend(check_frontend_build())
    all_issues.extend(test_docker_build())
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 RESUMO DA VALIDAÇÃO")
    print("=" * 80)
    
    if not all_issues:
        print("🎉 TODAS AS VALIDAÇÕES PASSARAM!")
        print("✅ Sistema otimizado compatível com Docker/Nginx")
        print("✅ Configuração de deploy está correta")
        print("✅ Pronto para deploy em produção")
    else:
        print(f"⚠️ ENCONTRADOS {len(all_issues)} PROBLEMAS:")
        for issue in all_issues:
            print(f"   {issue}")
        
        # Categorize issues
        critical = [i for i in all_issues if "❌" in i]
        warnings = [i for i in all_issues if "⚠️" in i]
        
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   🔴 Críticos: {len(critical)}")
        print(f"   🟡 Avisos: {len(warnings)}")
        
        if critical:
            print(f"\n🚨 PROBLEMAS CRÍTICOS DEVEM SER CORRIGIDOS ANTES DO DEPLOY")
        else:
            print(f"\n✅ Nenhum problema crítico - deploy pode prosseguir")
    
    # Generate deployment guide
    generate_deployment_guide()
    
    # Final status
    critical_issues = len([i for i in all_issues if "❌" in i])
    
    print(f"\n🏁 STATUS FINAL: {'✅ APROVADO' if critical_issues == 0 else '❌ REQUER CORREÇÕES'}")
    
    return critical_issues == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)