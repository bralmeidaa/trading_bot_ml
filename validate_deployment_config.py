#!/usr/bin/env python3
"""
Validate deployment configuration without requiring Docker to be running.
Focus on configuration files and compatibility.
"""

import os
import json
import yaml
import sys
from pathlib import Path

def print_banner():
    """Print validation banner."""
    print("=" * 80)
    print("🔧 VALIDAÇÃO DE CONFIGURAÇÃO DE DEPLOY")
    print("=" * 80)
    print("📋 Verificando arquivos de configuração Docker/Nginx")
    print("🎯 Compatibilidade com sistema otimizado")
    print("=" * 80)

def validate_dockerfile():
    """Validate Dockerfile configuration."""
    print("\n🐳 VALIDANDO DOCKERFILE")
    print("-" * 50)
    
    issues = []
    dockerfile = Path("Dockerfile")
    
    if not dockerfile.exists():
        issues.append("❌ Dockerfile não encontrado")
        return issues
    
    print("✅ Dockerfile encontrado")
    
    with open(dockerfile, 'r') as f:
        content = f.read()
    
    # Check multi-stage build
    if "FROM python:3.12 AS builder" in content:
        print("✅ Multi-stage build configurado")
    else:
        issues.append("⚠️ Multi-stage build não encontrado")
    
    # Check Node.js installation
    if "nodejs npm" in content:
        print("✅ Node.js instalação configurada")
    else:
        issues.append("❌ Node.js não configurado")
    
    # Check frontend build
    if "npm run build" in content:
        print("✅ Build do frontend configurado")
    else:
        issues.append("❌ Build do frontend não configurado")
    
    # Check port exposure
    if "EXPOSE 12000" in content:
        print("✅ Porta 12000 exposta")
    else:
        issues.append("❌ Porta não exposta corretamente")
    
    # Check health check
    if "HEALTHCHECK" in content:
        print("✅ Health check configurado")
    else:
        issues.append("⚠️ Health check não configurado")
    
    # Check user security
    if "useradd" in content and "USER tradingbot" in content:
        print("✅ Usuário não-root configurado")
    else:
        issues.append("⚠️ Usuário não-root não configurado")
    
    # Check Python path
    if "PYTHONPATH=/app" in content:
        print("✅ PYTHONPATH configurado")
    else:
        issues.append("⚠️ PYTHONPATH não configurado")
    
    return issues

def validate_docker_compose():
    """Validate docker-compose.yml configuration."""
    print("\n📦 VALIDANDO DOCKER-COMPOSE")
    print("-" * 50)
    
    issues = []
    compose_file = Path("docker-compose.yml")
    
    if not compose_file.exists():
        issues.append("❌ docker-compose.yml não encontrado")
        return issues
    
    print("✅ docker-compose.yml encontrado")
    
    try:
        with open(compose_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        issues.append(f"❌ Erro ao ler YAML: {e}")
        return issues
    
    # Check services
    if 'services' not in config:
        issues.append("❌ Seção 'services' não encontrada")
        return issues
    
    services = config['services']
    
    # Check trading-bot service
    if 'trading-bot' in services:
        print("✅ Serviço trading-bot encontrado")
        
        bot_service = services['trading-bot']
        
        # Check image
        if 'image' in bot_service:
            print(f"✅ Imagem configurada: {bot_service['image']}")
        else:
            issues.append("⚠️ Imagem não especificada")
        
        # Check ports
        if 'ports' in bot_service:
            ports = bot_service['ports']
            if '12000:12000' in ports:
                print("✅ Porta 12000 mapeada corretamente")
            else:
                issues.append("❌ Porta não mapeada corretamente")
        else:
            issues.append("❌ Portas não configuradas")
        
        # Check environment
        if 'environment' in bot_service:
            env = bot_service['environment']
            if any('BINANCE_API_KEY' in str(var) for var in env):
                print("✅ Variáveis de ambiente Binance configuradas")
            else:
                issues.append("⚠️ Variáveis Binance não configuradas")
        
        # Check health check
        if 'healthcheck' in bot_service:
            print("✅ Health check configurado")
        else:
            issues.append("⚠️ Health check não configurado")
        
        # Check restart policy
        if 'restart' in bot_service:
            print(f"✅ Política de restart: {bot_service['restart']}")
        else:
            issues.append("⚠️ Política de restart não configurada")
    else:
        issues.append("❌ Serviço trading-bot não encontrado")
    
    # Check nginx service
    if 'nginx' in services:
        print("✅ Serviço Nginx encontrado")
        
        nginx_service = services['nginx']
        
        # Check ports
        if 'ports' in nginx_service:
            ports = nginx_service['ports']
            if '80:80' in ports and '443:443' in ports:
                print("✅ Portas HTTP/HTTPS configuradas")
            else:
                issues.append("⚠️ Portas HTTP/HTTPS não configuradas")
        
        # Check volumes
        if 'volumes' in nginx_service:
            volumes = nginx_service['volumes']
            if any('nginx.conf' in str(vol) for vol in volumes):
                print("✅ Configuração Nginx montada")
            else:
                issues.append("❌ Configuração Nginx não montada")
    else:
        print("⚠️ Serviço Nginx não encontrado (opcional)")
    
    # Check networks
    if 'networks' in config:
        print("✅ Redes configuradas")
    else:
        issues.append("⚠️ Redes não configuradas")
    
    return issues

def validate_nginx_config():
    """Validate nginx.conf configuration."""
    print("\n🌐 VALIDANDO NGINX.CONF")
    print("-" * 50)
    
    issues = []
    nginx_conf = Path("nginx.conf")
    
    if not nginx_conf.exists():
        issues.append("❌ nginx.conf não encontrado")
        return issues
    
    print("✅ nginx.conf encontrado")
    
    with open(nginx_conf, 'r') as f:
        content = f.read()
    
    # Check upstream
    if "upstream trading_bot" in content:
        print("✅ Upstream trading_bot configurado")
        
        if "trading-bot:12000" in content:
            print("✅ Upstream aponta para porta correta")
        else:
            issues.append("❌ Upstream não aponta para porta 12000")
    else:
        issues.append("❌ Upstream não configurado")
    
    # Check SSL
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
    security_headers = [
        "X-Frame-Options",
        "X-Content-Type-Options", 
        "X-XSS-Protection",
        "Strict-Transport-Security"
    ]
    
    missing_headers = []
    for header in security_headers:
        if header not in content:
            missing_headers.append(header)
    
    if not missing_headers:
        print("✅ Headers de segurança configurados")
    else:
        issues.append(f"⚠️ Headers faltando: {', '.join(missing_headers)}")
    
    # Check proxy settings
    if "proxy_pass http://trading_bot" in content:
        print("✅ Proxy pass configurado")
    else:
        issues.append("❌ Proxy pass não configurado")
    
    return issues

def validate_api_server():
    """Validate API server configuration for Docker."""
    print("\n🔧 VALIDANDO API SERVER")
    print("-" * 50)
    
    issues = []
    api_server = Path("api_server.py")
    
    if not api_server.exists():
        issues.append("❌ api_server.py não encontrado")
        return issues
    
    print("✅ api_server.py encontrado")
    
    with open(api_server, 'r') as f:
        content = f.read()
    
    # Check port
    if "port=12000" in content:
        print("✅ Porta 12000 configurada")
    else:
        issues.append("❌ Porta não configurada para 12000")
    
    # Check host binding
    if 'host="0.0.0.0"' in content:
        print("✅ Host binding para Docker configurado")
    else:
        issues.append("❌ Host não configurado para 0.0.0.0")
    
    # Check static files
    if "StaticFiles" in content:
        print("✅ Servir arquivos estáticos configurado")
        
        if "frontend_react/dist" in content:
            print("✅ Diretório de build do frontend configurado")
        else:
            issues.append("❌ Diretório de build não configurado")
    else:
        issues.append("❌ Servir arquivos estáticos não configurado")
    
    # Check health endpoint
    if "/api/health" in content or "get_health" in content:
        print("✅ Health endpoint disponível")
    else:
        issues.append("⚠️ Health endpoint não encontrado")
    
    return issues

def validate_optimized_config():
    """Validate optimized trading configuration."""
    print("\n⚙️ VALIDANDO CONFIGURAÇÃO OTIMIZADA")
    print("-" * 50)
    
    issues = []
    config_file = Path("trading_config.json")
    
    if not config_file.exists():
        issues.append("❌ trading_config.json não encontrado")
        return issues
    
    print("✅ trading_config.json encontrado")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        issues.append(f"❌ Erro ao ler JSON: {e}")
        return issues
    
    # Check structure
    if 'global_config' in config and 'bot_configs' in config:
        print("✅ Estrutura de configuração válida")
        
        global_config = config['global_config']
        bot_configs = config['bot_configs']
        
        # Check global settings
        if 'total_capital' in global_config:
            capital = global_config['total_capital']
            print(f"✅ Capital configurado: ${capital:,.2f}")
        else:
            issues.append("❌ Capital não configurado")
        
        if global_config.get('paper_trading', False):
            print("✅ Paper trading ativo (seguro para deploy)")
        else:
            issues.append("⚠️ Paper trading não ativo")
        
        # Check bot count
        bot_count = len(bot_configs)
        if bot_count >= 3:
            print(f"✅ {bot_count} bots configurados")
        else:
            issues.append(f"⚠️ Apenas {bot_count} bots configurados")
        
        # Check bot configuration
        for i, bot in enumerate(bot_configs):
            if 'symbol' in bot and 'timeframe' in bot:
                print(f"✅ Bot {i+1}: {bot['symbol']} {bot['timeframe']}")
            else:
                issues.append(f"❌ Bot {i+1} mal configurado")
    else:
        issues.append("❌ Estrutura de configuração inválida")
    
    return issues

def validate_frontend_config():
    """Validate frontend configuration for Docker build."""
    print("\n📱 VALIDANDO CONFIGURAÇÃO DO FRONTEND")
    print("-" * 50)
    
    issues = []
    frontend_dir = Path("frontend_react")
    
    if not frontend_dir.exists():
        issues.append("❌ Diretório frontend_react não encontrado")
        return issues
    
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
                build_script = package_config['scripts']['build']
                print(f"✅ Script de build: {build_script}")
            else:
                issues.append("❌ Script de build não configurado")
            
            # Check dependencies
            deps = package_config.get('dependencies', {})
            dev_deps = package_config.get('devDependencies', {})
            all_deps = {**deps, **dev_deps}
            
            required_deps = ['react', 'react-dom', 'vite']
            missing_deps = []
            
            for dep in required_deps:
                if dep not in all_deps:
                    missing_deps.append(dep)
            
            if not missing_deps:
                print("✅ Dependências principais encontradas")
            else:
                issues.append(f"❌ Dependências faltando: {', '.join(missing_deps)}")
        
        except json.JSONDecodeError as e:
            issues.append(f"❌ Erro ao ler package.json: {e}")
    else:
        issues.append("❌ package.json não encontrado")
    
    # Check Vite config
    vite_config = frontend_dir / "vite.config.js"
    if vite_config.exists():
        print("✅ vite.config.js encontrado")
    else:
        issues.append("⚠️ vite.config.js não encontrado")
    
    # Check source files
    src_dir = frontend_dir / "src"
    if src_dir.exists():
        print("✅ Diretório src encontrado")
        
        # Check key files
        key_files = ['App.jsx', 'main.jsx']
        for file in key_files:
            if (src_dir / file).exists():
                print(f"✅ {file} encontrado")
            else:
                issues.append(f"❌ {file} não encontrado")
    else:
        issues.append("❌ Diretório src não encontrado")
    
    return issues

def generate_deployment_summary():
    """Generate deployment summary and instructions."""
    print("\n📖 RESUMO E INSTRUÇÕES DE DEPLOY")
    print("-" * 50)
    
    summary = """
🚀 DEPLOY DO SISTEMA OTIMIZADO

📋 ARQUIVOS VALIDADOS:
   ✅ Dockerfile (multi-stage build)
   ✅ docker-compose.yml (serviços configurados)
   ✅ nginx.conf (proxy reverso + SSL)
   ✅ api_server.py (porta 12000, host 0.0.0.0)
   ✅ trading_config.json (configuração otimizada)
   ✅ frontend_react/ (React + Vite)

🔧 COMANDOS DE DEPLOY:

1️⃣ Build da imagem:
   docker build -t trading-bot-ml:optimized .

2️⃣ Deploy com compose:
   docker-compose up -d

3️⃣ Verificar status:
   docker-compose ps
   docker logs trading-bot-master

4️⃣ Acessar aplicação:
   http://localhost:12000 (direto)
   http://localhost (via Nginx)

🔍 MONITORAMENTO:
   • docker-compose logs -f trading-bot
   • curl http://localhost:12000/api/health
   • docker stats trading-bot-master

⚠️ IMPORTANTE:
   • Sistema em paper trading por padrão
   • Configure SSL para produção
   • Defina variáveis de ambiente Binance
   • Monitore logs regularmente
"""
    
    print(summary)

def main():
    """Main validation function."""
    print_banner()
    
    all_issues = []
    
    # Run all validations
    all_issues.extend(validate_dockerfile())
    all_issues.extend(validate_docker_compose())
    all_issues.extend(validate_nginx_config())
    all_issues.extend(validate_api_server())
    all_issues.extend(validate_optimized_config())
    all_issues.extend(validate_frontend_config())
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 RESUMO DA VALIDAÇÃO")
    print("=" * 80)
    
    if not all_issues:
        print("🎉 TODAS AS VALIDAÇÕES PASSARAM!")
        print("✅ Configuração de deploy está correta")
        print("✅ Sistema otimizado compatível com Docker")
        print("✅ Pronto para deploy em produção")
        
        status = "✅ APROVADO PARA DEPLOY"
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
            print(f"\n🚨 PROBLEMAS CRÍTICOS DEVEM SER CORRIGIDOS")
            status = "❌ REQUER CORREÇÕES"
        else:
            print(f"\n✅ Nenhum problema crítico - deploy pode prosseguir")
            status = "⚠️ DEPLOY COM CUIDADO"
    
    # Generate deployment guide
    generate_deployment_summary()
    
    print(f"\n🏁 STATUS FINAL: {status}")
    
    return len([i for i in all_issues if "❌" in i]) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)