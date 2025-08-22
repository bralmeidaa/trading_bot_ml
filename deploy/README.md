# 🚀 Deployment Guide - Trading Bot ML

## 📋 Overview

Este guia fornece instruções completas para fazer deploy do Trading Bot ML em uma VM do Oracle Cloud Infrastructure (OCI) usando Azure DevOps Pipeline.

## 🏗️ Arquitetura de Deploy

```
Azure DevOps → Docker Build → OCI VM → Docker Containers
     ↓              ↓            ↓           ↓
  Pipeline    →  Image Build  →  Deploy  →  Running App
```

## 🔧 Pré-requisitos

### 1. VM OCI
- **OS**: Ubuntu 20.04 LTS ou superior
- **CPU**: Mínimo 2 vCPUs (recomendado 4 vCPUs)
- **RAM**: Mínimo 4GB (recomendado 8GB)
- **Storage**: Mínimo 50GB
- **Network**: Porta 8000 aberta para seu IP

### 2. Azure DevOps
- Projeto configurado
- Service Connection para OCI VM
- Variables configuradas (ver seção Variables)

### 3. Domínio/IP (Opcional)
- IP público da VM OCI
- Domínio apontando para o IP (opcional)

## 🚀 Passo a Passo de Deploy

### Etapa 1: Configurar VM OCI

1. **Criar VM no OCI**
   ```bash
   # Conectar via SSH
   ssh ubuntu@YOUR_OCI_VM_IP
   ```

2. **Executar script de setup**
   ```bash
   # Baixar e executar script de configuração
   curl -sSL https://raw.githubusercontent.com/your-repo/trading-bot-ml/main/deploy/setup-oci-vm.sh | bash
   
   # Ou manualmente:
   wget https://raw.githubusercontent.com/your-repo/trading-bot-ml/main/deploy/setup-oci-vm.sh
   chmod +x setup-oci-vm.sh
   ./setup-oci-vm.sh
   ```

3. **Configurar firewall para seu IP**
   ```bash
   # Substituir YOUR_IP pelo seu IP público
   sudo ufw allow from YOUR_IP_ADDRESS to any port 8000
   sudo ufw reload
   ```

### Etapa 2: Configurar Azure DevOps

1. **Criar Service Connection**
   - Vá para Project Settings → Service connections
   - Adicione "SSH" connection
   - Configure com IP, usuário e chave SSH da VM OCI

2. **Configurar Variables**
   No Azure DevOps, vá para Pipelines → Library → Variable groups:

   ```yaml
   # Required Variables
   OCI_VM_IP: "xxx.xxx.xxx.xxx"          # IP público da VM OCI
   OCI_VM_USER: "ubuntu"                 # Usuário SSH
   OCI_SSH_KEY: "-----BEGIN PRIVATE..."  # Chave SSH privada
   
   # Optional (for live trading)
   BINANCE_API_KEY: "your_api_key"       # Binance API Key
   BINANCE_API_SECRET: "your_secret"     # Binance API Secret
   ```

3. **Criar Pipeline**
   - Importe o arquivo `azure-pipelines.yml`
   - Configure as variables necessárias
   - Execute o pipeline

### Etapa 3: Configurar Aplicação

1. **Configurar ambiente na VM**
   ```bash
   cd /opt/trading-bot-ml
   cp .env.template .env
   nano .env  # Editar com suas configurações
   ```

2. **Exemplo de .env**
   ```bash
   # Trading Bot ML Environment
   BINANCE_API_KEY=your_binance_api_key_here
   BINANCE_API_SECRET=your_binance_api_secret_here
   ENVIRONMENT=production
   PYTHONPATH=/app
   PYTHONUNBUFFERED=1
   ```

## 📊 Monitoramento e Manutenção

### Scripts de Monitoramento

1. **Status do Sistema**
   ```bash
   /opt/trading-bot-ml/monitor.sh
   ```

2. **Logs em Tempo Real**
   ```bash
   cd /opt/trading-bot-ml
   docker-compose logs -f
   ```

3. **Backup Manual**
   ```bash
   /opt/trading-bot-ml/backup.sh
   ```

### Comandos Úteis

```bash
# Reiniciar aplicação
sudo systemctl restart trading-bot-ml

# Ver status dos containers
docker-compose ps

# Atualizar aplicação
docker-compose pull
docker-compose up -d

# Ver logs específicos
docker-compose logs trading-bot

# Entrar no container
docker-compose exec trading-bot bash
```

## 🔒 Segurança

### Configurações de Firewall

```bash
# Permitir apenas seu IP
sudo ufw allow from YOUR_IP to any port 8000

# Verificar regras
sudo ufw status numbered

# Remover regra (se necessário)
sudo ufw delete [número]
```

### SSL/HTTPS (Opcional)

1. **Obter certificado SSL**
   ```bash
   # Usando Let's Encrypt
   sudo apt install certbot
   sudo certbot certonly --standalone -d your-domain.com
   ```

2. **Configurar Nginx**
   - Descomente seção HTTPS no `nginx.conf`
   - Copie certificados para `/opt/trading-bot-ml/ssl/`

## 🚨 Troubleshooting

### Problemas Comuns

1. **Container não inicia**
   ```bash
   # Verificar logs
   docker-compose logs trading-bot
   
   # Verificar recursos
   free -h
   df -h
   ```

2. **API não responde**
   ```bash
   # Testar localmente
   curl http://localhost:8000/api/health
   
   # Verificar firewall
   sudo ufw status
   ```

3. **Pipeline falha**
   - Verificar variables do Azure DevOps
   - Testar conexão SSH manualmente
   - Verificar logs do pipeline

### Logs Importantes

```bash
# Logs da aplicação
tail -f /opt/trading-bot-ml/logs/trading_system.log

# Logs do Docker
docker-compose logs -f

# Logs do sistema
sudo journalctl -u trading-bot-ml -f
```

## 📈 Otimizações de Performance

### Configurações de VM

1. **Aumentar recursos**
   - CPU: 4+ vCPUs para melhor performance
   - RAM: 8GB+ para múltiplos bots
   - Storage: SSD para melhor I/O

2. **Otimizar Docker**
   ```bash
   # Limpar recursos não utilizados
   docker system prune -a
   
   # Configurar limites de memória
   # Editar docker-compose.yml
   ```

### Monitoramento Avançado

1. **Instalar Prometheus + Grafana** (Opcional)
   ```bash
   # Adicionar ao docker-compose.yml
   # Configurar dashboards de monitoramento
   ```

## 🔄 Processo de Atualização

### Deploy Automático (Azure DevOps)

1. Commit código para branch `main`
2. Pipeline executa automaticamente
3. Deploy é feito na VM OCI
4. Verificação automática de saúde

### Deploy Manual

```bash
cd /opt/trading-bot-ml

# Parar aplicação
docker-compose down

# Atualizar código
git pull origin main

# Rebuild e restart
docker-compose build
docker-compose up -d
```

## 📞 Suporte

### Contatos de Emergência

- **Sistema Down**: Verificar logs e reiniciar
- **Performance Issues**: Monitorar recursos
- **Security Issues**: Verificar firewall e logs

### Backup e Recovery

```bash
# Backup automático (configurado via cron)
# Executa diariamente às 2:00 AM

# Restore manual
cd /opt/trading-bot-ml
tar -xzf backups/trading-bot-backup-YYYYMMDD_HHMMSS.tar.gz
docker-compose up -d
```

---

## ✅ Checklist de Deploy

### Pré-Deploy
- [ ] VM OCI criada e configurada
- [ ] Script de setup executado
- [ ] Firewall configurado para seu IP
- [ ] Azure DevOps configurado
- [ ] Variables definidas
- [ ] SSH keys configuradas

### Deploy
- [ ] Pipeline executado com sucesso
- [ ] Containers rodando
- [ ] API respondendo (health check)
- [ ] Dashboard acessível
- [ ] Logs sem erros críticos

### Pós-Deploy
- [ ] Monitoramento configurado
- [ ] Backups funcionando
- [ ] SSL configurado (se aplicável)
- [ ] Documentação atualizada
- [ ] Equipe notificada

---

**🎉 Parabéns! Seu Trading Bot ML está rodando em produção!**

**📊 Acesse o dashboard em**: `http://YOUR_OCI_VM_IP:8000`