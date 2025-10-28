-- =====================================================
-- Trading Bot ML - MySQL HeatWave Database Schema
-- =====================================================
-- Este script cria todas as tabelas, índices e relações
-- necessárias para o sistema de trading automatizado
-- =====================================================

-- Configurações iniciais
SET FOREIGN_KEY_CHECKS = 0;
SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";

-- =====================================================
-- 1. TABELA DE CONFIGURAÇÕES GLOBAIS
-- =====================================================
CREATE TABLE IF NOT EXISTS `global_configs` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `total_capital` DECIMAL(15,2) NOT NULL DEFAULT 10000.00,
    `max_concurrent_trades` INT NOT NULL DEFAULT 4,
    `daily_loss_limit` DECIMAL(5,4) NOT NULL DEFAULT 0.0500,
    `daily_profit_target` DECIMAL(5,4) NOT NULL DEFAULT 0.0300,
    `emergency_stop_drawdown` DECIMAL(5,4) NOT NULL DEFAULT 0.0800,
    `paper_trading` BOOLEAN NOT NULL DEFAULT TRUE,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `updated_by` VARCHAR(100) DEFAULT 'system',
    `is_active` BOOLEAN NOT NULL DEFAULT TRUE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 2. TABELA DE BOTS DE TRADING
-- =====================================================
CREATE TABLE IF NOT EXISTS `trading_bots` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `bot_id` VARCHAR(50) NOT NULL UNIQUE,
    `symbol` VARCHAR(20) NOT NULL,
    `timeframe` VARCHAR(10) NOT NULL,
    `capital_allocation` DECIMAL(5,4) NOT NULL,
    `max_risk_per_trade` DECIMAL(5,4) NOT NULL,
    `confidence_threshold` DECIMAL(5,4) NOT NULL,
    `stop_loss_pct` DECIMAL(5,4) NOT NULL,
    `take_profit_pct` DECIMAL(5,4) NOT NULL,
    `enabled` BOOLEAN NOT NULL DEFAULT TRUE,
    `status` ENUM('stopped', 'running', 'paused', 'error') DEFAULT 'stopped',
    `total_trades` INT DEFAULT 0,
    `winning_trades` INT DEFAULT 0,
    `total_pnl` DECIMAL(15,8) DEFAULT 0.00000000,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX `idx_symbol` (`symbol`),
    INDEX `idx_status` (`status`),
    INDEX `idx_enabled` (`enabled`),
    INDEX `idx_bot_id` (`bot_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 3. TABELA DE TRADES
-- =====================================================
CREATE TABLE IF NOT EXISTS `trades` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `trade_id` VARCHAR(100) NOT NULL UNIQUE,
    `bot_id` VARCHAR(50) NOT NULL,
    `symbol` VARCHAR(20) NOT NULL,
    `direction` TINYINT NOT NULL COMMENT '1=long, -1=short',
    `entry_time` TIMESTAMP NOT NULL,
    `entry_price` DECIMAL(20,8) NOT NULL,
    `quantity` DECIMAL(20,8) NOT NULL,
    `stop_loss` DECIMAL(20,8) NOT NULL,
    `take_profit` DECIMAL(20,8) NOT NULL,
    `exit_time` TIMESTAMP NULL,
    `exit_price` DECIMAL(20,8) NULL,
    `pnl` DECIMAL(15,8) NULL,
    `pnl_pct` DECIMAL(8,4) NULL,
    `status` ENUM('open', 'closed', 'cancelled') NOT NULL DEFAULT 'open',
    `exit_reason` VARCHAR(100) NULL,
    `confidence` DECIMAL(5,4) NULL,
    `signal_strength` DECIMAL(5,4) NULL,
    `fees` DECIMAL(15,8) DEFAULT 0.00000000,
    `slippage` DECIMAL(8,4) DEFAULT 0.0000,
    `metadata` JSON NULL,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX `idx_trade_id` (`trade_id`),
    INDEX `idx_bot_id` (`bot_id`),
    INDEX `idx_symbol` (`symbol`),
    INDEX `idx_status` (`status`),
    INDEX `idx_entry_time` (`entry_time`),
    INDEX `idx_exit_time` (`exit_time`),
    INDEX `idx_pnl` (`pnl`),
    INDEX `idx_direction` (`direction`),
    
    FOREIGN KEY (`bot_id`) REFERENCES `trading_bots`(`bot_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 4. TABELA DE LOGS DO SISTEMA
-- =====================================================
CREATE TABLE IF NOT EXISTS `system_logs` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `timestamp` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `level` ENUM('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'SUCCESS') NOT NULL,
    `source` VARCHAR(100) NOT NULL,
    `message` TEXT NOT NULL,
    `bot_id` VARCHAR(50) NULL,
    `trade_id` VARCHAR(100) NULL,
    `metadata` JSON NULL,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX `idx_timestamp` (`timestamp`),
    INDEX `idx_level` (`level`),
    INDEX `idx_source` (`source`),
    INDEX `idx_bot_id` (`bot_id`),
    INDEX `idx_trade_id` (`trade_id`),
    
    FOREIGN KEY (`bot_id`) REFERENCES `trading_bots`(`bot_id`) ON DELETE SET NULL ON UPDATE CASCADE,
    FOREIGN KEY (`trade_id`) REFERENCES `trades`(`trade_id`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 5. TABELA DE SINAIS DE TRADING
-- =====================================================
CREATE TABLE IF NOT EXISTS `trading_signals` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `signal_id` VARCHAR(100) NOT NULL UNIQUE,
    `bot_id` VARCHAR(50) NOT NULL,
    `symbol` VARCHAR(20) NOT NULL,
    `direction` TINYINT NOT NULL COMMENT '1=long, -1=short',
    `strength` DECIMAL(5,4) NOT NULL,
    `confidence` DECIMAL(5,4) NOT NULL,
    `timestamp` TIMESTAMP NOT NULL,
    `entry_price` DECIMAL(20,8) NOT NULL,
    `stop_loss` DECIMAL(20,8) NOT NULL,
    `take_profit` DECIMAL(20,8) NOT NULL,
    `executed` BOOLEAN DEFAULT FALSE,
    `trade_id` VARCHAR(100) NULL,
    `metadata` JSON NULL,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX `idx_signal_id` (`signal_id`),
    INDEX `idx_bot_id` (`bot_id`),
    INDEX `idx_symbol` (`symbol`),
    INDEX `idx_timestamp` (`timestamp`),
    INDEX `idx_executed` (`executed`),
    INDEX `idx_confidence` (`confidence`),
    
    FOREIGN KEY (`bot_id`) REFERENCES `trading_bots`(`bot_id`) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (`trade_id`) REFERENCES `trades`(`trade_id`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 6. TABELA DE PERFORMANCE DIÁRIA
-- =====================================================
CREATE TABLE IF NOT EXISTS `daily_performance` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `date` DATE NOT NULL,
    `bot_id` VARCHAR(50) NULL,
    `total_trades` INT DEFAULT 0,
    `winning_trades` INT DEFAULT 0,
    `losing_trades` INT DEFAULT 0,
    `total_pnl` DECIMAL(15,8) DEFAULT 0.00000000,
    `gross_profit` DECIMAL(15,8) DEFAULT 0.00000000,
    `gross_loss` DECIMAL(15,8) DEFAULT 0.00000000,
    `win_rate` DECIMAL(5,4) DEFAULT 0.0000,
    `profit_factor` DECIMAL(8,4) DEFAULT 0.0000,
    `avg_win` DECIMAL(15,8) DEFAULT 0.00000000,
    `avg_loss` DECIMAL(15,8) DEFAULT 0.00000000,
    `max_drawdown` DECIMAL(8,4) DEFAULT 0.0000,
    `sharpe_ratio` DECIMAL(8,4) DEFAULT 0.0000,
    `total_fees` DECIMAL(15,8) DEFAULT 0.00000000,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    UNIQUE KEY `unique_daily_bot` (`date`, `bot_id`),
    INDEX `idx_date` (`date`),
    INDEX `idx_bot_id` (`bot_id`),
    INDEX `idx_total_pnl` (`total_pnl`),
    INDEX `idx_win_rate` (`win_rate`),
    
    FOREIGN KEY (`bot_id`) REFERENCES `trading_bots`(`bot_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 7. TABELA DE EQUITY CURVE
-- =====================================================
CREATE TABLE IF NOT EXISTS `equity_curve` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `timestamp` TIMESTAMP NOT NULL,
    `bot_id` VARCHAR(50) NULL,
    `balance` DECIMAL(15,8) NOT NULL,
    `equity` DECIMAL(15,8) NOT NULL,
    `drawdown` DECIMAL(8,4) DEFAULT 0.0000,
    `drawdown_pct` DECIMAL(8,4) DEFAULT 0.0000,
    `total_trades` INT DEFAULT 0,
    `open_trades` INT DEFAULT 0,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX `idx_timestamp` (`timestamp`),
    INDEX `idx_bot_id` (`bot_id`),
    INDEX `idx_equity` (`equity`),
    INDEX `idx_drawdown` (`drawdown`),
    
    FOREIGN KEY (`bot_id`) REFERENCES `trading_bots`(`bot_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 8. TABELA DE CONFIGURAÇÕES DE SISTEMA
-- =====================================================
CREATE TABLE IF NOT EXISTS `system_settings` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `setting_key` VARCHAR(100) NOT NULL UNIQUE,
    `setting_value` TEXT NOT NULL,
    `setting_type` ENUM('string', 'integer', 'float', 'boolean', 'json') NOT NULL DEFAULT 'string',
    `description` TEXT NULL,
    `category` VARCHAR(50) DEFAULT 'general',
    `is_sensitive` BOOLEAN DEFAULT FALSE,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX `idx_setting_key` (`setting_key`),
    INDEX `idx_category` (`category`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 9. INSERIR DADOS INICIAIS
-- =====================================================

-- Configuração global padrão
INSERT INTO `global_configs` (
    `total_capital`, `max_concurrent_trades`, `daily_loss_limit`, 
    `daily_profit_target`, `emergency_stop_drawdown`, `paper_trading`
) VALUES (
    1200.00, 4, 0.0350, 0.0250, 0.0800, TRUE
) ON DUPLICATE KEY UPDATE `updated_at` = CURRENT_TIMESTAMP;

-- Configurações de sistema padrão
INSERT INTO `system_settings` (`setting_key`, `setting_value`, `setting_type`, `description`, `category`) VALUES
('api_rate_limit', '100', 'integer', 'Limite de requisições por minuto', 'api'),
('log_retention_days', '30', 'integer', 'Dias para manter logs no banco', 'logging'),
('backup_enabled', 'true', 'boolean', 'Habilitar backup automático', 'backup'),
('websocket_enabled', 'true', 'boolean', 'Habilitar WebSocket para logs em tempo real', 'realtime'),
('max_equity_points', '1000', 'integer', 'Máximo de pontos na equity curve', 'performance')
ON DUPLICATE KEY UPDATE `updated_at` = CURRENT_TIMESTAMP;

-- =====================================================
-- 10. VIEWS PARA RELATÓRIOS
-- =====================================================

-- View para estatísticas gerais
CREATE OR REPLACE VIEW `v_trading_stats` AS
SELECT 
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
    ROUND(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate,
    ROUND(SUM(pnl), 8) as total_pnl,
    ROUND(AVG(CASE WHEN pnl > 0 THEN pnl END), 8) as avg_win,
    ROUND(AVG(CASE WHEN pnl < 0 THEN pnl END), 8) as avg_loss,
    ROUND(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 8) as gross_profit,
    ROUND(ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)), 8) as gross_loss,
    ROUND(SUM(fees), 8) as total_fees
FROM trades 
WHERE status = 'closed';

-- View para estatísticas por bot
CREATE OR REPLACE VIEW `v_bot_stats` AS
SELECT 
    t.bot_id,
    tb.symbol,
    tb.timeframe,
    tb.status as bot_status,
    COUNT(*) as total_trades,
    SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    ROUND(SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate,
    ROUND(SUM(t.pnl), 8) as total_pnl,
    ROUND(AVG(CASE WHEN t.pnl > 0 THEN t.pnl END), 8) as avg_win,
    ROUND(AVG(CASE WHEN t.pnl < 0 THEN t.pnl END), 8) as avg_loss,
    ROUND(SUM(t.fees), 8) as total_fees,
    MAX(t.updated_at) as last_trade
FROM trades t
JOIN trading_bots tb ON t.bot_id = tb.bot_id
WHERE t.status = 'closed'
GROUP BY t.bot_id, tb.symbol, tb.timeframe, tb.status;

-- View para trades ativos
CREATE OR REPLACE VIEW `v_active_trades` AS
SELECT 
    t.trade_id,
    t.bot_id,
    tb.symbol,
    t.direction,
    t.entry_time,
    t.entry_price,
    t.stop_loss,
    t.take_profit,
    t.quantity,
    TIMESTAMPDIFF(MINUTE, t.entry_time, NOW()) as duration_minutes,
    t.confidence,
    t.signal_strength
FROM trades t
JOIN trading_bots tb ON t.bot_id = tb.bot_id
WHERE t.status = 'open'
ORDER BY t.entry_time DESC;

-- =====================================================
-- 11. STORED PROCEDURES
-- =====================================================

DELIMITER //

-- Procedure para calcular métricas de performance
CREATE PROCEDURE `CalculatePerformanceMetrics`(
    IN p_bot_id VARCHAR(50),
    IN p_start_date DATE,
    IN p_end_date DATE
)
BEGIN
    DECLARE v_total_trades INT DEFAULT 0;
    DECLARE v_winning_trades INT DEFAULT 0;
    DECLARE v_total_pnl DECIMAL(15,8) DEFAULT 0;
    DECLARE v_gross_profit DECIMAL(15,8) DEFAULT 0;
    DECLARE v_gross_loss DECIMAL(15,8) DEFAULT 0;
    DECLARE v_win_rate DECIMAL(5,4) DEFAULT 0;
    DECLARE v_profit_factor DECIMAL(8,4) DEFAULT 0;
    DECLARE v_sharpe_ratio DECIMAL(8,4) DEFAULT 0;
    
    -- Calcular métricas básicas
    SELECT 
        COUNT(*),
        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
        SUM(pnl),
        SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END),
        ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END))
    INTO v_total_trades, v_winning_trades, v_total_pnl, v_gross_profit, v_gross_loss
    FROM trades 
    WHERE status = 'closed'
        AND (p_bot_id IS NULL OR bot_id = p_bot_id)
        AND DATE(entry_time) BETWEEN p_start_date AND p_end_date;
    
    -- Calcular win rate
    IF v_total_trades > 0 THEN
        SET v_win_rate = v_winning_trades / v_total_trades;
    END IF;
    
    -- Calcular profit factor
    IF v_gross_loss > 0 THEN
        SET v_profit_factor = v_gross_profit / v_gross_loss;
    END IF;
    
    -- Calcular Sharpe ratio (simplificado)
    IF v_total_trades > 1 THEN
        SELECT 
            CASE 
                WHEN STDDEV(pnl) > 0 THEN AVG(pnl) / STDDEV(pnl) * SQRT(252)
                ELSE 0 
            END
        INTO v_sharpe_ratio
        FROM trades 
        WHERE status = 'closed'
            AND (p_bot_id IS NULL OR bot_id = p_bot_id)
            AND DATE(entry_time) BETWEEN p_start_date AND p_end_date;
    END IF;
    
    -- Inserir/atualizar performance diária
    INSERT INTO daily_performance (
        date, bot_id, total_trades, winning_trades, total_pnl,
        gross_profit, gross_loss, win_rate, profit_factor, sharpe_ratio
    ) VALUES (
        p_end_date, p_bot_id, v_total_trades, v_winning_trades, v_total_pnl,
        v_gross_profit, v_gross_loss, v_win_rate, v_profit_factor, v_sharpe_ratio
    ) ON DUPLICATE KEY UPDATE
        total_trades = v_total_trades,
        winning_trades = v_winning_trades,
        total_pnl = v_total_pnl,
        gross_profit = v_gross_profit,
        gross_loss = v_gross_loss,
        win_rate = v_win_rate,
        profit_factor = v_profit_factor,
        sharpe_ratio = v_sharpe_ratio,
        updated_at = CURRENT_TIMESTAMP;
        
END //

-- Procedure para limpar dados antigos
CREATE PROCEDURE `CleanOldData`(
    IN p_days_to_keep INT DEFAULT 30
)
BEGIN
    DECLARE v_cutoff_date DATE;
    SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL p_days_to_keep DAY);
    
    -- Limpar logs antigos
    DELETE FROM system_logs WHERE DATE(timestamp) < v_cutoff_date;
    
    -- Limpar equity curve antiga (manter apenas pontos importantes)
    DELETE FROM equity_curve 
    WHERE DATE(timestamp) < v_cutoff_date 
        AND id NOT IN (
            SELECT * FROM (
                SELECT id FROM equity_curve 
                WHERE DATE(timestamp) < v_cutoff_date
                ORDER BY timestamp DESC 
                LIMIT 100
            ) as temp
        );
        
END //

DELIMITER ;

-- =====================================================
-- 12. TRIGGERS
-- =====================================================

DELIMITER //

-- Trigger para atualizar estatísticas do bot após inserir trade
CREATE TRIGGER `tr_trades_after_insert` 
AFTER INSERT ON `trades`
FOR EACH ROW
BEGIN
    UPDATE trading_bots 
    SET total_trades = total_trades + 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE bot_id = NEW.bot_id;
END //

-- Trigger para atualizar estatísticas do bot após atualizar trade
CREATE TRIGGER `tr_trades_after_update` 
AFTER UPDATE ON `trades`
FOR EACH ROW
BEGIN
    IF OLD.status != 'closed' AND NEW.status = 'closed' THEN
        UPDATE trading_bots 
        SET winning_trades = winning_trades + CASE WHEN NEW.pnl > 0 THEN 1 ELSE 0 END,
            total_pnl = total_pnl + COALESCE(NEW.pnl, 0),
            updated_at = CURRENT_TIMESTAMP
        WHERE bot_id = NEW.bot_id;
    END IF;
END //

DELIMITER ;

-- =====================================================
-- 13. TABELA DE RESTRIÇÕES DE HORÁRIO
-- =====================================================

-- Tabela para controlar restrições de operação por dia/horário
CREATE TABLE IF NOT EXISTS `trading_restrictions` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `name` VARCHAR(255) NOT NULL,
    `description` TEXT,
    `start_day_of_week` TINYINT NOT NULL COMMENT '0=Domingo, 1=Segunda, ..., 6=Sábado',
    `start_time` TIME NOT NULL,
    `end_day_of_week` TINYINT NOT NULL,
    `end_time` TIME NOT NULL,
    `timezone` VARCHAR(50) DEFAULT 'America/Sao_Paulo',
    `is_active` BOOLEAN DEFAULT TRUE,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX `idx_active_restrictions` (`is_active`),
    INDEX `idx_day_time` (`start_day_of_week`, `start_time`, `end_day_of_week`, `end_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Inserir restrição padrão: Sábado 07:00 a Domingo 20:00
INSERT INTO `trading_restrictions` (
    `name`, `description`, `start_day_of_week`, `start_time`, 
    `end_day_of_week`, `end_time`, `timezone`, `is_active`
) VALUES (
    'Weekend Restriction', 
    'Bloquear operações de sábado 07:00 até domingo 20:00 devido à instabilidade',
    6, '07:00:00',  -- Sábado às 07:00
    0, '20:00:00',  -- Domingo às 20:00
    'America/Sao_Paulo', 
    TRUE
) ON DUPLICATE KEY UPDATE `updated_at` = CURRENT_TIMESTAMP;

-- =====================================================
-- 14. FUNÇÃO PARA VERIFICAR RESTRIÇÕES
-- =====================================================

DELIMITER //

-- Função para verificar se trading está permitido no momento atual
CREATE FUNCTION `IsTradingAllowed`(
    p_check_time DATETIME,
    p_timezone VARCHAR(50)
) RETURNS BOOLEAN
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE v_day_of_week TINYINT;
    DECLARE v_time_of_day TIME;
    DECLARE v_restriction_count INT DEFAULT 0;
    
    -- Converter para timezone especificado (simplificado)
    SET v_day_of_week = DAYOFWEEK(p_check_time) - 1; -- MySQL DAYOFWEEK: 1=Sunday, converter para 0=Sunday
    SET v_time_of_day = TIME(p_check_time);
    
    -- Verificar se existe alguma restrição ativa que bloqueia este horário
    SELECT COUNT(*) INTO v_restriction_count
    FROM trading_restrictions
    WHERE is_active = TRUE
        AND (
            -- Restrição no mesmo dia
            (start_day_of_week = end_day_of_week 
             AND start_day_of_week = v_day_of_week
             AND v_time_of_day BETWEEN start_time AND end_time)
            OR
            -- Restrição que atravessa dias (ex: Sábado 07:00 a Domingo 20:00)
            (start_day_of_week != end_day_of_week
             AND (
                 (v_day_of_week = start_day_of_week AND v_time_of_day >= start_time)
                 OR
                 (v_day_of_week = end_day_of_week AND v_time_of_day <= end_time)
                 OR
                 (v_day_of_week > start_day_of_week AND v_day_of_week < end_day_of_week)
                 OR
                 -- Caso especial: Sábado para Domingo (6 -> 0)
                 (start_day_of_week > end_day_of_week 
                  AND (v_day_of_week >= start_day_of_week OR v_day_of_week <= end_day_of_week))
             ))
        );
    
    -- Retornar TRUE se não há restrições, FALSE se há restrições
    RETURN v_restriction_count = 0;
END //

DELIMITER ;

-- =====================================================
-- 15. FINALIZAÇÃO
-- =====================================================

SET FOREIGN_KEY_CHECKS = 1;
COMMIT;

-- Mostrar resumo das tabelas criadas
SELECT 
    TABLE_NAME as 'Tabela',
    TABLE_ROWS as 'Registros',
    ROUND(((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024), 2) as 'Tamanho (MB)'
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME IN (
        'global_configs', 'trading_bots', 'trades', 'system_logs',
        'trading_signals', 'daily_performance', 'equity_curve', 'system_settings',
        'trading_restrictions'
    )
ORDER BY TABLE_NAME;

-- =====================================================
-- SCRIPT CONCLUÍDO COM SUCESSO!
-- =====================================================
-- Para usar este script:
-- 1. Conecte-se ao seu MySQL HeatWave
-- 2. Selecione o database: USE your_database_name;
-- 3. Execute: SOURCE schema.sql;
-- =====================================================