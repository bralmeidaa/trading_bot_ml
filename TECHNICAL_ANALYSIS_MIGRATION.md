# Migração de Análise Técnica: pandas-ta → ta

## Resumo das Mudanças

Para resolver conflitos de dependências do Docker, substituímos `pandas-ta` pela biblioteca `ta` (Technical Analysis Library).

## Biblioteca Anterior vs Nova

| Aspecto | pandas-ta | ta |
|---------|-----------|-----|
| **Versão** | 0.4.67b0 | 0.10.2 |
| **Numpy** | >=2.2.6 | Compatível com numpy<2.0 |
| **Estabilidade** | Beta | Estável |
| **Compatibilidade** | Conflitos com ML libs | Compatível com scipy, sklearn |

## Principais Diferenças de API

### Importação
```python
# Antes (pandas-ta)
import pandas_ta as ta

# Agora (ta)
import ta
```

### Indicadores Comuns

#### RSI (Relative Strength Index)
```python
# Antes
df.ta.rsi(length=14)

# Agora
ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
```

#### MACD
```python
# Antes
df.ta.macd()

# Agora
macd = ta.trend.MACD(close=df['close'])
macd_line = macd.macd()
macd_signal = macd.macd_signal()
macd_histogram = macd.macd_diff()
```

#### Bollinger Bands
```python
# Antes
df.ta.bbands()

# Agora
bb = ta.volatility.BollingerBands(close=df['close'])
bb_upper = bb.bollinger_hband()
bb_middle = bb.bollinger_mavg()
bb_lower = bb.bollinger_lband()
```

#### Moving Averages
```python
# Antes
df.ta.sma(length=20)
df.ta.ema(length=20)

# Agora
ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
```

#### Stochastic
```python
# Antes
df.ta.stoch()

# Agora
stoch = ta.momentum.StochasticOscillator(
    high=df['high'], 
    low=df['low'], 
    close=df['close']
)
stoch_k = stoch.stoch()
stoch_d = stoch.stoch_signal()
```

## Categorias de Indicadores na Biblioteca `ta`

### 1. Volume
- `ta.volume.AccDistIndexIndicator`
- `ta.volume.OnBalanceVolumeIndicator`
- `ta.volume.ChaikinMoneyFlowIndicator`
- `ta.volume.VolumeSMAIndicator`

### 2. Volatility
- `ta.volatility.BollingerBands`
- `ta.volatility.KeltnerChannel`
- `ta.volatility.DonchianChannel`
- `ta.volatility.AverageTrueRange`

### 3. Trend
- `ta.trend.MACD`
- `ta.trend.EMAIndicator`
- `ta.trend.SMAIndicator`
- `ta.trend.ADXIndicator`
- `ta.trend.AroonIndicator`

### 4. Momentum
- `ta.momentum.RSIIndicator`
- `ta.momentum.StochasticOscillator`
- `ta.momentum.WilliamsRIndicator`
- `ta.momentum.ROCIndicator`

### 5. Others
- `ta.others.DailyReturnIndicator`
- `ta.others.CumulativeReturnIndicator`

## Exemplo de Função Helper

```python
import ta
import pandas as pd

def add_technical_indicators(df):
    """
    Adiciona indicadores técnicos ao DataFrame
    """
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    
    # Moving Averages
    df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['ema_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close']
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    return df
```

## Vantagens da Nova Biblioteca

1. **Estabilidade**: Versão estável vs beta
2. **Compatibilidade**: Sem conflitos de numpy
3. **Documentação**: Melhor documentada
4. **Performance**: Otimizada para DataFrames grandes
5. **Manutenção**: Ativamente mantida

## Próximos Passos

1. Atualizar código existente que usa indicadores técnicos
2. Testar todos os indicadores com dados reais
3. Validar performance dos novos indicadores
4. Atualizar documentação do sistema

## Links Úteis

- [Documentação da biblioteca `ta`](https://technical-analysis-library-in-python.readthedocs.io/)
- [Repositório GitHub](https://github.com/bukosabino/ta)
- [Exemplos de uso](https://github.com/bukosabino/ta/tree/master/examples)