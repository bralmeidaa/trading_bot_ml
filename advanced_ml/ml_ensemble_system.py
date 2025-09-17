#!/usr/bin/env python3
"""
Advanced ML Ensemble System
Sistema de ensemble com múltiplos modelos de ML para trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import joblib
import json
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from .feature_engineering import AdvancedFeatureEngineer
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from feature_engineering import AdvancedFeatureEngineer

@dataclass
class ModelConfig:
    """Configuração de um modelo individual."""
    name: str
    model_type: str
    hyperparameters: Dict
    weight: float
    enabled: bool
    performance_threshold: float

@dataclass
class EnsembleResult:
    """Resultado do ensemble."""
    prediction: int  # 0=sell, 1=hold, 2=buy
    confidence: float
    individual_predictions: Dict[str, Dict]
    feature_importance: Dict[str, float]
    model_weights: Dict[str, float]
    timestamp: datetime

class MLEnsembleSystem:
    """Sistema avançado de ensemble de ML para trading."""
    
    def __init__(self, config_path: str = 'ml_ensemble_config.json'):
        self.config_path = config_path
        self.feature_engineer = AdvancedFeatureEngineer()
        self.models = {}
        self.model_configs = self._load_model_configs()
        self.ensemble_history = []
        self.performance_metrics = {}
        
        # Label encoder for targets
        self.label_encoder = LabelEncoder()
        
        # Initialize models
        self._initialize_models()
        
    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Carrega configurações dos modelos."""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            configs = {}
            for name, config_data in data.items():
                configs[name] = ModelConfig(**config_data)
            
            return configs
        except FileNotFoundError:
            return self._create_default_model_configs()
    
    def _create_default_model_configs(self) -> Dict[str, ModelConfig]:
        """Cria configurações padrão dos modelos."""
        
        configs = {
            'random_forest': ModelConfig(
                name='random_forest',
                model_type='RandomForestClassifier',
                hyperparameters={
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                },
                weight=0.20,
                enabled=True,
                performance_threshold=0.55
            ),
            
            'gradient_boosting': ModelConfig(
                name='gradient_boosting',
                model_type='GradientBoostingClassifier',
                hyperparameters={
                    'n_estimators': 150,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'random_state': 42
                },
                weight=0.18,
                enabled=True,
                performance_threshold=0.55
            ),
            
            'svm': ModelConfig(
                name='svm',
                model_type='SVC',
                hyperparameters={
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'probability': True,
                    'random_state': 42
                },
                weight=0.15,
                enabled=True,
                performance_threshold=0.52
            ),
            
            'logistic_regression': ModelConfig(
                name='logistic_regression',
                model_type='LogisticRegression',
                hyperparameters={
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42
                },
                weight=0.12,
                enabled=True,
                performance_threshold=0.50
            ),
            
            'neural_network': ModelConfig(
                name='neural_network',
                model_type='MLPClassifier',
                hyperparameters={
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.001,
                    'learning_rate': 'adaptive',
                    'max_iter': 500,
                    'random_state': 42
                },
                weight=0.15,
                enabled=True,
                performance_threshold=0.53
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            configs['xgboost'] = ModelConfig(
                name='xgboost',
                model_type='XGBClassifier',
                hyperparameters={
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                weight=0.20,
                enabled=True,
                performance_threshold=0.58
            )
        
        # Save default configs
        self._save_model_configs(configs)
        return configs
    
    def _save_model_configs(self, configs: Dict[str, ModelConfig]):
        """Salva configurações dos modelos."""
        data = {}
        for name, config in configs.items():
            data[name] = {
                'name': config.name,
                'model_type': config.model_type,
                'hyperparameters': config.hyperparameters,
                'weight': config.weight,
                'enabled': config.enabled,
                'performance_threshold': config.performance_threshold
            }
        
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _initialize_models(self):
        """Inicializa todos os modelos."""
        
        for name, config in self.model_configs.items():
            if not config.enabled:
                continue
            
            try:
                if config.model_type == 'RandomForestClassifier':
                    model = RandomForestClassifier(**config.hyperparameters)
                elif config.model_type == 'GradientBoostingClassifier':
                    model = GradientBoostingClassifier(**config.hyperparameters)
                elif config.model_type == 'SVC':
                    model = SVC(**config.hyperparameters)
                elif config.model_type == 'LogisticRegression':
                    model = LogisticRegression(**config.hyperparameters)
                elif config.model_type == 'MLPClassifier':
                    model = MLPClassifier(**config.hyperparameters)
                elif config.model_type == 'XGBClassifier' and XGBOOST_AVAILABLE:
                    model = xgb.XGBClassifier(**config.hyperparameters)
                else:
                    print(f"Model type {config.model_type} not supported or library not available")
                    continue
                
                self.models[name] = {
                    'model': model,
                    'config': config,
                    'trained': False,
                    'performance': {}
                }
                
            except Exception as e:
                print(f"Error initializing model {name}: {e}")
    
    def prepare_training_data(self, price_data: pd.DataFrame, 
                            lookforward_periods: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara dados para treinamento."""
        
        # Generate features
        features_df = self.feature_engineer.generate_all_features(price_data)
        
        # Create target variable (future price direction)
        close_prices = features_df['close'].values
        targets = []
        
        for i in range(len(close_prices)):
            if i >= len(close_prices) - lookforward_periods:
                targets.append(1)  # Hold for last periods
            else:
                future_price = close_prices[i + lookforward_periods]
                current_price = close_prices[i]
                price_change = (future_price - current_price) / current_price
                
                if price_change > 0.01:  # 1% gain threshold
                    targets.append(2)  # Buy
                elif price_change < -0.01:  # 1% loss threshold
                    targets.append(0)  # Sell
                else:
                    targets.append(1)  # Hold
        
        # Remove OHLCV columns for training
        feature_columns = [col for col in features_df.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        X = features_df[feature_columns]
        y = pd.Series(targets)
        
        # Scale features
        X_scaled = self.feature_engineer.scale_features(X)
        
        return X_scaled[feature_columns], y
    
    def train_ensemble(self, price_data: pd.DataFrame, 
                      test_size: float = 0.2) -> Dict[str, Any]:
        """Treina todos os modelos do ensemble."""
        
        print("🤖 TREINANDO ENSEMBLE DE ML")
        print("=" * 50)
        
        # Prepare data
        X, y = self.prepare_training_data(price_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📊 Dados de treinamento: {X_train.shape}")
        print(f"📊 Dados de teste: {X_test.shape}")
        print(f"📊 Distribuição de classes: {y.value_counts().to_dict()}")
        
        training_results = {}
        
        # Train each model
        for name, model_data in self.models.items():
            try:
                print(f"\n🔧 Treinando {name}...")
                
                model = model_data['model']
                config = model_data['config']
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on test set
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = model.score(X_test, y_test)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store performance
                performance = {
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'test_predictions': y_pred.tolist(),
                    'feature_importance': self._get_feature_importance(model, X.columns)
                }
                
                if y_pred_proba is not None:
                    try:
                        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                        performance['auc_score'] = auc_score
                    except:
                        performance['auc_score'] = None
                
                model_data['performance'] = performance
                model_data['trained'] = True
                
                training_results[name] = performance
                
                print(f"  ✅ Accuracy: {accuracy:.3f}")
                print(f"  ✅ CV Score: {cv_mean:.3f} ± {cv_std:.3f}")
                if performance.get('auc_score'):
                    print(f"  ✅ AUC Score: {performance['auc_score']:.3f}")
                
            except Exception as e:
                print(f"  ❌ Erro ao treinar {name}: {e}")
                training_results[name] = {'error': str(e)}
        
        # Update model weights based on performance
        self._update_model_weights()
        
        print(f"\n🎯 TREINAMENTO CONCLUÍDO")
        print(f"Modelos treinados: {len([m for m in self.models.values() if m['trained']])}")
        
        return training_results
    
    def predict_ensemble(self, price_data: pd.DataFrame) -> EnsembleResult:
        """Faz predição usando ensemble de modelos."""
        
        # Generate features for latest data point
        features_df = self.feature_engineer.generate_all_features(price_data)
        
        # Get latest features
        feature_columns = [col for col in features_df.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        X_scaled = self.feature_engineer.scale_features(features_df)
        latest_features = X_scaled[feature_columns].iloc[-1:].values
        
        individual_predictions = {}
        weighted_predictions = []
        model_weights = {}
        
        # Get predictions from each model
        for name, model_data in self.models.items():
            if not model_data['trained']:
                continue
            
            try:
                model = model_data['model']
                config = model_data['config']
                
                # Get prediction
                prediction = model.predict(latest_features)[0]
                
                # Get prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(latest_features)[0]
                    confidence = np.max(probabilities)
                else:
                    probabilities = None
                    confidence = 0.6  # Default confidence
                
                # Check if model meets performance threshold
                model_performance = model_data['performance'].get('accuracy', 0)
                if model_performance < config.performance_threshold:
                    continue  # Skip underperforming models
                
                individual_predictions[name] = {
                    'prediction': int(prediction),
                    'confidence': float(confidence),
                    'probabilities': probabilities.tolist() if probabilities is not None else None,
                    'weight': config.weight,
                    'performance': model_performance
                }
                
                # Add to weighted ensemble
                weighted_predictions.append({
                    'prediction': prediction,
                    'weight': config.weight,
                    'confidence': confidence
                })
                
                model_weights[name] = config.weight
                
            except Exception as e:
                print(f"Error getting prediction from {name}: {e}")
        
        # Calculate ensemble prediction
        if not weighted_predictions:
            return EnsembleResult(
                prediction=1,  # Hold
                confidence=0.0,
                individual_predictions={},
                feature_importance={},
                model_weights={},
                timestamp=datetime.now()
            )
        
        # Weighted voting
        total_weight = sum(p['weight'] for p in weighted_predictions)
        class_votes = {0: 0, 1: 0, 2: 0}  # sell, hold, buy
        
        for pred in weighted_predictions:
            class_votes[pred['prediction']] += pred['weight']
        
        # Get final prediction
        final_prediction = max(class_votes, key=class_votes.get)
        
        # Calculate ensemble confidence
        ensemble_confidence = class_votes[final_prediction] / total_weight
        
        # Adjust confidence based on agreement
        agreement_bonus = len([p for p in weighted_predictions 
                              if p['prediction'] == final_prediction]) / len(weighted_predictions)
        ensemble_confidence = (ensemble_confidence + agreement_bonus) / 2
        
        # Get feature importance from best performing model
        feature_importance = {}
        best_model = max(individual_predictions.items(), 
                        key=lambda x: x[1]['performance'])
        if best_model:
            best_model_name = best_model[0]
            if best_model_name in self.models:
                feature_importance = self.models[best_model_name]['performance'].get('feature_importance', {})
        
        result = EnsembleResult(
            prediction=final_prediction,
            confidence=ensemble_confidence,
            individual_predictions=individual_predictions,
            feature_importance=feature_importance,
            model_weights=model_weights,
            timestamp=datetime.now()
        )
        
        # Store in history
        self.ensemble_history.append(result)
        if len(self.ensemble_history) > 1000:
            self.ensemble_history = self.ensemble_history[-1000:]
        
        return result
    
    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """Extrai importância das features do modelo."""
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                return {}
            
            # Create importance dictionary
            importance_dict = {}
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = float(importance)
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            return {}
    
    def _update_model_weights(self):
        """Atualiza pesos dos modelos baseado na performance."""
        
        total_performance = 0
        model_performances = {}
        
        # Calculate total performance
        for name, model_data in self.models.items():
            if model_data['trained']:
                performance = model_data['performance'].get('cv_mean', 0)
                model_performances[name] = max(performance, 0.1)  # Minimum weight
                total_performance += model_performances[name]
        
        # Update weights proportionally
        if total_performance > 0:
            for name, performance in model_performances.items():
                new_weight = performance / total_performance
                self.model_configs[name].weight = new_weight
                
                print(f"📊 {name}: weight updated to {new_weight:.3f}")
    
    def get_ensemble_performance(self) -> Dict:
        """Retorna métricas de performance do ensemble."""
        
        if not self.ensemble_history:
            return {}
        
        recent_predictions = self.ensemble_history[-50:]  # Last 50 predictions
        
        # Calculate metrics
        avg_confidence = np.mean([p.confidence for p in recent_predictions])
        
        # Prediction distribution
        prediction_counts = {0: 0, 1: 0, 2: 0}
        for pred in recent_predictions:
            prediction_counts[pred.prediction] += 1
        
        # Model agreement analysis
        model_agreements = []
        for pred in recent_predictions:
            if pred.individual_predictions:
                predictions = [p['prediction'] for p in pred.individual_predictions.values()]
                agreement = len(set(predictions)) == 1  # All models agree
                model_agreements.append(agreement)
        
        agreement_rate = np.mean(model_agreements) if model_agreements else 0
        
        return {
            'total_predictions': len(self.ensemble_history),
            'recent_predictions': len(recent_predictions),
            'avg_confidence': avg_confidence,
            'prediction_distribution': prediction_counts,
            'model_agreement_rate': agreement_rate,
            'active_models': len([m for m in self.models.values() if m['trained']]),
            'model_performances': {
                name: model_data['performance'].get('cv_mean', 0)
                for name, model_data in self.models.items()
                if model_data['trained']
            }
        }
    
    def save_ensemble(self, filepath: str):
        """Salva o ensemble treinado."""
        
        ensemble_data = {
            'models': {},
            'model_configs': {},
            'feature_engineer': self.feature_engineer,
            'ensemble_history': self.ensemble_history[-100:],  # Save last 100
            'timestamp': datetime.now().isoformat()
        }
        
        # Save trained models
        for name, model_data in self.models.items():
            if model_data['trained']:
                ensemble_data['models'][name] = {
                    'model': model_data['model'],
                    'performance': model_data['performance'],
                    'trained': True
                }
        
        # Save configs
        for name, config in self.model_configs.items():
            ensemble_data['model_configs'][name] = {
                'name': config.name,
                'model_type': config.model_type,
                'hyperparameters': config.hyperparameters,
                'weight': config.weight,
                'enabled': config.enabled,
                'performance_threshold': config.performance_threshold
            }
        
        joblib.dump(ensemble_data, filepath)
        print(f"💾 Ensemble salvo em: {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Carrega ensemble salvo."""
        
        try:
            ensemble_data = joblib.load(filepath)
            
            self.models = ensemble_data.get('models', {})
            self.ensemble_history = ensemble_data.get('ensemble_history', [])
            
            # Reconstruct model configs
            config_data = ensemble_data.get('model_configs', {})
            for name, config_dict in config_data.items():
                self.model_configs[name] = ModelConfig(**config_dict)
            
            print(f"📂 Ensemble carregado de: {filepath}")
            print(f"Modelos carregados: {len(self.models)}")
            
        except Exception as e:
            print(f"❌ Erro ao carregar ensemble: {e}")

# Test the ML Ensemble System
async def test_ml_ensemble():
    """Test the ML Ensemble System."""
    
    # Create sample data
    periods = 1000  # More data for ML training
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
    
    # Generate realistic price data with trends
    base_price = 50000
    trend = np.linspace(0, 0.2, periods)  # 20% trend over period
    noise = np.random.normal(0, 0.02, periods)
    returns = trend/periods + noise
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
        'close': prices,
        'volume': np.random.normal(1000, 200, periods)
    }, index=dates)
    
    # Initialize ensemble
    ensemble = MLEnsembleSystem()
    
    print("🤖 TESTE DO SISTEMA ML ENSEMBLE")
    print("=" * 60)
    
    # Train ensemble
    training_results = ensemble.train_ensemble(df)
    
    # Make predictions
    print(f"\n🔮 FAZENDO PREDIÇÕES...")
    
    for i in range(5):
        # Use different portions of data for prediction
        test_data = df.iloc[:-10+i*2]  # Varying test data
        result = ensemble.predict_ensemble(test_data)
        
        prediction_labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
        
        print(f"\n📊 PREDIÇÃO {i+1}:")
        print(f"  Decisão: {prediction_labels[result.prediction]}")
        print(f"  Confiança: {result.confidence:.3f}")
        print(f"  Modelos ativos: {len(result.individual_predictions)}")
        
        # Show individual model predictions
        for model_name, pred_data in result.individual_predictions.items():
            print(f"    {model_name}: {prediction_labels[pred_data['prediction']]} "
                  f"(conf: {pred_data['confidence']:.3f})")
    
    # Performance summary
    performance = ensemble.get_ensemble_performance()
    print(f"\n📈 PERFORMANCE DO ENSEMBLE:")
    print(f"Total de predições: {performance['total_predictions']}")
    print(f"Confiança média: {performance['avg_confidence']:.3f}")
    print(f"Taxa de concordância: {performance['model_agreement_rate']:.3f}")
    print(f"Modelos ativos: {performance['active_models']}")
    
    # Feature importance
    if result.feature_importance:
        print(f"\n🎯 TOP 10 FEATURES MAIS IMPORTANTES:")
        top_features = list(result.feature_importance.items())[:10]
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")
    
    # Save ensemble
    ensemble.save_ensemble('ml_ensemble_model.pkl')

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_ml_ensemble())