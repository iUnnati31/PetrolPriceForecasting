import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings

from evaluation_metrics import EvaluationEngine, InvalidValueStrategy

class SimplifiedNotebookEvaluationIntegrator:
    """
    Simplified integration class that provides individual model evaluation
    without comprehensive comparison, visualization, or export functionality.
    """
    
    def __init__(self):
        """Initialize the simplified notebook evaluation integrator."""
        self.evaluation_engine = EvaluationEngine(InvalidValueStrategy.INTERPOLATE)
        
        # Storage for evaluation results
        self.lstm_results = None
        self.arima_results = None
        
        print(f"🚀 Simplified evaluation system initialized")
    
    def integrate_lstm_evaluation(self, 
                                model, 
                                X_train, X_test, 
                                y_train, y_test, 
                                scaler,
                                train_predict=None,
                                test_predict=None) -> Dict:
        """
        Integrate LSTM evaluation after model predictions.
        
        Args:
            model: Trained LSTM model
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            scaler: MinMaxScaler used for data scaling
            train_predict: Pre-computed train predictions (optional)
            test_predict: Pre-computed test predictions (optional)
            
        Returns:
            Dict: LSTM evaluation results
        """
        try:
            print("\n" + "="*60)
            print("🧠 INTEGRATING LSTM EVALUATION")
            print("="*60)
            
            # Generate predictions if not provided
            if train_predict is None:
                print("Generating train predictions...")
                train_predict = model.predict(X_train, verbose=0)
            
            if test_predict is None:
                print("Generating test predictions...")
                test_predict = model.predict(X_test, verbose=0)
            
            # Apply inverse scaling to get original scale predictions
            print("Applying inverse scaling...")
            train_predict_original = scaler.inverse_transform(train_predict)
            test_predict_original = scaler.inverse_transform(test_predict)
            
            # Prepare actual test values
            if y_test.ndim > 1:
                y_test_original = scaler.inverse_transform(y_test)
            else:
                y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Flatten arrays for evaluation
            test_predictions_flat = test_predict_original.flatten()
            y_test_flat = y_test_original.flatten()
            
            # Calculate LSTM metrics
            print("Calculating LSTM performance metrics...")
            lstm_metrics = self.evaluation_engine.calculate_model_metrics(
                test_predictions_flat,
                y_test_flat,
                'LSTM'
            )
            
            # Store LSTM results
            self.lstm_results = {
                'model_name': 'LSTM',
                'test_predictions': test_predictions_flat,
                'train_predictions': train_predict_original.flatten(),
                'actual_values': y_test_flat,
                'metrics': lstm_metrics,
                'model_info': {
                    'type': 'LSTM',
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features': X_train.shape[1] if len(X_train.shape) > 1 else 1,
                    'scaler_used': True
                }
            }
            
            # Display results
            print(f"\n📊 LSTM Evaluation Results:")
            print(f"  MAE:  {lstm_metrics['mae']:.6f}")
            print(f"  RMSE: {lstm_metrics['rmse']:.6f}")
            print(f"  Predictions: {lstm_metrics['prediction_count']} samples")
            print(f"  Data Quality: {lstm_metrics['data_quality_score']:.1f}%")
            
            if 'quality_warning' in lstm_metrics:
                print(f"  ⚠️  Warning: {lstm_metrics['quality_warning']}")
            
            print("✅ LSTM evaluation integrated successfully!")
            
            return self.lstm_results
            
        except Exception as e:
            error_msg = f"Error integrating LSTM evaluation: {str(e)}"
            print(f"❌ {error_msg}")
            warnings.warn(error_msg)
            return {'error': error_msg}
    
    def integrate_arima_evaluation(self, 
                                 df4, 
                                 order=(1,1,0), 
                                 forecast_steps=30,
                                 arima_output=None) -> Dict:
        """
        Integrate ARIMA evaluation after model predictions.
        
        Args:
            df4: Monthly averaged data (pandas Series)
            order: ARIMA order tuple (p,d,q)
            forecast_steps: Number of forecast steps
            arima_output: Pre-computed ARIMA forecast (optional)
            
        Returns:
            Dict: ARIMA evaluation results
        """
        try:
            print("\n" + "="*60)
            print("📈 INTEGRATING ARIMA EVALUATION")
            print("="*60)
            
            # Import ARIMA here to avoid dependency issues
            from statsmodels.tsa.arima.model import ARIMA
            
            # Fit ARIMA model if output not provided
            if arima_output is None:
                print(f"Fitting ARIMA{order} model...")
                model = ARIMA(df4.values, order=order)
                model_fit = model.fit()
                arima_output = model_fit.forecast(steps=forecast_steps)
                print(f"Model AIC: {model_fit.aic:.2f}")
            else:
                print("Using provided ARIMA forecast...")
            
            # Ensure ARIMA predictions are numpy array
            arima_predictions = np.array(arima_output).flatten()
            
            # Use last part of df4 as actual values for evaluation
            actual_values = df4.values[-len(arima_predictions):]
            
            # Calculate ARIMA metrics
            print("Calculating ARIMA performance metrics...")
            arima_metrics = self.evaluation_engine.calculate_model_metrics(
                arima_predictions,
                actual_values,
                'ARIMA'
            )
            
            # Store ARIMA results
            self.arima_results = {
                'model_name': 'ARIMA',
                'test_predictions': arima_predictions,
                'actual_values': actual_values,
                'metrics': arima_metrics,
                'model_info': {
                    'type': 'ARIMA',
                    'order': order,
                    'forecast_steps': forecast_steps,
                    'train_samples': len(df4),
                    'test_samples': len(arima_predictions),
                    'scaler_used': False
                }
            }
            
            # Display results
            print(f"\n📊 ARIMA Evaluation Results:")
            print(f"  MAE:  {arima_metrics['mae']:.6f}")
            print(f"  RMSE: {arima_metrics['rmse']:.6f}")
            print(f"  Predictions: {arima_metrics['prediction_count']} samples")
            print(f"  Data Quality: {arima_metrics['data_quality_score']:.1f}%")
            
            if 'quality_warning' in arima_metrics:
                print(f"  ⚠️  Warning: {arima_metrics['quality_warning']}")
            
            print("✅ ARIMA evaluation integrated successfully!")
            
            return self.arima_results
            
        except Exception as e:
            error_msg = f"Error integrating ARIMA evaluation: {str(e)}"
            print(f"❌ {error_msg}")
            warnings.warn(error_msg)
            return {'error': error_msg}
    
    def get_individual_summaries(self) -> str:
        """
        Get formatted summaries of individual model evaluations.
        
        Returns:
            str: Formatted individual model summaries
        """
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("INDIVIDUAL MODEL EVALUATION SUMMARIES")
        summary_lines.append("=" * 60)
        
        if self.lstm_results:
            lstm_metrics = self.lstm_results['metrics']
            summary_lines.append("\n🧠 LSTM MODEL RESULTS:")
            summary_lines.append("-" * 30)
            summary_lines.append(f"MAE:  {lstm_metrics['mae']:.6f}")
            summary_lines.append(f"RMSE: {lstm_metrics['rmse']:.6f}")
            summary_lines.append(f"Predictions: {lstm_metrics['prediction_count']} samples")
            summary_lines.append(f"Data Quality: {lstm_metrics['data_quality_score']:.1f}%")
        
        if self.arima_results:
            arima_metrics = self.arima_results['metrics']
            summary_lines.append("\n📈 ARIMA MODEL RESULTS:")
            summary_lines.append("-" * 30)
            summary_lines.append(f"MAE:  {arima_metrics['mae']:.6f}")
            summary_lines.append(f"RMSE: {arima_metrics['rmse']:.6f}")
            summary_lines.append(f"Predictions: {arima_metrics['prediction_count']} samples")
            summary_lines.append(f"Data Quality: {arima_metrics['data_quality_score']:.1f}%")
        
        summary_lines.append("\n" + "=" * 60)
        
        return "\n".join(summary_lines)


# Convenience function for simplified integration
def simple_evaluation_integration(model, X_train, X_test, y_train, y_test, scaler, df4, 
                                arima_order=(1,1,0), forecast_steps=30,
                                train_predict=None, test_predict=None, arima_output=None) -> Dict:
    """
    Simplified evaluation integration function for individual model evaluation only.
    
    Args:
        model: Trained LSTM model
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        scaler: MinMaxScaler used for data scaling
        df4: Monthly averaged data for ARIMA
        arima_order: ARIMA order tuple (default: (1,1,0))
        forecast_steps: ARIMA forecast steps (default: 30)
        train_predict: Pre-computed LSTM train predictions (optional)
        test_predict: Pre-computed LSTM test predictions (optional)
        arima_output: Pre-computed ARIMA forecast (optional)
        
    Returns:
        Dict: Individual evaluation results for both models
    """
    
    print("🚀 STARTING SIMPLIFIED EVALUATION INTEGRATION")
    print("="*60)
    
    # Initialize integrator
    integrator = SimplifiedNotebookEvaluationIntegrator()
    
    # Integrate LSTM evaluation
    lstm_results = integrator.integrate_lstm_evaluation(
        model, X_train, X_test, y_train, y_test, scaler, train_predict, test_predict
    )
    
    # Integrate ARIMA evaluation
    arima_results = integrator.integrate_arima_evaluation(
        df4, arima_order, forecast_steps, arima_output
    )
    
    # Display individual summaries
    print("\n" + integrator.get_individual_summaries())
    
    # Return individual results
    return {
        'integrator': integrator,
        'lstm_results': lstm_results,
        'arima_results': arima_results,
        'summary': integrator.get_individual_summaries()
    }


if __name__ == "__main__":
    print("Simplified Notebook Evaluation Integration Module")
    print("="*60)
    print("This module provides individual model evaluation without")
    print("comprehensive comparison, visualization, or export functionality.")
    print("\nKey Functions:")
    print("- SimplifiedNotebookEvaluationIntegrator: Main integration class")
    print("- simple_evaluation_integration(): One-function individual evaluation")