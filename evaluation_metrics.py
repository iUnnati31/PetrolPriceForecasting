import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional, Union, Literal
from enum import Enum
import warnings

class InvalidValueStrategy(Enum):
    """Strategies for handling invalid values (NaN, infinity) in time series data."""
    INTERPOLATE = "interpolate" 
    REMOVE = "remove"           
    FORWARD_FILL = "ffill"     
    MEAN_FILL = "mean_fill"    
    RAISE_ERROR = "error"      

class EvaluationEngine:
    """
    Core evaluation engine for calculating and comparing forecasting model performance.
    Implements MAE and RMSE metrics with proper error handling and data alignment.
    Uses interpolation by default to preserve temporal alignment in time series data.
    """
    
    def __init__(self, invalid_strategy: InvalidValueStrategy = InvalidValueStrategy.INTERPOLATE):
        """
        Initialize the evaluation engine.
        
        Args:
            invalid_strategy: Strategy for handling invalid values (default: interpolate)
        """
        self.invalid_strategy = invalid_strategy
        self.metrics_history = []
        
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict]:
        """
        Calculate Mean Absolute Error (MAE) between actual and predicted values.
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
            
        Returns:
            Tuple of (mae_value, metadata_dict)
            
        Raises:
            ValueError: If arrays have different lengths or contain invalid data
        """
        try:
            # Validate and handle invalid values
            y_true_clean, y_pred_clean, metadata = self._handle_invalid_values(y_true, y_pred)
            
            if len(y_true_clean) == 0:
                return np.nan, {**metadata, 'error': 'No valid data points'}
            
            mae = np.mean(np.abs(y_true_clean - y_pred_clean))
            
            return float(mae), metadata
            
        except Exception as e:
            return np.nan, {'error': f"Error calculating MAE: {str(e)}", 'original_length': len(y_true)}
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict]:
        """
        Calculate Root Mean Square Error (RMSE) between actual and predicted values.
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
            
        Returns:
            Tuple of (rmse_value, metadata_dict)
            
        Raises:
            ValueError: If arrays have different lengths or contain invalid data
        """
        try:
            # Validate and handle invalid values
            y_true_clean, y_pred_clean, metadata = self._handle_invalid_values(y_true, y_pred)
            
            if len(y_true_clean) == 0:
                return np.nan, {**metadata, 'error': 'No valid data points'}
            
            # Calculate RMSE
            mse = np.mean((y_true_clean - y_pred_clean) ** 2)
            rmse = np.sqrt(mse)
            
            return float(rmse), metadata
            
        except Exception as e:
            return np.nan, {'error': f"Error calculating RMSE: {str(e)}", 'original_length': len(y_true)}
    
    def prepare_predictions(self, 
                          model_predictions: np.ndarray, 
                          scaler: Optional[object] = None,
                          model_name: str = "Unknown") -> np.ndarray:
        """
        Prepare model predictions by applying inverse scaling and ensuring proper format.
        
        Args:
            model_predictions: Raw model predictions
            scaler: Sklearn scaler object for inverse transformation (optional)
            model_name: Name of the model for logging purposes
            
        Returns:
            np.ndarray: Processed predictions ready for evaluation
            
        Raises:
            ValueError: If predictions cannot be processed
        """
        try:
         
            if not isinstance(model_predictions, np.ndarray):
                predictions = np.array(model_predictions)
            else:
                predictions = model_predictions.copy()
            
            # Handle different array shapes
            if predictions.ndim > 1:
                # If 2D array, flatten it
                if predictions.shape[1] == 1:
                    predictions = predictions.flatten()
                else:
                    warnings.warn(f"Unexpected prediction shape for {model_name}: {predictions.shape}")
            
            # Apply inverse scaling 
            if scaler is not None:
                try:
                    # Reshape for scaler if needed
                    if predictions.ndim == 1:
                        predictions_reshaped = predictions.reshape(-1, 1)
                    else:
                        predictions_reshaped = predictions
                    
                    # Apply inverse transform
                    predictions = scaler.inverse_transform(predictions_reshaped)
                    
                    if predictions.ndim > 1 and predictions.shape[1] == 1:
                        predictions = predictions.flatten()
                        
                except Exception as e:
                    warnings.warn(f"Could not apply inverse scaling for {model_name}: {str(e)}")
            
            # Remove any NaN or infinite values
            predictions = self._clean_array(predictions, f"{model_name} predictions")
            
            return predictions
            
        except Exception as e:
            raise ValueError(f"Error preparing predictions for {model_name}: {str(e)}")
    
    def align_predictions(self, 
                         lstm_predictions: np.ndarray,
                         arima_predictions: np.ndarray,
                         test_actual: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align LSTM and ARIMA predictions with test data to ensure fair comparison.
        
        Args:
            lstm_predictions: LSTM model predictions
            arima_predictions: ARIMA model predictions  
            test_actual: Actual test values
            
        Returns:
            Tuple of aligned (lstm_pred, arima_pred, actual) arrays
            
        Raises:
            ValueError: If arrays cannot be properly aligned
        """
        try:
            # Convert all to numpy arrays
            lstm_pred = np.array(lstm_predictions).flatten()
            arima_pred = np.array(arima_predictions).flatten()
            actual = np.array(test_actual).flatten()
            
            # Find the minimum length to align all arrays
            min_length = min(len(lstm_pred), len(arima_pred), len(actual))
            
            if min_length == 0:
                raise ValueError("One or more prediction arrays is empty")
            
            # Truncate all arrays to the same length
            lstm_aligned = lstm_pred[:min_length]
            arima_aligned = arima_pred[:min_length]
            actual_aligned = actual[:min_length]
            
            # Clean arrays
            lstm_aligned = self._clean_array(lstm_aligned, "LSTM predictions")
            arima_aligned = self._clean_array(arima_aligned, "ARIMA predictions")
            actual_aligned = self._clean_array(actual_aligned, "Actual values")
            
            # Final length check after cleaning
            final_length = min(len(lstm_aligned), len(arima_aligned), len(actual_aligned))
            
            return (lstm_aligned[:final_length], 
                   arima_aligned[:final_length], 
                   actual_aligned[:final_length])
            
        except Exception as e:
            raise ValueError(f"Error aligning predictions: {str(e)}")
    
    def calculate_model_metrics(self, 
                              predictions: np.ndarray, 
                              actual: np.ndarray,
                              model_name: str) -> Dict:
        """
        Calculate comprehensive metrics for a single model.
        
        Args:
            predictions: Model predictions
            actual: Actual values
            model_name: Name of the model
            
        Returns:
            Dict: Dictionary containing calculated metrics and data quality info
        """
        try:
            # Calculate metrics with quality metadata
            mae, mae_metadata = self.calculate_mae(actual, predictions)
            rmse, rmse_metadata = self.calculate_rmse(actual, predictions)
            
            # Create comprehensive metrics dictionary
            metrics = {
                'model_name': model_name,
                'mae': mae,
                'rmse': rmse,
                'prediction_count': mae_metadata.get('final_length', 0),
                'original_count': mae_metadata.get('original_length', 0),
                'invalid_count': mae_metadata.get('invalid_count', 0),
                'data_quality_score': self._calculate_quality_score(mae_metadata),
                'strategy_used': mae_metadata.get('strategy_used', 'none'),
                'calculation_timestamp': datetime.now(),
                'metadata': mae_metadata
            }
            
            # Add quality warnings if needed
            if metrics['invalid_count'] > 0:
                quality_pct = (metrics['invalid_count'] / metrics['original_count']) * 100
                if quality_pct > 15:
                    metrics['quality_warning'] = f"High invalid data rate: {quality_pct:.1f}%"
                elif quality_pct > 5:
                    metrics['quality_warning'] = f"Moderate invalid data rate: {quality_pct:.1f}%"
            
            # Store in history
            self.metrics_history.append(metrics.copy())
            
            return metrics
            
        except Exception as e:
            return {
                'model_name': model_name,
                'mae': np.nan,
                'rmse': np.nan,
                'error': f"Error calculating metrics for {model_name}: {str(e)}",
                'calculation_timestamp': datetime.now()
            }
    
    def _handle_invalid_values(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Handle invalid values according to the specified strategy.
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
            
        Returns:
            Tuple of (cleaned_y_true, cleaned_y_pred, metadata)
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"Array length mismatch: actual={len(y_true)}, predicted={len(y_pred)}")
        
        # Check for invalid values
        mask_true = np.isfinite(y_true)
        mask_pred = np.isfinite(y_pred)
        valid_mask = mask_true & mask_pred
        
        invalid_count = np.sum(~valid_mask)
        metadata = {
            'original_length': len(y_true),
            'invalid_count': invalid_count,
            'strategy_used': self.invalid_strategy.value
        }
        
        if invalid_count == 0:
            metadata['final_length'] = len(y_true)
            return y_true, y_pred, metadata
        
        # Handle invalid values based on strategy
        if self.invalid_strategy == InvalidValueStrategy.INTERPOLATE:
            return self._interpolate_invalid(y_true, y_pred, valid_mask, metadata)
        elif self.invalid_strategy == InvalidValueStrategy.REMOVE:
            return self._remove_invalid(y_true, y_pred, valid_mask, metadata)
        elif self.invalid_strategy == InvalidValueStrategy.FORWARD_FILL:
            return self._forward_fill_invalid(y_true, y_pred, valid_mask, metadata)
        elif self.invalid_strategy == InvalidValueStrategy.MEAN_FILL:
            return self._mean_fill_invalid(y_true, y_pred, valid_mask, metadata)
        elif self.invalid_strategy == InvalidValueStrategy.RAISE_ERROR:
            raise ValueError(f"Found {invalid_count} invalid values. Strategy is set to raise error.")
        else:
            # Default to interpolation for time series
            return self._interpolate_invalid(y_true, y_pred, valid_mask, metadata)
    
    def _interpolate_invalid(self, y_true: np.ndarray, y_pred: np.ndarray,
                           valid_mask: np.ndarray, metadata: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Interpolate invalid values to preserve temporal alignment."""
        if metadata['invalid_count'] > 0:
            warnings.warn(f"Interpolating {metadata['invalid_count']} invalid values to preserve temporal alignment")
        
        y_true_clean = y_true.copy()
        y_pred_clean = y_pred.copy()
        
        # Interpolate invalid values in y_true
        if not np.all(np.isfinite(y_true)):
            y_true_clean = self._interpolate_array(y_true)
        
        # Interpolate invalid values in y_pred  
        if not np.all(np.isfinite(y_pred)):
            y_pred_clean = self._interpolate_array(y_pred)
        
        metadata['final_length'] = len(y_true_clean)
        metadata['interpolated'] = True
        
        return y_true_clean, y_pred_clean, metadata
    
    def _remove_invalid(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       valid_mask: np.ndarray, metadata: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Remove invalid value pairs (may cause temporal misalignment)."""
        warnings.warn(f"Removing {metadata['invalid_count']} invalid value pairs - this may cause temporal misalignment")
        
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        metadata['final_length'] = len(y_true_clean)
        metadata['data_loss_percentage'] = (metadata['invalid_count'] / metadata['original_length']) * 100
        
        return y_true_clean, y_pred_clean, metadata
    
    def _forward_fill_invalid(self, y_true: np.ndarray, y_pred: np.ndarray,
                            valid_mask: np.ndarray, metadata: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Forward fill invalid values."""
        warnings.warn(f"Forward filling {metadata['invalid_count']} invalid values")
        
        y_true_clean = self._forward_fill_array(y_true)
        y_pred_clean = self._forward_fill_array(y_pred)
        
        metadata['final_length'] = len(y_true_clean)
        metadata['forward_filled'] = True
        
        return y_true_clean, y_pred_clean, metadata
    
    def _mean_fill_invalid(self, y_true: np.ndarray, y_pred: np.ndarray,
                         valid_mask: np.ndarray, metadata: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Fill invalid values with mean of valid values."""
        warnings.warn(f"Mean filling {metadata['invalid_count']} invalid values")
        
        # Calculate means of valid values
        true_mean = np.nanmean(y_true)
        pred_mean = np.nanmean(y_pred)
        
        y_true_clean = np.where(np.isfinite(y_true), y_true, true_mean)
        y_pred_clean = np.where(np.isfinite(y_pred), y_pred, pred_mean)
        
        metadata['final_length'] = len(y_true_clean)
        metadata['mean_filled'] = True
        metadata['true_fill_value'] = true_mean
        metadata['pred_fill_value'] = pred_mean
        
        return y_true_clean, y_pred_clean, metadata
    
    def _interpolate_array(self, arr: np.ndarray) -> np.ndarray:
        """Interpolate missing values in an array using linear interpolation."""
        arr_clean = arr.copy()
        
        # Find valid indices
        valid_indices = np.where(np.isfinite(arr))[0]
        
        if len(valid_indices) < 2:
            # Not enough valid points for interpolation, use mean
            mean_val = np.nanmean(arr)
            if np.isfinite(mean_val):
                arr_clean = np.where(np.isfinite(arr), arr, mean_val)
            return arr_clean
        
        # Interpolate invalid values
        invalid_indices = np.where(~np.isfinite(arr))[0]
        
        for idx in invalid_indices:
            # Find nearest valid values
            left_idx = valid_indices[valid_indices < idx]
            right_idx = valid_indices[valid_indices > idx]
            
            if len(left_idx) > 0 and len(right_idx) > 0:
                # Linear interpolation between left and right
                left_val = arr[left_idx[-1]]
                right_val = arr[right_idx[0]]
                left_pos = left_idx[-1]
                right_pos = right_idx[0]
                
                # Linear interpolation formula
                weight = (idx - left_pos) / (right_pos - left_pos)
                arr_clean[idx] = left_val + weight * (right_val - left_val)
                
            elif len(left_idx) > 0:
                # Use last valid value (forward fill)
                arr_clean[idx] = arr[left_idx[-1]]
            elif len(right_idx) > 0:
                # Use next valid value (backward fill)
                arr_clean[idx] = arr[right_idx[0]]
        
        return arr_clean
    
    def _forward_fill_array(self, arr: np.ndarray) -> np.ndarray:
        """Forward fill missing values in an array."""
        arr_clean = arr.copy()
        
        last_valid = None
        for i in range(len(arr)):
            if np.isfinite(arr[i]):
                last_valid = arr[i]
            elif last_valid is not None:
                arr_clean[i] = last_valid
        
        # If there are still NaN values at the beginning, use the first valid value
        first_valid_idx = np.where(np.isfinite(arr_clean))[0]
        if len(first_valid_idx) > 0:
            first_valid = arr_clean[first_valid_idx[0]]
            arr_clean[:first_valid_idx[0]] = first_valid
        
        return arr_clean
    
    def _calculate_quality_score(self, metadata: Dict) -> float:
        """Calculate a data quality score (0-100) based on metadata."""
        if 'original_length' not in metadata or metadata['original_length'] == 0:
            return 0.0
        
        valid_ratio = 1 - (metadata.get('invalid_count', 0) / metadata['original_length'])
        return valid_ratio * 100
    
    def _clean_array(self, arr: np.ndarray, name: str) -> np.ndarray:
        """
        Clean array by removing NaN and infinite values.
        
        Args:
            arr: Input array
            name: Name for error reporting
            
        Returns:
            np.ndarray: Cleaned array
            
        Raises:
            ValueError: If array becomes empty after cleaning
        """
        # Check for NaN or infinite values
        mask = np.isfinite(arr)
        
        if not np.all(mask):
            invalid_count = np.sum(~mask)
            warnings.warn(f"Found {invalid_count} invalid values in {name}, removing them")
            arr = arr[mask]
        
        if len(arr) == 0:
            raise ValueError(f"Array {name} is empty after cleaning invalid values")
        
        return arr


class DataAlignmentUtility:
    """
    Utility class for aligning different model predictions with test periods.
    Ensures LSTM and ARIMA predictions match test periods for fair comparison.
    """
    
    @staticmethod
    def extract_test_period_data(test_data: np.ndarray, 
                               start_idx: int = 0, 
                               length: Optional[int] = None) -> np.ndarray:
        """
        Extract test period data for evaluation.
        
        Args:
            test_data: Full test dataset
            start_idx: Starting index for extraction
            length: Length of data to extract (optional)
            
        Returns:
            np.ndarray: Extracted test data
        """
        test_array = np.array(test_data).flatten()
        
        if length is None:
            return test_array[start_idx:]
        else:
            end_idx = start_idx + length
            return test_array[start_idx:end_idx]
    
    @staticmethod
    def match_prediction_lengths(lstm_pred: np.ndarray, 
                               arima_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match the lengths of LSTM and ARIMA predictions.
        
        Args:
            lstm_pred: LSTM predictions
            arima_pred: ARIMA predictions
            
        Returns:
            Tuple of matched prediction arrays
        """
        lstm_array = np.array(lstm_pred).flatten()
        arima_array = np.array(arima_pred).flatten()
        
        min_length = min(len(lstm_array), len(arima_array))
        
        return lstm_array[:min_length], arima_array[:min_length]
    
    @staticmethod
    def prepare_inverse_scaling(predictions: np.ndarray, 
                              scaler: object) -> np.ndarray:
        """
        Prepare predictions for inverse scaling transformation.
        
        Args:
            predictions: Raw predictions
            scaler: Fitted scaler object
            
        Returns:
            np.ndarray: Inverse scaled predictions
        """
        try:
            # Ensure proper shape for scaler
            if predictions.ndim == 1:
                predictions_reshaped = predictions.reshape(-1, 1)
            else:
                predictions_reshaped = predictions
            
            # Apply inverse transform
            inverse_scaled = scaler.inverse_transform(predictions_reshaped)
            
            # Return flattened array
            return inverse_scaled.flatten()
            
        except Exception as e:
            raise ValueError(f"Error in inverse scaling: {str(e)}")


# Convenience functions for easy integration
def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray, 
                 strategy: InvalidValueStrategy = InvalidValueStrategy.INTERPOLATE) -> Tuple[float, Dict]:
    """
    Convenience function to calculate MAE with data quality reporting.
    
    Returns:
        Tuple of (mae_value, metadata_dict)
    """
    engine = EvaluationEngine(strategy)
    return engine.calculate_mae(y_true, y_pred)

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray,
                  strategy: InvalidValueStrategy = InvalidValueStrategy.INTERPOLATE) -> Tuple[float, Dict]:
    """
    Convenience function to calculate RMSE with data quality reporting.
    
    Returns:
        Tuple of (rmse_value, metadata_dict)
    """
    engine = EvaluationEngine(strategy)
    return engine.calculate_rmse(y_true, y_pred)

def prepare_predictions(model_predictions: np.ndarray, 
                       scaler: Optional[object] = None,
                       model_name: str = "Unknown",
                       strategy: InvalidValueStrategy = InvalidValueStrategy.INTERPOLATE) -> np.ndarray:
    """Convenience function to prepare predictions."""
    engine = EvaluationEngine(strategy)
    return engine.prepare_predictions(model_predictions, scaler, model_name)

class PerformanceComparator:
    """
    Performance comparison engine for evaluating LSTM vs ARIMA model performance.
    Calculates improvement percentages and determines best performing model.
    """
    
    def __init__(self):
        """Initialize the performance comparator."""
        self.comparison_history = []
        
    def calculate_improvement_percentage(self, 
                                       lstm_metric: float, 
                                       arima_metric: float) -> float:
        """
        Calculate percentage improvement of LSTM over ARIMA for a given metric.
        
        Args:
            lstm_metric: LSTM model metric value (MAE or RMSE)
            arima_metric: ARIMA model metric value (MAE or RMSE)
            
        Returns:
            float: Percentage improvement (positive means LSTM is better)
            
        Note:
            For error metrics (MAE, RMSE), lower values are better.
            Improvement = ((arima_metric - lstm_metric) / arima_metric) * 100
        """
        try:
            # Validate inputs
            if not np.isfinite(lstm_metric) or not np.isfinite(arima_metric):
                return np.nan
            
            if arima_metric == 0:
                # Handle division by zero
                if lstm_metric == 0:
                    return 0.0  # Both metrics are zero, no improvement
                else:
                    return -np.inf  # ARIMA is perfect, LSTM is worse
            
            # Calculate improvement percentage
            # For error metrics, lower is better, so improvement is:
            # (baseline - new) / baseline * 100
            improvement = ((arima_metric - lstm_metric) / arima_metric) * 100
            
            return float(improvement)
            
        except Exception as e:
            warnings.warn(f"Error calculating improvement percentage: {str(e)}")
            return np.nan
    
    def determine_best_model(self, 
                           lstm_metrics: Dict, 
                           arima_metrics: Dict) -> Tuple[str, Dict]:
        """
        Determine which model performs better based on MAE and RMSE metrics.
        
        Args:
            lstm_metrics: Dictionary containing LSTM MAE and RMSE
            arima_metrics: Dictionary containing ARIMA MAE and RMSE
            
        Returns:
            Tuple of (best_model_name, comparison_details)
        """
        try:
            # Extract metrics
            lstm_mae = lstm_metrics.get('mae', np.nan)
            lstm_rmse = lstm_metrics.get('rmse', np.nan)
            arima_mae = arima_metrics.get('mae', np.nan)
            arima_rmse = arima_metrics.get('rmse', np.nan)
            
            # Calculate improvements
            mae_improvement = self.calculate_improvement_percentage(lstm_mae, arima_mae)
            rmse_improvement = self.calculate_improvement_percentage(lstm_rmse, arima_rmse)
            
            # Determine winner based on both metrics
            lstm_wins = 0
            arima_wins = 0
            
            # MAE comparison
            if np.isfinite(mae_improvement):
                if mae_improvement > 0:
                    lstm_wins += 1
                elif mae_improvement < 0:
                    arima_wins += 1
            
            # RMSE comparison
            if np.isfinite(rmse_improvement):
                if rmse_improvement > 0:
                    lstm_wins += 1
                elif rmse_improvement < 0:
                    arima_wins += 1
            
            # Determine overall winner
            if lstm_wins > arima_wins:
                best_model = "LSTM"
                confidence = "High" if lstm_wins == 2 else "Medium"
            elif arima_wins > lstm_wins:
                best_model = "ARIMA"
                confidence = "High" if arima_wins == 2 else "Medium"
            else:
                best_model = "Tie"
                confidence = "Low"
            
            # Create detailed comparison
            comparison_details = {
                'mae_improvement_pct': mae_improvement,
                'rmse_improvement_pct': rmse_improvement,
                'lstm_wins_count': lstm_wins,
                'arima_wins_count': arima_wins,
                'confidence_level': confidence,
                'lstm_mae': lstm_mae,
                'lstm_rmse': lstm_rmse,
                'arima_mae': arima_mae,
                'arima_rmse': arima_rmse,
                'comparison_timestamp': datetime.now()
            }
            
            return best_model, comparison_details
            
        except Exception as e:
            return "Error", {'error': f"Error determining best model: {str(e)}"}
    
    def check_target_achievement(self, 
                               mae_improvement: float, 
                               rmse_improvement: float,
                               target_min: float = 20.0,
                               target_max: float = 30.0) -> Dict:
        """
        Check if LSTM achieves target improvement percentages (20-30% by default).
        
        Args:
            mae_improvement: MAE improvement percentage
            rmse_improvement: RMSE improvement percentage
            target_min: Minimum target improvement percentage (default: 20%)
            target_max: Maximum target improvement percentage (default: 30%)
            
        Returns:
            Dict: Target achievement analysis
        """
        try:
            # Initialize results
            results = {
                'target_min': target_min,
                'target_max': target_max,
                'mae_improvement': mae_improvement,
                'rmse_improvement': rmse_improvement,
                'mae_meets_min_target': False,
                'rmse_meets_min_target': False,
                'mae_meets_max_target': False,
                'rmse_meets_max_target': False,
                'overall_target_achievement': 'None',
                'achievement_details': []
            }
            
            # Check MAE target achievement
            if np.isfinite(mae_improvement):
                results['mae_meets_min_target'] = mae_improvement >= target_min
                results['mae_meets_max_target'] = mae_improvement >= target_max
                
                if mae_improvement >= target_max:
                    results['achievement_details'].append(f"MAE exceeds maximum target ({mae_improvement:.1f}% >= {target_max}%)")
                elif mae_improvement >= target_min:
                    results['achievement_details'].append(f"MAE meets minimum target ({mae_improvement:.1f}% >= {target_min}%)")
                else:
                    results['achievement_details'].append(f"MAE below target ({mae_improvement:.1f}% < {target_min}%)")
            else:
                results['achievement_details'].append("MAE improvement could not be calculated")
            
            # Check RMSE target achievement
            if np.isfinite(rmse_improvement):
                results['rmse_meets_min_target'] = rmse_improvement >= target_min
                results['rmse_meets_max_target'] = rmse_improvement >= target_max
                
                if rmse_improvement >= target_max:
                    results['achievement_details'].append(f"RMSE exceeds maximum target ({rmse_improvement:.1f}% >= {target_max}%)")
                elif rmse_improvement >= target_min:
                    results['achievement_details'].append(f"RMSE meets minimum target ({rmse_improvement:.1f}% >= {target_min}%)")
                else:
                    results['achievement_details'].append(f"RMSE below target ({rmse_improvement:.1f}% < {target_min}%)")
            else:
                results['achievement_details'].append("RMSE improvement could not be calculated")
            
            # Determine overall achievement level
            min_targets_met = results['mae_meets_min_target'] and results['rmse_meets_min_target']
            max_targets_met = results['mae_meets_max_target'] and results['rmse_meets_max_target']
            
            if max_targets_met:
                results['overall_target_achievement'] = 'Exceeds Maximum Target'
            elif min_targets_met:
                results['overall_target_achievement'] = 'Meets Minimum Target'
            elif results['mae_meets_min_target'] or results['rmse_meets_min_target']:
                results['overall_target_achievement'] = 'Partial Target Achievement'
            else:
                results['overall_target_achievement'] = 'Below Target'
            
            return results
            
        except Exception as e:
            return {
                'error': f"Error checking target achievement: {str(e)}",
                'overall_target_achievement': 'Error'
            }
    
    def generate_comparison_summary(self, 
                                  lstm_metrics: Dict, 
                                  arima_metrics: Dict,
                                  target_min: float = 20.0,
                                  target_max: float = 30.0) -> Dict:
        """
        Generate comprehensive comparison summary between LSTM and ARIMA models.
        
        Args:
            lstm_metrics: LSTM model metrics dictionary
            arima_metrics: ARIMA model metrics dictionary
            target_min: Minimum target improvement percentage
            target_max: Maximum target improvement percentage
            
        Returns:
            Dict: Comprehensive comparison results
        """
        try:
            # Calculate improvements
            mae_improvement = self.calculate_improvement_percentage(
                lstm_metrics.get('mae', np.nan), 
                arima_metrics.get('mae', np.nan)
            )
            rmse_improvement = self.calculate_improvement_percentage(
                lstm_metrics.get('rmse', np.nan), 
                arima_metrics.get('rmse', np.nan)
            )
            
            # Determine best model
            best_model, model_comparison = self.determine_best_model(lstm_metrics, arima_metrics)
            
            # Check target achievement
            target_results = self.check_target_achievement(
                mae_improvement, rmse_improvement, target_min, target_max
            )
            
            # Create comprehensive summary
            summary = {
                'lstm_metrics': lstm_metrics,
                'arima_metrics': arima_metrics,
                'mae_improvement_pct': mae_improvement,
                'rmse_improvement_pct': rmse_improvement,
                'best_model': best_model,
                'model_comparison_details': model_comparison,
                'target_achievement': target_results,
                'meets_target_improvement': target_results.get('overall_target_achievement') in [
                    'Meets Minimum Target', 'Exceeds Maximum Target'
                ],
                'summary_timestamp': datetime.now(),
                'data_quality_notes': []
            }
            
            # Add data quality notes
            for model_name, metrics in [('LSTM', lstm_metrics), ('ARIMA', arima_metrics)]:
                if 'quality_warning' in metrics:
                    summary['data_quality_notes'].append(f"{model_name}: {metrics['quality_warning']}")
                
                quality_score = metrics.get('data_quality_score', 100)
                if quality_score < 90:
                    summary['data_quality_notes'].append(
                        f"{model_name} data quality score: {quality_score:.1f}%"
                    )
            
            # Store in history
            self.comparison_history.append(summary.copy())
            
            return summary
            
        except Exception as e:
            return {
                'error': f"Error generating comparison summary: {str(e)}",
                'summary_timestamp': datetime.now()
            }


# Convenience functions for performance comparison
def calculate_improvement_percentage(lstm_metric: float, arima_metric: float) -> float:
    """
    Convenience function to calculate improvement percentage.
    
    Args:
        lstm_metric: LSTM model metric value
        arima_metric: ARIMA model metric value
        
    Returns:
        float: Percentage improvement (positive means LSTM is better)
    """
    comparator = PerformanceComparator()
    return comparator.calculate_improvement_percentage(lstm_metric, arima_metric)

def determine_best_model(lstm_metrics: Dict, arima_metrics: Dict) -> Tuple[str, Dict]:
    """
    Convenience function to determine which model performs better.
    
    Returns:
        Tuple of (best_model_name, comparison_details)
    """
    comparator = PerformanceComparator()
    return comparator.determine_best_model(lstm_metrics, arima_metrics)

def check_target_achievement(mae_improvement: float, 
                           rmse_improvement: float,
                           target_min: float = 20.0,
                           target_max: float = 30.0) -> Dict:
    """
    Convenience function to check target achievement.
    
    Returns:
        Dict: Target achievement analysis
    """
    comparator = PerformanceComparator()
    return comparator.check_target_achievement(mae_improvement, rmse_improvement, target_min, target_max)

def generate_comparison_summary(lstm_metrics: Dict, 
                              arima_metrics: Dict,
                              target_min: float = 20.0,
                              target_max: float = 30.0) -> Dict:
    """
    Convenience function to generate comprehensive comparison summary.
    
    Returns:
        Dict: Comprehensive comparison results
    """
    comparator = PerformanceComparator()
    return comparator.generate_comparison_summary(lstm_metrics, arima_metrics, target_min, target_max)

# Legacy functions for backward compatibility (return only the metric value)
def calculate_mae_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Legacy function that returns only MAE value (for backward compatibility)."""
    mae, _ = calculate_mae(y_true, y_pred)
    return mae

def calculate_rmse_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Legacy function that returns only RMSE value (for backward compatibility)."""
    rmse, _ = calculate_rmse(y_true, y_pred)
    return rmse