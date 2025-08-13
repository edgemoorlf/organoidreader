"""
Time Series Analysis Module

This module provides comprehensive time series analysis for organoid tracking,
including temporal tracking algorithms, growth trend analysis, drug response modeling,
and trajectory prediction for longitudinal studies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import interpolate, optimize, stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from organoidreader.core.parameter_extraction import OrganoidParameters
from organoidreader.analysis.viability import ViabilityResults
from organoidreader.analysis.apoptosis import ApoptosisResults

logger = logging.getLogger(__name__)


@dataclass
class TimePoint:
    """Single time point data for an organoid."""
    timestamp: float  # Time in hours or days
    organoid_params: OrganoidParameters
    viability_results: Optional[ViabilityResults] = None
    apoptosis_results: Optional[ApoptosisResults] = None
    treatment_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackingResults:
    """Results of organoid tracking over time."""
    organoid_id: int
    time_points: List[TimePoint]
    growth_metrics: Dict[str, Any]
    viability_trajectory: Optional[Dict[str, Any]] = None
    drug_response: Optional[Dict[str, Any]] = None
    predictions: Optional[Dict[str, Any]] = None
    tracking_quality: float = 0.0


@dataclass
class GrowthModel:
    """Growth model parameters and predictions."""
    model_type: str  # "linear", "exponential", "logistic", "polynomial"
    parameters: Dict[str, float]
    r_squared: float
    growth_rate: float
    doubling_time: Optional[float] = None
    carrying_capacity: Optional[float] = None
    predictions: Dict[str, List[float]] = field(default_factory=dict)


class OrganoidTracker:
    """
    Tracks individual organoids across multiple time points.
    
    Provides functionality for:
    1. Temporal tracking and matching
    2. Growth curve analysis
    3. Treatment response assessment
    4. Trajectory prediction
    """
    
    def __init__(self, matching_threshold: float = 50.0):
        """
        Initialize organoid tracker.
        
        Args:
            matching_threshold: Maximum distance for organoid matching between frames
        """
        self.matching_threshold = matching_threshold
        self.tracked_organoids: Dict[int, List[TimePoint]] = {}
        self.next_id = 1
    
    def add_time_point(self, 
                      timestamp: float,
                      organoid_params_list: List[OrganoidParameters],
                      viability_results: Optional[List[ViabilityResults]] = None,
                      apoptosis_results: Optional[List[ApoptosisResults]] = None,
                      treatment_info: Optional[Dict[str, Any]] = None) -> Dict[int, int]:
        """
        Add a new time point with organoid measurements.
        
        Args:
            timestamp: Time point (hours, days, etc.)
            organoid_params_list: List of organoid parameters
            viability_results: Optional viability results
            apoptosis_results: Optional apoptosis results
            treatment_info: Optional treatment information
            
        Returns:
            Dictionary mapping original organoid IDs to tracking IDs
        """
        logger.info(f"Adding time point {timestamp} with {len(organoid_params_list)} organoids")
        
        # Match organoids to existing tracks
        id_mapping = self._match_organoids_to_tracks(organoid_params_list)
        
        # Create time points
        for i, params in enumerate(organoid_params_list):
            tracking_id = id_mapping[params.label]
            
            # Get corresponding results
            viability = viability_results[i] if viability_results and i < len(viability_results) else None
            apoptosis = apoptosis_results[i] if apoptosis_results and i < len(apoptosis_results) else None
            
            # Create time point
            time_point = TimePoint(
                timestamp=timestamp,
                organoid_params=params,
                viability_results=viability,
                apoptosis_results=apoptosis,
                treatment_info=treatment_info,
                metadata={'original_id': params.label}
            )
            
            # Add to track
            if tracking_id not in self.tracked_organoids:
                self.tracked_organoids[tracking_id] = []
            
            self.tracked_organoids[tracking_id].append(time_point)
        
        logger.info(f"Time point added. Total tracks: {len(self.tracked_organoids)}")
        return id_mapping
    
    def _match_organoids_to_tracks(self, organoid_params_list: List[OrganoidParameters]) -> Dict[int, int]:
        """Match organoids to existing tracks based on position."""
        id_mapping = {}
        
        if not self.tracked_organoids:
            # First time point - create new tracks
            for params in organoid_params_list:
                id_mapping[params.label] = self.next_id
                self.next_id += 1
            return id_mapping
        
        # Get last positions of existing tracks
        last_positions = {}
        for track_id, time_points in self.tracked_organoids.items():
            if time_points:
                last_point = time_points[-1]
                centroid = last_point.organoid_params.spatial.get('centroid_row', 0), \
                          last_point.organoid_params.spatial.get('centroid_col', 0)
                last_positions[track_id] = centroid
        
        # Match current organoids to tracks
        used_tracks = set()
        
        for params in organoid_params_list:
            current_centroid = (params.spatial.get('centroid_row', 0),
                               params.spatial.get('centroid_col', 0))
            
            # Find closest track
            best_track = None
            min_distance = float('inf')
            
            for track_id, last_pos in last_positions.items():
                if track_id in used_tracks:
                    continue
                
                distance = np.sqrt((current_centroid[0] - last_pos[0])**2 + 
                                 (current_centroid[1] - last_pos[1])**2)
                
                if distance < self.matching_threshold and distance < min_distance:
                    min_distance = distance
                    best_track = track_id
            
            if best_track is not None:
                id_mapping[params.label] = best_track
                used_tracks.add(best_track)
            else:
                # Create new track
                id_mapping[params.label] = self.next_id
                self.next_id += 1
        
        return id_mapping
    
    def get_tracking_results(self, organoid_id: int) -> Optional[TrackingResults]:
        """Get tracking results for a specific organoid."""
        if organoid_id not in self.tracked_organoids:
            return None
        
        time_points = sorted(self.tracked_organoids[organoid_id], 
                           key=lambda tp: tp.timestamp)
        
        if len(time_points) < 2:
            logger.warning(f"Organoid {organoid_id} has insufficient time points for analysis")
            return TrackingResults(
                organoid_id=organoid_id,
                time_points=time_points,
                growth_metrics={},
                tracking_quality=0.0
            )
        
        # Calculate growth metrics
        growth_metrics = self._analyze_growth_trajectory(time_points)
        
        # Analyze viability trajectory if available
        viability_trajectory = self._analyze_viability_trajectory(time_points)
        
        # Analyze drug response if treatment info available
        drug_response = self._analyze_drug_response(time_points)
        
        # Generate predictions
        predictions = self._generate_predictions(time_points, growth_metrics)
        
        # Calculate tracking quality
        tracking_quality = self._calculate_tracking_quality(time_points)
        
        return TrackingResults(
            organoid_id=organoid_id,
            time_points=time_points,
            growth_metrics=growth_metrics,
            viability_trajectory=viability_trajectory,
            drug_response=drug_response,
            predictions=predictions,
            tracking_quality=tracking_quality
        )
    
    def _analyze_growth_trajectory(self, time_points: List[TimePoint]) -> Dict[str, Any]:
        """Analyze growth trajectory for an organoid."""
        timestamps = [tp.timestamp for tp in time_points]
        areas = [tp.organoid_params.morphological.get('area_pixels', 0) for tp in time_points]
        
        # Fit different growth models
        models = {}
        
        # Linear model
        models['linear'] = self._fit_linear_model(timestamps, areas)
        
        # Exponential model
        models['exponential'] = self._fit_exponential_model(timestamps, areas)
        
        # Logistic model (if sufficient data points)
        if len(time_points) >= 5:
            models['logistic'] = self._fit_logistic_model(timestamps, areas)
        
        # Polynomial model
        models['polynomial'] = self._fit_polynomial_model(timestamps, areas)
        
        # Select best model based on R²
        best_model_name = max(models.keys(), key=lambda k: models[k].r_squared if models[k] else 0)
        best_model = models[best_model_name]
        
        # Calculate additional growth metrics
        initial_area = areas[0]
        final_area = areas[-1]
        total_growth = final_area - initial_area
        relative_growth = (final_area / initial_area - 1) * 100 if initial_area > 0 else 0
        
        # Calculate average growth rate
        total_time = timestamps[-1] - timestamps[0]
        avg_growth_rate = total_growth / total_time if total_time > 0 else 0
        
        return {
            'models': models,
            'best_model': best_model_name,
            'initial_area': initial_area,
            'final_area': final_area,
            'total_growth': total_growth,
            'relative_growth_percent': relative_growth,
            'average_growth_rate': avg_growth_rate,
            'growth_rate_units': 'pixels/timeunit',
            'measurement_count': len(time_points),
            'time_span': total_time
        }
    
    def _fit_linear_model(self, times: List[float], values: List[float]) -> Optional[GrowthModel]:
        """Fit linear growth model."""
        try:
            X = np.array(times).reshape(-1, 1)
            y = np.array(values)
            
            model = LinearRegression()
            model.fit(X, y)
            
            predictions = model.predict(X)
            r2 = r2_score(y, predictions)
            
            growth_rate = model.coef_[0]
            
            return GrowthModel(
                model_type="linear",
                parameters={'slope': growth_rate, 'intercept': model.intercept_},
                r_squared=r2,
                growth_rate=growth_rate,
                predictions={'times': times, 'values': predictions.tolist()}
            )
        except:
            return None
    
    def _fit_exponential_model(self, times: List[float], values: List[float]) -> Optional[GrowthModel]:
        """Fit exponential growth model."""
        try:
            def exponential_func(t, a, b):
                return a * np.exp(b * t)
            
            times_arr = np.array(times)
            values_arr = np.array(values)
            
            # Initial guess
            p0 = [values_arr[0], 0.1]
            
            popt, _ = optimize.curve_fit(exponential_func, times_arr, values_arr, 
                                       p0=p0, maxfev=1000)
            
            predictions = exponential_func(times_arr, *popt)
            r2 = r2_score(values_arr, predictions)
            
            growth_rate = popt[1]
            doubling_time = np.log(2) / abs(growth_rate) if growth_rate != 0 else None
            
            return GrowthModel(
                model_type="exponential",
                parameters={'a': popt[0], 'b': popt[1]},
                r_squared=r2,
                growth_rate=growth_rate,
                doubling_time=doubling_time,
                predictions={'times': times, 'values': predictions.tolist()}
            )
        except:
            return None
    
    def _fit_logistic_model(self, times: List[float], values: List[float]) -> Optional[GrowthModel]:
        """Fit logistic growth model."""
        try:
            def logistic_func(t, K, P0, r):
                return K / (1 + ((K - P0) / P0) * np.exp(-r * t))
            
            times_arr = np.array(times)
            values_arr = np.array(values)
            
            # Initial guess
            K_guess = max(values_arr) * 1.2  # Carrying capacity
            P0_guess = values_arr[0]  # Initial population
            r_guess = 0.1  # Growth rate
            
            popt, _ = optimize.curve_fit(logistic_func, times_arr, values_arr, 
                                       p0=[K_guess, P0_guess, r_guess], maxfev=1000)
            
            predictions = logistic_func(times_arr, *popt)
            r2 = r2_score(values_arr, predictions)
            
            return GrowthModel(
                model_type="logistic",
                parameters={'K': popt[0], 'P0': popt[1], 'r': popt[2]},
                r_squared=r2,
                growth_rate=popt[2],
                carrying_capacity=popt[0],
                predictions={'times': times, 'values': predictions.tolist()}
            )
        except:
            return None
    
    def _fit_polynomial_model(self, times: List[float], values: List[float]) -> Optional[GrowthModel]:
        """Fit polynomial growth model."""
        try:
            degree = min(3, len(times) - 1)  # Avoid overfitting
            
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(np.array(times).reshape(-1, 1))
            
            model = LinearRegression()
            model.fit(X_poly, values)
            
            predictions = model.predict(X_poly)
            r2 = r2_score(values, predictions)
            
            # Estimate growth rate as derivative at midpoint
            mid_time = (times[0] + times[-1]) / 2
            mid_X = poly_features.transform([[mid_time]])
            
            # Approximate derivative
            delta = 0.1
            X1 = poly_features.transform([[mid_time - delta]])
            X2 = poly_features.transform([[mid_time + delta]])
            growth_rate = (model.predict(X2)[0] - model.predict(X1)[0]) / (2 * delta)
            
            return GrowthModel(
                model_type="polynomial",
                parameters={'degree': degree, 'coefficients': model.coef_.tolist()},
                r_squared=r2,
                growth_rate=growth_rate,
                predictions={'times': times, 'values': predictions.tolist()}
            )
        except:
            return None
    
    def _analyze_viability_trajectory(self, time_points: List[TimePoint]) -> Optional[Dict[str, Any]]:
        """Analyze viability changes over time."""
        viability_data = []
        
        for tp in time_points:
            if tp.viability_results:
                viability_data.append({
                    'timestamp': tp.timestamp,
                    'viability_score': tp.viability_results.viability_score,
                    'live_percentage': tp.viability_results.live_cell_percentage,
                    'classification': tp.viability_results.viability_classification
                })
        
        if len(viability_data) < 2:
            return None
        
        timestamps = [vd['timestamp'] for vd in viability_data]
        viability_scores = [vd['viability_score'] for vd in viability_data]
        
        # Calculate viability trend
        if len(timestamps) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, viability_scores)
            
            trend = "stable"
            if slope > 0.01 and p_value < 0.05:
                trend = "improving"
            elif slope < -0.01 and p_value < 0.05:
                trend = "declining"
        else:
            slope, r_value, p_value = 0, 0, 1
            trend = "stable"
        
        return {
            'data': viability_data,
            'trend': trend,
            'slope': slope,
            'correlation': r_value,
            'p_value': p_value,
            'initial_viability': viability_scores[0],
            'final_viability': viability_scores[-1],
            'viability_change': viability_scores[-1] - viability_scores[0]
        }
    
    def _analyze_drug_response(self, time_points: List[TimePoint]) -> Optional[Dict[str, Any]]:
        """Analyze drug response over time."""
        treatment_points = [tp for tp in time_points if tp.treatment_info]
        
        if len(treatment_points) < 2:
            return None
        
        # Group by treatment conditions
        treatment_groups = {}
        
        for tp in treatment_points:
            treatment_key = str(tp.treatment_info)  # Simple grouping
            if treatment_key not in treatment_groups:
                treatment_groups[treatment_key] = []
            
            treatment_groups[treatment_key].append({
                'timestamp': tp.timestamp,
                'area': tp.organoid_params.morphological.get('area_pixels', 0),
                'viability': tp.viability_results.viability_score if tp.viability_results else None,
                'treatment_info': tp.treatment_info
            })
        
        # Analyze each treatment group
        response_analysis = {}
        
        for treatment, data_points in treatment_groups.items():
            if len(data_points) >= 2:
                times = [dp['timestamp'] for dp in data_points]
                areas = [dp['area'] for dp in data_points]
                
                # Calculate response metrics
                initial_area = areas[0]
                final_area = areas[-1]
                response_ratio = final_area / initial_area if initial_area > 0 else 1.0
                
                # Growth rate under treatment
                slope, _, r_val, p_val, _ = stats.linregress(times, areas)
                
                response_analysis[treatment] = {
                    'initial_area': initial_area,
                    'final_area': final_area,
                    'response_ratio': response_ratio,
                    'growth_rate': slope,
                    'correlation': r_val,
                    'p_value': p_val,
                    'data_points': len(data_points)
                }
        
        return response_analysis
    
    def _generate_predictions(self, time_points: List[TimePoint], growth_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions for future time points."""
        if len(time_points) < 3:
            return {}
        
        best_model = growth_metrics.get('best_model')
        models = growth_metrics.get('models', {})
        
        if best_model not in models or not models[best_model]:
            return {}
        
        model = models[best_model]
        
        # Predict for next few time points
        last_time = time_points[-1].timestamp
        future_times = [last_time + i for i in range(1, 6)]  # Next 5 time units
        
        try:
            future_predictions = []
            
            for t in future_times:
                if model.model_type == "linear":
                    pred = model.parameters['slope'] * t + model.parameters['intercept']
                elif model.model_type == "exponential":
                    pred = model.parameters['a'] * np.exp(model.parameters['b'] * t)
                elif model.model_type == "logistic":
                    K, P0, r = model.parameters['K'], model.parameters['P0'], model.parameters['r']
                    pred = K / (1 + ((K - P0) / P0) * np.exp(-r * t))
                else:
                    pred = None
                
                future_predictions.append(pred)
            
            return {
                'future_times': future_times,
                'predicted_areas': future_predictions,
                'model_used': best_model,
                'confidence': model.r_squared
            }
        except:
            return {}
    
    def _calculate_tracking_quality(self, time_points: List[TimePoint]) -> float:
        """Calculate quality score for tracking."""
        if len(time_points) < 2:
            return 0.0
        
        # Factors affecting tracking quality:
        # 1. Number of time points
        # 2. Temporal consistency
        # 3. Position stability
        
        time_count_score = min(len(time_points) / 10, 1.0)  # Max score at 10 points
        
        # Check temporal spacing consistency
        time_intervals = []
        for i in range(1, len(time_points)):
            interval = time_points[i].timestamp - time_points[i-1].timestamp
            time_intervals.append(interval)
        
        if time_intervals:
            mean_interval = np.mean(time_intervals)
            interval_cv = np.std(time_intervals) / mean_interval if mean_interval > 0 else 1
            temporal_score = max(0, 1.0 - interval_cv)
        else:
            temporal_score = 0.0
        
        # Check position stability (not too much jumping around)
        positions = []
        for tp in time_points:
            centroid = (tp.organoid_params.spatial.get('centroid_row', 0),
                       tp.organoid_params.spatial.get('centroid_col', 0))
            positions.append(centroid)
        
        if len(positions) > 1:
            distances = []
            for i in range(1, len(positions)):
                dist = np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                              (positions[i][1] - positions[i-1][1])**2)
                distances.append(dist)
            
            mean_distance = np.mean(distances)
            position_score = max(0, 1.0 - mean_distance / 100)  # Penalize large movements
        else:
            position_score = 1.0
        
        # Combine scores
        quality = (time_count_score + temporal_score + position_score) / 3
        
        return float(np.clip(quality, 0.0, 1.0))


class TimeSeriesAnalyzer:
    """
    High-level time series analysis for organoid studies.
    
    Provides population-level analysis and comparative studies.
    """
    
    def __init__(self):
        """Initialize time series analyzer."""
        self.trackers: Dict[str, OrganoidTracker] = {}
        self.experiments: Dict[str, Dict[str, Any]] = {}
    
    def create_experiment(self, experiment_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a new experiment for tracking."""
        self.trackers[experiment_id] = OrganoidTracker()
        self.experiments[experiment_id] = metadata or {}
        logger.info(f"Created experiment: {experiment_id}")
    
    def add_experiment_timepoint(self, 
                                experiment_id: str,
                                timestamp: float,
                                organoid_params_list: List[OrganoidParameters],
                                viability_results: Optional[List[ViabilityResults]] = None,
                                apoptosis_results: Optional[List[ApoptosisResults]] = None,
                                treatment_info: Optional[Dict[str, Any]] = None):
        """Add time point data to an experiment."""
        if experiment_id not in self.trackers:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        tracker = self.trackers[experiment_id]
        return tracker.add_time_point(timestamp, organoid_params_list, 
                                    viability_results, apoptosis_results, treatment_info)
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get summary of an experiment."""
        if experiment_id not in self.trackers:
            return {}
        
        tracker = self.trackers[experiment_id]
        
        # Get all tracking results
        all_results = []
        for org_id in tracker.tracked_organoids:
            result = tracker.get_tracking_results(org_id)
            if result:
                all_results.append(result)
        
        if not all_results:
            return {'experiment_id': experiment_id, 'organoid_count': 0}
        
        # Calculate population statistics
        growth_rates = []
        final_areas = []
        viability_trends = []
        
        for result in all_results:
            if result.growth_metrics:
                growth_rates.append(result.growth_metrics.get('average_growth_rate', 0))
                final_areas.append(result.growth_metrics.get('final_area', 0))
            
            if result.viability_trajectory:
                viability_trends.append(result.viability_trajectory.get('viability_change', 0))
        
        summary = {
            'experiment_id': experiment_id,
            'organoid_count': len(all_results),
            'average_growth_rate': float(np.mean(growth_rates)) if growth_rates else 0,
            'growth_rate_std': float(np.std(growth_rates)) if growth_rates else 0,
            'average_final_area': float(np.mean(final_areas)) if final_areas else 0,
            'average_viability_change': float(np.mean(viability_trends)) if viability_trends else 0,
            'tracking_quality_mean': float(np.mean([r.tracking_quality for r in all_results])),
            'metadata': self.experiments[experiment_id]
        }
        
        return summary
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        experiment_summaries = []
        
        for exp_id in experiment_ids:
            summary = self.get_experiment_summary(exp_id)
            if summary:
                experiment_summaries.append(summary)
        
        if len(experiment_summaries) < 2:
            return {'error': 'Need at least 2 experiments for comparison'}
        
        # Statistical comparison
        growth_rates_by_exp = [s['average_growth_rate'] for s in experiment_summaries]
        final_areas_by_exp = [s['average_final_area'] for s in experiment_summaries]
        
        comparison = {
            'experiments': experiment_summaries,
            'growth_rate_comparison': {
                'mean': float(np.mean(growth_rates_by_exp)),
                'std': float(np.std(growth_rates_by_exp)),
                'range': (float(np.min(growth_rates_by_exp)), float(np.max(growth_rates_by_exp)))
            },
            'final_area_comparison': {
                'mean': float(np.mean(final_areas_by_exp)),
                'std': float(np.std(final_areas_by_exp)),
                'range': (float(np.min(final_areas_by_exp)), float(np.max(final_areas_by_exp)))
            }
        }
        
        return comparison


# Utility functions
def track_single_organoid(time_series_data: List[Dict[str, Any]]) -> TrackingResults:
    """
    Convenience function for tracking a single organoid.
    
    Args:
        time_series_data: List of dictionaries with timestamp and organoid data
        
    Returns:
        TrackingResults
    """
    tracker = OrganoidTracker()
    
    for data_point in time_series_data:
        timestamp = data_point['timestamp']
        params = data_point['organoid_params']
        viability = data_point.get('viability_results')
        apoptosis = data_point.get('apoptosis_results')
        treatment = data_point.get('treatment_info')
        
        tracker.add_time_point(timestamp, [params], [viability] if viability else None,
                             [apoptosis] if apoptosis else None, treatment)
    
    # Get results for the single tracked organoid
    organoid_ids = list(tracker.tracked_organoids.keys())
    if organoid_ids:
        return tracker.get_tracking_results(organoid_ids[0])
    
    return None