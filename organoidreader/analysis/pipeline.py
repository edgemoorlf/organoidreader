"""
Advanced Analysis Pipeline Integration

This module integrates all advanced analysis components (viability, apoptosis, time series)
into a unified pipeline for comprehensive organoid analysis.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

from organoidreader.core.image_loader import ImageLoader
from organoidreader.core.preprocessing import ImagePreprocessor
from organoidreader.core.segmentation import SegmentationEngine
from organoidreader.core.parameter_extraction import ParameterExtractor, OrganoidParameters
from organoidreader.analysis.viability import ViabilityAnalyzer, ViabilityResults
from organoidreader.analysis.apoptosis import ApoptosisDetector, ApoptosisResults
from organoidreader.analysis.time_series import TimeSeriesAnalyzer, TimePoint
from organoidreader.config.config_manager import Config

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResults:
    """Comprehensive analysis results for a single image."""
    image_path: str
    timestamp: Optional[float]
    organoid_count: int
    organoid_parameters: List[OrganoidParameters]
    viability_results: List[ViabilityResults]
    apoptosis_results: List[ApoptosisResults]
    summary_statistics: Dict[str, Any]
    quality_score: float
    processing_metadata: Dict[str, Any]


@dataclass
class ExperimentResults:
    """Results for an entire experiment with multiple time points."""
    experiment_id: str
    analysis_results: List[AnalysisResults]
    time_series_data: Dict[str, Any]
    comparative_analysis: Optional[Dict[str, Any]] = None
    experiment_metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedAnalysisPipeline:
    """
    Unified pipeline for comprehensive organoid analysis.
    
    Integrates all analysis components:
    1. Basic segmentation and parameter extraction
    2. Viability analysis
    3. Apoptosis detection
    4. Time series tracking
    5. Comparative analysis
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 viability_model_path: Optional[str] = None,
                 segmentation_model_path: Optional[str] = None):
        """
        Initialize advanced analysis pipeline.
        
        Args:
            config: Configuration object
            viability_model_path: Path to trained viability model
            segmentation_model_path: Path to trained segmentation model
        """
        self.config = config or Config()
        
        # Initialize core components
        self.image_loader = ImageLoader()
        self.preprocessor = ImagePreprocessor()
        self.segmentation_engine = SegmentationEngine()
        self.parameter_extractor = ParameterExtractor()
        
        # Initialize advanced analysis components
        self.viability_analyzer = ViabilityAnalyzer(viability_model_path)
        self.apoptosis_detector = ApoptosisDetector()
        self.time_series_analyzer = TimeSeriesAnalyzer()
        
        # Load segmentation model if provided
        if segmentation_model_path:
            self.segmentation_engine.load_model(segmentation_model_path)
        else:
            # Load default model (creates untrained model)
            self.segmentation_engine.load_model()
        
        logger.info("Advanced analysis pipeline initialized")
    
    def analyze_single_image(self, 
                           image_path: str,
                           timestamp: Optional[float] = None,
                           nuclear_channel_path: Optional[str] = None,
                           treatment_info: Optional[Dict[str, Any]] = None) -> AnalysisResults:
        """
        Perform comprehensive analysis on a single image.
        
        Args:
            image_path: Path to the image file
            timestamp: Optional timestamp for time series analysis
            nuclear_channel_path: Optional path to nuclear staining channel
            treatment_info: Optional treatment information
            
        Returns:
            AnalysisResults with comprehensive analysis
        """
        logger.info(f"Analyzing single image: {image_path}")
        
        try:
            # Load and preprocess image
            image, image_metadata = self.image_loader.load_image(image_path)
            
            # Load nuclear channel if provided
            nuclear_channel = None
            if nuclear_channel_path:
                nuclear_channel, _ = self.image_loader.load_image(nuclear_channel_path)
            
            # Preprocess image
            preprocessing_result = self.preprocessor.preprocess(image)
            processed_image = preprocessing_result.processed_image
            
            # Segment organoids
            segmentation_result = self.segmentation_engine.segment_image(processed_image)
            
            # Extract parameters
            organoid_parameters = self.parameter_extractor.extract_parameters(
                image, segmentation_result['labeled_mask']
            )
            
            # Advanced analysis for each organoid
            viability_results = []
            apoptosis_results = []
            
            for params in organoid_parameters:
                # Create individual organoid mask
                organoid_mask = (segmentation_result['labeled_mask'] == params.label)
                
                # Viability analysis
                viability_result = self.viability_analyzer.analyze_viability(
                    image, organoid_mask, params
                )
                viability_results.append(viability_result)
                
                # Apoptosis detection
                apoptosis_result = self.apoptosis_detector.detect_apoptosis(
                    image, organoid_mask, nuclear_channel, params
                )
                apoptosis_results.append(apoptosis_result)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(
                organoid_parameters, viability_results, apoptosis_results
            )
            
            # Overall quality score
            quality_score = self._calculate_overall_quality_score(
                preprocessing_result.quality_metrics,
                viability_results,
                apoptosis_results
            )
            
            result = AnalysisResults(
                image_path=image_path,
                timestamp=timestamp,
                organoid_count=len(organoid_parameters),
                organoid_parameters=organoid_parameters,
                viability_results=viability_results,
                apoptosis_results=apoptosis_results,
                summary_statistics=summary_stats,
                quality_score=quality_score,
                processing_metadata={
                    'preprocessing_steps': preprocessing_result.preprocessing_steps,
                    'segmentation_metrics': segmentation_result['statistics'],
                    'image_metadata': image_metadata,
                    'treatment_info': treatment_info
                }
            )
            
            logger.info(f"Single image analysis completed: {len(organoid_parameters)} organoids analyzed")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze image {image_path}: {e}")
            raise
    
    def analyze_time_series(self, 
                          experiment_id: str,
                          image_series: List[Dict[str, Any]],
                          experiment_metadata: Optional[Dict[str, Any]] = None) -> ExperimentResults:
        """
        Analyze a time series of images.
        
        Args:
            experiment_id: Unique identifier for the experiment
            image_series: List of dictionaries with image info and timestamps
            experiment_metadata: Optional experiment metadata
            
        Returns:
            ExperimentResults with time series analysis
        """
        logger.info(f"Starting time series analysis for experiment: {experiment_id}")
        
        # Create experiment in time series analyzer
        self.time_series_analyzer.create_experiment(experiment_id, experiment_metadata)
        
        # Process each time point
        analysis_results = []
        
        for time_point_data in image_series:
            image_path = time_point_data['image_path']
            timestamp = time_point_data['timestamp']
            nuclear_channel = time_point_data.get('nuclear_channel_path')
            treatment_info = time_point_data.get('treatment_info')
            
            # Analyze single image
            result = self.analyze_single_image(
                image_path, timestamp, nuclear_channel, treatment_info
            )
            analysis_results.append(result)
            
            # Add to time series tracker
            self.time_series_analyzer.add_experiment_timepoint(
                experiment_id, timestamp, result.organoid_parameters,
                result.viability_results, result.apoptosis_results, treatment_info
            )
        
        # Get time series analysis
        time_series_data = self.time_series_analyzer.get_experiment_summary(experiment_id)
        
        # Perform comparative analysis
        comparative_analysis = self._perform_comparative_analysis(analysis_results)
        
        experiment_result = ExperimentResults(
            experiment_id=experiment_id,
            analysis_results=analysis_results,
            time_series_data=time_series_data,
            comparative_analysis=comparative_analysis,
            experiment_metadata=experiment_metadata or {}
        )
        
        logger.info(f"Time series analysis completed: {len(analysis_results)} time points processed")
        return experiment_result
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Dictionary with comparative analysis
        """
        return self.time_series_analyzer.compare_experiments(experiment_ids)
    
    def _calculate_summary_statistics(self, 
                                    parameters: List[OrganoidParameters],
                                    viability: List[ViabilityResults],
                                    apoptosis: List[ApoptosisResults]) -> Dict[str, Any]:
        """Calculate summary statistics for an analysis result."""
        if not parameters:
            return {}
        
        # Morphological statistics
        areas = [p.morphological.get('area_pixels', 0) for p in parameters]
        circularities = [p.morphological.get('circularity', 0) for p in parameters]
        
        # Viability statistics
        viability_scores = [v.viability_score for v in viability]
        viable_count = sum(1 for v in viability if v.viability_classification == 'viable')
        
        # Apoptosis statistics
        apoptosis_scores = [a.apoptosis_score for a in apoptosis]
        apoptotic_count = sum(1 for a in apoptosis if a.apoptosis_stage != 'none')
        
        summary = {
            'morphological': {
                'mean_area': float(np.mean(areas)),
                'std_area': float(np.std(areas)),
                'mean_circularity': float(np.mean(circularities)),
                'total_area': float(np.sum(areas))
            },
            'viability': {
                'mean_score': float(np.mean(viability_scores)),
                'std_score': float(np.std(viability_scores)),
                'viable_percentage': (viable_count / len(viability)) * 100 if viability else 0,
                'viable_count': viable_count
            },
            'apoptosis': {
                'mean_score': float(np.mean(apoptosis_scores)),
                'std_score': float(np.std(apoptosis_scores)),
                'apoptotic_percentage': (apoptotic_count / len(apoptosis)) * 100 if apoptosis else 0,
                'apoptotic_count': apoptotic_count
            },
            'counts': {
                'total_organoids': len(parameters),
                'viable_organoids': viable_count,
                'apoptotic_organoids': apoptotic_count,
                'healthy_organoids': len(parameters) - apoptotic_count
            }
        }
        
        return summary
    
    def _calculate_overall_quality_score(self, 
                                       preprocessing_metrics: Dict[str, float],
                                       viability_results: List[ViabilityResults],
                                       apoptosis_results: List[ApoptosisResults]) -> float:
        """Calculate overall quality score for the analysis."""
        scores = []
        
        # Image quality score
        image_quality = preprocessing_metrics.get('final_snr', 0) / 10  # Normalize
        scores.append(min(image_quality, 1.0))
        
        # Analysis confidence scores
        if viability_results:
            mean_viability_confidence = np.mean([v.confidence for v in viability_results])
            scores.append(mean_viability_confidence)
        
        if apoptosis_results:
            mean_apoptosis_confidence = np.mean([a.confidence for a in apoptosis_results])
            scores.append(mean_apoptosis_confidence)
        
        # Overall quality is the mean of all scores
        overall_quality = np.mean(scores) if scores else 0.0
        
        return float(np.clip(overall_quality, 0.0, 1.0))
    
    def _perform_comparative_analysis(self, analysis_results: List[AnalysisResults]) -> Dict[str, Any]:
        """Perform comparative analysis across time points."""
        if len(analysis_results) < 2:
            return {}
        
        # Extract time series data
        timestamps = [r.timestamp for r in analysis_results if r.timestamp is not None]
        organoid_counts = [r.organoid_count for r in analysis_results]
        quality_scores = [r.quality_score for r in analysis_results]
        
        # Viability trends
        viability_means = []
        for result in analysis_results:
            if result.viability_results:
                mean_viability = np.mean([v.viability_score for v in result.viability_results])
                viability_means.append(mean_viability)
            else:
                viability_means.append(0.0)
        
        # Apoptosis trends
        apoptosis_means = []
        for result in analysis_results:
            if result.apoptosis_results:
                mean_apoptosis = np.mean([a.apoptosis_score for a in result.apoptosis_results])
                apoptosis_means.append(mean_apoptosis)
            else:
                apoptosis_means.append(0.0)
        
        comparative = {
            'time_trends': {
                'organoid_count_trend': self._calculate_trend(timestamps, organoid_counts),
                'viability_trend': self._calculate_trend(timestamps, viability_means),
                'apoptosis_trend': self._calculate_trend(timestamps, apoptosis_means),
                'quality_trend': self._calculate_trend(timestamps, quality_scores)
            },
            'summary_changes': {
                'initial_organoid_count': organoid_counts[0] if organoid_counts else 0,
                'final_organoid_count': organoid_counts[-1] if organoid_counts else 0,
                'organoid_count_change': organoid_counts[-1] - organoid_counts[0] if len(organoid_counts) >= 2 else 0,
                'initial_viability': viability_means[0] if viability_means else 0,
                'final_viability': viability_means[-1] if viability_means else 0,
                'viability_change': viability_means[-1] - viability_means[0] if len(viability_means) >= 2 else 0
            }
        }
        
        return comparative
    
    def _calculate_trend(self, x_values: List[float], y_values: List[float]) -> Dict[str, float]:
        """Calculate trend statistics for time series data."""
        if len(x_values) < 2 or len(y_values) < 2:
            return {'slope': 0.0, 'correlation': 0.0, 'trend': 'insufficient_data'}
        
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
            
            # Classify trend
            if abs(slope) < 0.01 or p_value > 0.05:
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            
            return {
                'slope': float(slope),
                'correlation': float(r_value),
                'p_value': float(p_value),
                'trend': trend
            }
        except:
            return {'slope': 0.0, 'correlation': 0.0, 'trend': 'calculation_failed'}
    
    def export_results(self, 
                      results: Union[AnalysisResults, ExperimentResults],
                      output_path: str,
                      format: str = 'json') -> None:
        """
        Export analysis results to file.
        
        Args:
            results: Analysis or experiment results
            output_path: Path for output file
            format: Export format ('json', 'csv')
        """
        output_path = Path(output_path)
        
        if format == 'json':
            self._export_json(results, output_path)
        elif format == 'csv':
            self._export_csv(results, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, results: Union[AnalysisResults, ExperimentResults], output_path: Path):
        """Export results as JSON."""
        # Convert results to serializable format
        if isinstance(results, AnalysisResults):
            data = self._analysis_results_to_dict(results)
        elif isinstance(results, ExperimentResults):
            data = self._experiment_results_to_dict(results)
        else:
            raise ValueError("Unsupported results type")
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results exported to JSON: {output_path}")
    
    def _export_csv(self, results: Union[AnalysisResults, ExperimentResults], output_path: Path):
        """Export results as CSV."""
        try:
            import pandas as pd
            
            if isinstance(results, AnalysisResults):
                # Create DataFrame from organoid parameters
                df = self.parameter_extractor.create_parameters_dataframe(results.organoid_parameters)
                
                # Add viability and apoptosis data
                if results.viability_results:
                    viability_data = {
                        'viability_score': [v.viability_score for v in results.viability_results],
                        'viability_classification': [v.viability_classification for v in results.viability_results],
                        'live_percentage': [v.live_cell_percentage for v in results.viability_results]
                    }
                    viability_df = pd.DataFrame(viability_data)
                    df = pd.concat([df, viability_df], axis=1)
                
                if results.apoptosis_results:
                    apoptosis_data = {
                        'apoptosis_score': [a.apoptosis_score for a in results.apoptosis_results],
                        'apoptosis_stage': [a.apoptosis_stage for a in results.apoptosis_results],
                        'apoptotic_percentage': [a.apoptotic_cell_percentage for a in results.apoptosis_results]
                    }
                    apoptosis_df = pd.DataFrame(apoptosis_data)
                    df = pd.concat([df, apoptosis_df], axis=1)
                
            elif isinstance(results, ExperimentResults):
                # Create time series DataFrame
                all_data = []
                
                for result in results.analysis_results:
                    df_single = self.parameter_extractor.create_parameters_dataframe(result.organoid_parameters)
                    df_single['timestamp'] = result.timestamp
                    df_single['image_path'] = result.image_path
                    all_data.append(df_single)
                
                df = pd.concat(all_data, ignore_index=True)
            
            df.to_csv(output_path, index=False)
            logger.info(f"Results exported to CSV: {output_path}")
            
        except ImportError:
            logger.error("pandas not available for CSV export")
            raise
    
    def _analysis_results_to_dict(self, results: AnalysisResults) -> Dict[str, Any]:
        """Convert AnalysisResults to dictionary."""
        return {
            'image_path': results.image_path,
            'timestamp': results.timestamp,
            'organoid_count': results.organoid_count,
            'summary_statistics': results.summary_statistics,
            'quality_score': results.quality_score,
            'organoid_data': [
                {
                    'id': i,
                    'parameters': {
                        'morphological': params.morphological,
                        'intensity': params.intensity,
                        'spatial': params.spatial
                    },
                    'viability': {
                        'score': results.viability_results[i].viability_score,
                        'classification': results.viability_results[i].viability_classification,
                        'confidence': results.viability_results[i].confidence
                    } if i < len(results.viability_results) else None,
                    'apoptosis': {
                        'score': results.apoptosis_results[i].apoptosis_score,
                        'stage': results.apoptosis_results[i].apoptosis_stage,
                        'confidence': results.apoptosis_results[i].confidence
                    } if i < len(results.apoptosis_results) else None
                }
                for i, params in enumerate(results.organoid_parameters)
            ]
        }
    
    def _experiment_results_to_dict(self, results: ExperimentResults) -> Dict[str, Any]:
        """Convert ExperimentResults to dictionary."""
        return {
            'experiment_id': results.experiment_id,
            'time_points': len(results.analysis_results),
            'time_series_data': results.time_series_data,
            'comparative_analysis': results.comparative_analysis,
            'experiment_metadata': results.experiment_metadata,
            'analysis_results': [
                self._analysis_results_to_dict(result) for result in results.analysis_results
            ]
        }


# Utility functions
def analyze_organoid_image(image_path: str, 
                          config: Optional[Config] = None) -> AnalysisResults:
    """
    Convenience function for single image analysis.
    
    Args:
        image_path: Path to image file
        config: Optional configuration
        
    Returns:
        AnalysisResults
    """
    pipeline = AdvancedAnalysisPipeline(config)
    return pipeline.analyze_single_image(image_path)


def create_analysis_report(results: Union[AnalysisResults, ExperimentResults]) -> str:
    """
    Create a formatted text report from analysis results.
    
    Args:
        results: Analysis or experiment results
        
    Returns:
        Formatted report string
    """
    if isinstance(results, AnalysisResults):
        return _create_single_analysis_report(results)
    elif isinstance(results, ExperimentResults):
        return _create_experiment_report(results)
    else:
        return "Unsupported results type"


def _create_single_analysis_report(results: AnalysisResults) -> str:
    """Create report for single image analysis."""
    report = []
    report.append("=== ORGANOID ANALYSIS REPORT ===")
    report.append(f"Image: {Path(results.image_path).name}")
    report.append(f"Timestamp: {results.timestamp}")
    report.append(f"Quality Score: {results.quality_score:.3f}")
    report.append("")
    
    report.append(f"ORGANOIDS DETECTED: {results.organoid_count}")
    
    if results.summary_statistics:
        stats = results.summary_statistics
        
        report.append("\nMORPHOLOGY:")
        report.append(f"  Average Area: {stats['morphological']['mean_area']:.1f} ± {stats['morphological']['std_area']:.1f} pixels")
        report.append(f"  Average Circularity: {stats['morphological']['mean_circularity']:.3f}")
        report.append(f"  Total Area: {stats['morphological']['total_area']:.1f} pixels")
        
        report.append("\nVIABILITY:")
        report.append(f"  Average Score: {stats['viability']['mean_score']:.3f} ± {stats['viability']['std_score']:.3f}")
        report.append(f"  Viable Organoids: {stats['viability']['viable_count']} ({stats['viability']['viable_percentage']:.1f}%)")
        
        report.append("\nAPOPTOSIS:")
        report.append(f"  Average Score: {stats['apoptosis']['mean_score']:.3f} ± {stats['apoptosis']['std_score']:.3f}")
        report.append(f"  Apoptotic Organoids: {stats['apoptosis']['apoptotic_count']} ({stats['apoptosis']['apoptotic_percentage']:.1f}%)")
    
    return "\n".join(report)


def _create_experiment_report(results: ExperimentResults) -> str:
    """Create report for experiment results."""
    report = []
    report.append("=== EXPERIMENT ANALYSIS REPORT ===")
    report.append(f"Experiment ID: {results.experiment_id}")
    report.append(f"Time Points: {len(results.analysis_results)}")
    report.append("")
    
    if results.time_series_data:
        ts_data = results.time_series_data
        report.append("TIME SERIES SUMMARY:")
        report.append(f"  Total Organoids Tracked: {ts_data.get('organoid_count', 0)}")
        report.append(f"  Average Growth Rate: {ts_data.get('average_growth_rate', 0):.3f}")
        report.append(f"  Average Final Area: {ts_data.get('average_final_area', 0):.1f}")
        report.append(f"  Tracking Quality: {ts_data.get('tracking_quality_mean', 0):.3f}")
    
    if results.comparative_analysis:
        comp = results.comparative_analysis
        report.append("\nCOMPARATIVE ANALYSIS:")
        
        if 'summary_changes' in comp:
            changes = comp['summary_changes']
            report.append(f"  Organoid Count Change: {changes.get('organoid_count_change', 0)}")
            report.append(f"  Viability Change: {changes.get('viability_change', 0):.3f}")
        
        if 'time_trends' in comp:
            trends = comp['time_trends']
            report.append(f"  Viability Trend: {trends.get('viability_trend', {}).get('trend', 'unknown')}")
            report.append(f"  Apoptosis Trend: {trends.get('apoptosis_trend', {}).get('trend', 'unknown')}")
    
    return "\n".join(report)