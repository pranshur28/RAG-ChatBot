import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

class CSVAnalyzer:
    def __init__(self):
        self.data = None
        self.summary_stats = {}

    def load_csv(self, file_path: str) -> bool:
        """Load and validate CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            return True
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            return False

    def get_basic_stats(self) -> Dict:
        """Calculate basic statistics for numerical columns"""
        if self.data is None:
            return {"error": "No data loaded"}

        stats = {}
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns

        for col in numerical_columns:
            stats[col] = {
                "mean": float(self.data[col].mean()),
                "median": float(self.data[col].median()),
                "std": float(self.data[col].std()),
                "min": float(self.data[col].min()),
                "max": float(self.data[col].max())
            }

        return stats

    def detect_anomalies(self, z_score_threshold: float = 3.0) -> Dict[str, List]:
        """Detect anomalies using Z-score method"""
        if self.data is None:
            return {"error": "No data loaded"}

        anomalies = {}
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns

        for col in numerical_columns:
            z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
            anomalies[col] = self.data[z_scores > z_score_threshold].index.tolist()

        return anomalies

    def analyze_trends(self) -> Dict:
        """Analyze trends in the data"""
        if self.data is None:
            return {"error": "No data loaded"}

        trends = {}
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns

        for col in numerical_columns:
            trends[col] = {
                "increasing": bool(self.data[col].is_monotonic_increasing),
                "decreasing": bool(self.data[col].is_monotonic_decreasing),
                "trend_strength": float(self.data[col].corr(pd.Series(range(len(self.data)))))
            }

        return trends

    def generate_insights(self) -> Dict:
        """Generate comprehensive insights from the data"""
        if self.data is None:
            return {"error": "No data loaded"}

        insights = {
            "basic_stats": self.get_basic_stats(),
            "anomalies": self.detect_anomalies(),
            "trends": self.analyze_trends(),
            "data_quality": {
                "missing_values": self.data.isnull().sum().to_dict(),
                "total_rows": len(self.data),
                "total_columns": len(self.data.columns),
                "column_types": self.data.dtypes.astype(str).to_dict()
            }
        }

        return insights
