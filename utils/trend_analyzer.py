from collections import defaultdict
from models.schemas import TrendPoint, TrendResult

def compute_trends_for_all_parameters(history: list[TrendPoint]) -> list[TrendResult]:
    """
    Compute patient health trends for all parameters present in the history.
    
    Args:
        history: List of TrendPoint objects, expected to be sorted by date.
        
    Returns:
        List of TrendResult objects summarizing each parameter's history.
    """
    # Group by parameter
    by_param = defaultdict(list)
    for tp in history:
        by_param[tp.parameter].append(tp)
        
    results = []
    for param, points in by_param.items():
        # Make sure they are sorted chronologically
        points_sorted = sorted(points, key=lambda x: x.report_date)
        values = [p.value for p in points_sorted]
        dates = [p.report_date for p in points_sorted]
        
        if len(values) < 2:
            direction = "Insufficient data"
            note = "Need at least two reports to compute a trend."
        else:
            last_val = values[-1]
            prev_val = values[-2]
            diff = last_val - prev_val
            
            # Simple threshold for stable (e.g. less than 1% change)
            percent_change = (diff / prev_val * 100) if prev_val != 0 else 0
            if abs(percent_change) < 1.0:
                direction = "Stable"
                note = f"Remained stable around {last_val} {points_sorted[-1].unit}."
            elif diff > 0:
                direction = "Increasing"
                note = f"Increased from {prev_val} to {last_val} {points_sorted[-1].unit} (+{percent_change:.1f}%)."
            else:
                direction = "Decreasing"
                note = f"Decreased from {prev_val} to {last_val} {points_sorted[-1].unit} ({percent_change:.1f}%)."
                
        results.append(TrendResult(
            parameter=param,
            values=values,
            dates=dates,
            direction=direction,
            note=note
        ))
        
    return results
