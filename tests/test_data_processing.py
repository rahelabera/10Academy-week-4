import pandas as pd
from src.data_processing import *  # Or import specific functions

def test_agg_features():
    # Small test df
    test_df = pd.DataFrame({...})  # Create small sample
    # Assert columns exist
    assert 'total_amount' in result.columns