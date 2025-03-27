import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from data_cleaning import clean_data, impute_missing_values
from feature_engineering import feature_engineering_pipeline
from feature_selection import feature_selection_pipeline  
from feature_selection import feature_selection_pipeline_lasso  


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

cleaner = FunctionTransformer(clean_data)
fe_pipeline = FunctionTransformer(feature_engineering_pipeline)
fs_pipeline = FunctionTransformer(lambda df: feature_selection_pipeline_lasso(df, target="Rating", cv=5, random_state=42))
# fs_pipeline = FunctionTransformer(lambda df: feature_selection_pipeline(df, target="Rating", corr_threshold=0.1, multicol_threshold=0.9))
imputer = FunctionTransformer(lambda df: impute_missing_values(df, strategy="median"))

full_pipeline = Pipeline([
    ('cleaning', cleaner),
    ('feature_engineering', fe_pipeline),  
    ('feature_selection', fs_pipeline),
    ('imputation', imputer)  
])

if __name__ == "__main__":
    file_path = "data/raw_data.csv"
    df = load_data(file_path)
    print("Raw data loaded.")
    
    df_transformed = full_pipeline.fit_transform(df)
    print("Pipeline executed successfully. Preview of transformed data:")
    print(df_transformed.head())
    
    output_file = "data/processed_pipeline.csv"
    df_transformed.to_csv(output_file, index=False)
    print(f"Transformed data saved to: {output_file}")
