from sagemaker_sklearn_extension.decomposition import RobustPCA
from sagemaker_sklearn_extension.externals import Header
from sagemaker_sklearn_extension.impute import RobustImputer
from sagemaker_sklearn_extension.preprocessing import RobustLabelEncoder
from sagemaker_sklearn_extension.preprocessing import RobustStandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Given a list of column names and target column name, Header can return the index
# for given column name
HEADER = Header(
    column_names=[
        'time', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
        'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
        'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'amount',
        'class'
    ],
    target_column_name='class'
)


def build_feature_transform():
    """ Returns the model definition representing feature processing."""

    # These features can be parsed as numeric.

    numeric = HEADER.as_feature_indices(
        [
            'time', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
            'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19',
            'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28',
            'amount'
        ]
    )

    numeric_processors = Pipeline(steps=[('robustimputer', RobustImputer())])

    column_transformer = ColumnTransformer(
        transformers=[('numeric_processing', numeric_processors, numeric)]
    )

    return Pipeline(
        steps=[
            ('column_transformer',
             column_transformer), ('robustpca', RobustPCA(n_components=147)),
            ('robuststandardscaler', RobustStandardScaler())
        ]
    )


def build_label_transform():
    """Returns the model definition representing feature processing."""

    return RobustLabelEncoder(
        labels=['0'], fill_label_value='1', include_unseen_class=True
    )
