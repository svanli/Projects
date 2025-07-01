# helper_functions.py

import pandas as pd

from IPython.display import display, HTML

def scrollable_output(output_height = 500):
    """Enable scrollable output."""
    display(HTML(f"""
    <style>
    .jp-OutputArea-output {{
        max-height: {output_height}px;
        overflow-y: auto;
    }}
    </style>
    """))

def dataset_check(
    dataset,
    shape_check = True,
    dup_check = True,
    dtypes_check = True,
    missing_values_check = True,
    unique_values_check = True,
    unique_values_percentage_check = True,
    stats_check = False,
    corr_check = False
):
    """
    Provides a comprehensive overview of the dataset, including:

    * The shape of the dataset (number of rows and columns).
    * Detection of duplicate rows.
    * Data types of each column.
    * Identification of columns with missing values, including the count and percentage of missing values.
    * Unique values in each column.
    * Percentage of unique values (unique-to-total ratio) for each column.
    * Basic statistical summary (e.g., count, mean, standard deviation, minimum, maximum) for numerical columns.
    * Correlation matrix between numerical columns.

    By default, the function performs all checks except for basic statistics and correlations, which can be enabled with the `stats_check` and `corr_check` parameters.
    
    Parameters:
    - dataset: The DataFrame to be analyzed.
    - shape_check (bool): Whether to print the shape of the dataset.
    - dup_check (bool): Whether to check for duplicate rows.
    - dtypes_check (bool): Whether to print the data types of each column.
    - missing_values_check (bool): Whether to check for missing values.
    - unique_values_check (bool): Whether to print the number of unique values in each column.
    - unique_values_percentage_check (bool): Whether to print unique-to-total value percentages per column.
    - stats_check (bool): Whether to print basic statistics of numerical columns.
    - corr_check (bool): Whether to print the correlation matrix of numerical columns.
    """
    # Calculate
    df = dataset
    columns = df.shape[1]
    rows = df.shape[0]
    duplicate_rows = df.duplicated().sum()
    dtypes = df.dtypes
    missing = df.isna().sum()
    missing = missing[missing > 0]
    missing_percentage = (df.isna().sum() / len(df) * 100).round(2)
    missing_percentage = missing_percentage[missing_percentage > 0]
    unique = df.nunique()
    unique_percentage = (df.nunique() / len(df)).apply(lambda x: f"{x:.2%}")
    stats = df.describe() if stats_check else None
    corr = df.corr() if corr_check else None

    # Print
    if shape_check:
        print(f'COLUMNS:\n{columns:,}\n'.replace(',', ' '))
        print(f'ROWS:\n{rows:,}\n'.replace(',', ' '))
        
    if dup_check:
        print(f'DUPLICATE ROWS:\n{duplicate_rows}\n')

    if dtypes_check:
        print(f'DATATYPES:\n{dtypes}\n')

    if missing_values_check:
        if missing.empty:
            print('MISSING VALUES:\n0\n')
            print('MISSING VALUES IN %:\n0%\n')
        else:
            print(f'MISSING VALUES:\n{missing}\n')
            print(f'MISSING VALUES IN %:\n{missing_percentage.apply(lambda x: f"{x:.2f}%")}\n')

    if unique_values_check:    
        print(f'UNIQUE VALUES:\n{unique}\n')

    if unique_values_percentage_check:
        print(f'UNIQUE VALUES IN %:\n{unique_percentage}\n')
        
    if stats_check:
        print(f'BASIC STATISTICS:\n{stats}\n')
        
    if corr_check:
        print(f'CORRELATION MATRIX:\n{corr}\n')

def uvalues_check(dataset, columns):
    """
    Provides a list of unique values for specified columns in the dataset, sorted alphabetically.

    This function takes a dataset and a list of column names, then prints the unique values for each specified column.
    The unique values are sorted alphabetically to enhance readability.

    Parameters:
    - dataset: The DataFrame to be analyzed.
    - columns: A list of column names for which unique values will be printed.
    
    Example:
    >>> df = pd.DataFrame({'A': [1, 2, 2], 'B': ['x', 'y', 'x']})
    >>> uvalues_check(df, ['A', 'B'])
    UNIQUE VALUES

    A
    [1, 2]

    B
    ['x', 'y']
    """
    print('UNIQUE VALUES\n')
    for column in columns:
        print(column.upper())
        uvalues = dataset[column].unique()
        uvalues_sorted = sorted(uvalues, key = lambda x: str(x).lower())
        print(uvalues_sorted, '\n')

def text_check(dataset, columns):
    """
    Identifies and prints non-numeric values in specified columns of the dataset.

    This function examines the specified columns in the provided dataset and identifies values that are not numeric.
    It prints out these non-numeric values, sorted alphabetically. If a column contains only numeric values, it indicates this as well.

    Parameters:
    - dataset: The DataFrame to be analyzed.
    - columns: A list of column names to be checked for non-numeric values.
    
    Example:
    >>> df = pd.DataFrame({'A': [1, 'two', 3], 'B': ['x', '2', '3.5']})
    >>> text_check(df, ['A', 'B'])
    NON-NUMERIC VALUES

    A
    ['two']

    B
    ['x']
    """
    df = dataset
    print('NON-NUMERIC VALUES\n')
    for column in columns:
        numeric = pd.to_numeric(df[column], errors = 'coerce')
        text_values = df[column][numeric.isna() & df[column].notna()].unique()
        text_values_sorted = sorted(text_values, key = lambda x: x.lower())
        
        if len(text_values_sorted) > 0:
            print(column.upper())
            print(text_values_sorted, '\n')
        else:
            print(column.upper())
            print('Numeric values only\n')

import unicodedata

def clean_text(dataset, columns):
    """
    Cleans text columns in the specified columns of the dataset by:
    - Normalizing Unicode to remove non-standard characters.
    - Stripping leading/trailing whitespace.
    - Replacing multiple spaces with a single space.

    Parameters:
    - dataset: The DataFrame containing the columns to be cleaned.
    - columns: A list of column names to be cleaned.
    
    Example:
    >>> df = pd.DataFrame({'A': ['  hello  ', 'world!'], 'B': [' 123 ', 'abc   ']})
    >>> text_clean(df, ['A', 'B'])
    """
    df = dataset

    for column in columns:
        # Ensure the column is string-like before processing
        df[column] = df[column].astype(str)
        df[column] = df[column].apply(lambda x: unicodedata.normalize("NFKC", x))  # Normalize Unicode
        df[column] = df[column].str.strip()  # Strip leading and trailing whitespace
        df[column] = df[column].str.replace(r"\s+", " ", regex=True)  # Replace multiple spaces with one
    return df.head()