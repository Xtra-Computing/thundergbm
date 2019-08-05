import pandas as pd


def add_data(df, algorithm, data, elapsed, metric):
    time_col = (data.name, 'Time(s)')
    metric_col = (data.name, data.metric)
    try:
        df.insert(len(df.columns), time_col, '-')
        df.insert(len(df.columns), metric_col, '-')
    except:
        pass

    df.at[algorithm, time_col] = elapsed
    df.at[algorithm, metric_col] = metric


def write_results(df, filename, format):
    if format == "latex":
        tmp_df = df.copy()
        tmp_df.columns = pd.MultiIndex.from_tuples(tmp_df.columns)
        with open(filename, "a") as file:
            file.write(tmp_df.to_latex())
    elif format == "csv":
        with open(filename, "a") as file:
            file.write(df.to_csv())
    else:
        raise ValueError("Unknown format: " + format)

    print(format + " results written to: " + filename)