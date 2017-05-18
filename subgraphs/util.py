import csv


def get_map_from_tsv(tsv_path, k_col, v_col, cast=float):
    """
    Build a key-value map from two columns of a TSV file.

    Arguments:
        csv_path:
        k_col: Column which should yield keys of map. Values in this should
            column should be unique, otherwise things break!
        v_col: Column which should yield values of map.

    Returns:
        dict
    """

    ret = {}
    with open(tsv_path, "rU") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row[v_col] == "n/a":
                row[v_col] = 0
            ret[row[k_col]] = cast(row[v_col])

    return ret
