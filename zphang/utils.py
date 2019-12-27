import glob
import pandas as pd
import os
import re

import pyutils.datastructures as datastructures
import pyutils.io as io


def find_best_model(base_path):
    path_ls = sorted(glob.glob("{}/*.chk".format(base_path)))
    if len(path_ls) == 1:
        path = path_ls[0]
    else:
        path = sorted(path_ls, key=lambda p: -int(p.split("_")[-1].split(".")[0]))[0]
    return path


def tags_to_regex(tag_pattern, format_dict=None, default_format="\\w+"):
    if format_dict is None:
        format_dict = {}
    last_end = 0
    new_tokens = []
    for m in re.finditer("\\{(?P<tag>\\w+)\\}", tag_pattern):
        start, end = m.span()
        tag = m["tag"]
        new_tokens.append(tag_pattern[last_end:start])
        tag_format = format_dict.get(tag, default_format)
        new_tokens.append(f"(?P<{tag}>{tag_format})")
        last_end = end
    new_pattern = "".join(new_tokens)
    return new_pattern


def load_score(path_ls, regex_str, return_df=True, filename="score.json"):
    regex = re.compile(regex_str)
    results = []
    for path in path_ls:
        matched = next(regex.finditer(path)).groupdict()
        score_path = os.path.join(os.path.abspath(os.path.join(path, "..")), filename)
        if not os.path.exists(score_path):
            continue
        scores = io.read_json(score_path)
        scores["path"] = path
        results.append(datastructures.combine_dicts([matched, scores]))
    if return_df:
        return pd.DataFrame(results)
    else:
        return results
