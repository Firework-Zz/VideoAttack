import json
import os
import pandas as pd
import collections

from global_parameters import MSRVTT_PATH

os.chdir(MSRVTT_PATH)
OUT_DIR = "/home/maviuserfjh/fjh/just-ask/MSRVTT-QA"
os.makedirs(OUT_DIR, exist_ok=True)

train_json = json.load(open("qa_train.json", "r"))      
val_json   = json.load(open("qa_val.json", "r"))
test_json  = json.load(open("qa_test.json", "r"))
types = {"what": 0, "how": 1, "color": 2, "where": 3, "who": 4, "when": 5}
if isinstance(train_json, dict):
    train_json = list(train_json.values())
if isinstance(val_json, dict):
    val_json = list(val_json.values())
if isinstance(test_json, dict):
    test_json = list(test_json.values())

def get_vocabulary(train_json, save=False):
    train_counter = collections.Counter([x["answer"] for x in train_json])
    most_common = train_counter.most_common(4000)  # top 4K answers
    vocab = {}
    for i, x in enumerate(most_common):
        vocab[x[0]] = i
    print(len(vocab))
    if save:
        with open(os.path.join(OUT_DIR, "vocab.json"), "w") as outfile:
            json.dump(vocab, outfile)
    return vocab


def get_type(question):
    if "color" in question:
        return types["color"]
    elif question.split(" ")[0] in ["what", "who", "where", "when", "how"]:
        return types[question.split(" ")[0]]
    else:
        raise NotImplementedError


def json_to_df(vocab, train_json, val_json, test_json, save=False):
    def to_df(split_json):
        return pd.DataFrame(
            {
                "question": [x["question"] for x in split_json],
                "answer":   [x["answer"] for x in split_json],
                # 你这份叫 video
                "video_id": [x["video"] for x in split_json],
                # 你这份叫 question_id
                "id":       [x.get("question_id", i) for i, x in enumerate(split_json)],
                # 可选：保留 answer_type 方便统计（不是必须）
                "answer_type": [x.get("answer_type", "") for x in split_json],
                # 可选：frame_length 想留也行
                "frame_length": [x.get("frame_length", -1) for x in split_json],
            }
        )

    train_df = to_df(train_json)
    print(len(train_df))
    train_df = train_df[train_df["answer"].isin(vocab)]

    val_df = to_df(val_json)
    test_df = to_df(test_json)

    train_df["type"] = [get_type(x) for x in train_df["question"]]
    val_df["type"]   = [get_type(x) for x in val_df["question"]]
    test_df["type"]  = [get_type(x) for x in test_df["question"]]

    print(len(train_df), len(val_df), len(test_df))

    if save:
        train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
        val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
        test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    return train_df, val_df, test_df



vocab = get_vocabulary(train_json, True)
train_df, val_df, test_df = json_to_df(vocab, train_json, val_json, test_json, True)
