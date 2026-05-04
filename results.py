"""
results.py
Authors: Alicia Maranville, Fred Solis
reads test files and writes txt files with one sense predicted per line
uses the functions from cs5322f25prog3 to predict
"""

import os
from cs5322f25prog3 import WSD_Test_director, WSD_Test_overtime, WSD_Test_rubbish

if __name__ == "__main__":
    words = ["director", "overtime", "rubbish"]
    wsd_functions = {
        "director": WSD_Test_director,
        "overtime": WSD_Test_overtime,
        "rubbish": WSD_Test_rubbish,
    }

    for word in words:
        print(f"-PREDICTING FOR {word}-")

        path = f"data/test/{word}_testdata.txt"
        with open(path, "r", encoding="utf-8") as f:
            sentences = [ln.strip() for ln in f if ln.strip()]

        preds = wsd_functions[word](sentences)
        out_path = os.path.join("results", f"result_{word}_AliciaMaranville.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(f"{p}\n") # write each prediction as one line

        # print sentences and their predicted senses: use for debugging
        #for i in range(len(sentences)):
        #    print(sentences[i])
        #    print(f"\tprediction -> sense {preds[i]}")
