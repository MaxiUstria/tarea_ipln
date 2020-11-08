import sys
from datetime import datetime
from pathlib import Path
from time import time

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocesado import Predictors, spacy_tokenizer


def main():
    data_path_str, input_data_list = sys.argv[1], sys.argv[2:]
    data_path_str = Path(data_path_str)
    input_data_list = [Path(input_path) for input_path in input_data_list]
    if not data_path_str.is_dir():
        print(f"{data_path_str} no es un directorio")
        return
    else:
        train_ds = (
            pd.read_csv(data_path_str / "data_train.csv")
            if (data_path_str / "data_train.csv").exists()
            else pd.read_csv(data_path_str / "humor_train.csv")
        )

    # Pre-processing
    print(f'{datetime.now().strftime("%H:%M:%S")}: Pre-procesando... \n')

    bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 2))

    classifier = LogisticRegression(solver="liblinear")

    # Create pipeline using Bag of Words
    pipe = Pipeline(
        [("cleaner", Predictors()), ("vectorizer", bow_vector), ("classifier", classifier)]
    )

    # Train model
    tic = time()
    print(f'{datetime.now().strftime("%H:%M:%S")}: ' f"Entrenando el modelo...\n")

    # model generation
    pipe.fit(train_ds["text"], train_ds["humor"].values)

    print(f'{datetime.now().strftime("%H:%M:%S")}: ' f"Entrenado en {time()-tic:.2f}s\n")

    # Classification
    print("Evaluación...\n")
    for input_filename in input_data_list:
        if not input_filename.exists():
            print(f"{input_filename} no existe en este directorio.")
            print("Continuando con el siguiente archivo de entrada.\n")

        else:
            print(f"Evaluando {input_filename}")
            data_to_eval = pd.read_csv(input_filename)

            output_list = pipe.predict(data_to_eval["text"])

            with open(f"{input_filename.name[:-4]}.out", "w+") as output_file:
                output_file.write("\n".join(map(str, output_list)))


if __name__ == "__main__":
    main()
