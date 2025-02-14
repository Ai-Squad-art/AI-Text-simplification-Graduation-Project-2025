# -- coding: utf-8 --
"""
Created on Sat May 28 07:34:16 2022

@author: Amma
"""

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline
from time import perf_counter

# Disable TensorFlow logs (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Model list
pModels = [
    "UBC-NLP/AraT5-base",
    "facebook/mbart-large-50",
    "google/mt5-base",
    "UBC-NLP/AraT5v2-base-1024"
]

# Function to read text files
def readTxtFile(strPath):
    with open(strPath, 'r', encoding="utf-8") as file:
        return file.read().replace("\n", " ")

# Load dataset
def dsLoad(ins=498):
    refTexts, hSimTexts = [], []
    for i in range(1, ins):
        refTexts.append(readTxtFile("ArSummarizer/TSimAr/resources/Simplification_Datasets/References/{}.txt".format(i)))
        hSimTexts.append(readTxtFile("ArSummarizer/TSimAr/resources/Simplification_Datasets/hSimplification/{}.txt".format(i)))
    return refTexts, hSimTexts

# Download and load models
def load_ptModel(model_name):
    model_path = f"ptModels/{model_name.replace('/', '_')}"  # Save locally
    
    # Check if the model is already downloaded
    if os.path.exists(model_path):
        print(f"Loading {model_name} from local storage...")
    else:
        print(f"Downloading {model_name} from Hugging Face...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Convert tensors to contiguous format
    for param in model.parameters():
        param.data = param.data.contiguous()

    # Save model with fix
    model.save_pretrained(model_path, safe_serialization=False)
    tokenizer.save_pretrained(model_path)

    # Create a text generation pipeline
    pipeline = Text2TextGenerationPipeline(model, tokenizer)
    return pipeline

# Load dataset
refTexts, hSimTexts = dsLoad()

# Run all models
for pM in pModels:
    t1_start = perf_counter()
    pipeline = load_ptModel(pM)

    icount = 0

    for i in range(len(refTexts)):
        output_path = f"evaluation/Rephrase/{pM.replace('/', '_')}"
        os.makedirs(output_path, exist_ok=True)  # Create folder if not exists

        with open(f"{output_path}/{i+1}.txt", "w", encoding="utf-8") as resultFile:
            resultFile.write(pipeline(refTexts[i])[0]['generated_text'])

        icount += 1
        print("\r", f"Progress {icount / len(refTexts):.2%}", end="")

    t1_stop = perf_counter()
    print(f"\n{pM} processing time: {t1_stop - t1_start:.2f} seconds")

print("Done ")
