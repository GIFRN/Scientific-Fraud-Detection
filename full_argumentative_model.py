from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, AutoTokenizer, AutoModel
from model_quality_rob_binary_classification import ModelQualityPredictor
import torch
import csv
import json
import numpy as np
import pandas as pd
import pickle
import lib
import random
import tensorflow as tf
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from lib import encode_text, pad_array, load_dataset, evaluate
from pathlib import Path
from sacred import Experiment
from tqdm import tqdm
from models import bert_transformer
from scipy.special import softmax
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



ex = Experiment(name='Argumentation-based Model')
tqdm.pandas(smoothing=0)


@ex.config
def config():
    seed = 2020
    # Argument Labels.
    label2id = {'PAD': 0, 'NONE': 1, 'EVIDENCE': 2, 'CLAIM': 3}
    num_classes = len(label2id)

    # Max sentences per abstract.
    max_sentences = 20
    # Input of an abstract: max_sentences x 768 embedding vector.
    sents_shape = (max_sentences, 768)
    model_fit = dict(epochs=40, validation_split=0.2)

    # A BERT model to use as a Sentence Encoder,
    # e.g., 'allenai/scibert_scivocab_uncased', 'bert_base_uncased', etc.
    sentence_encoder = 'allenai/scibert_scivocab_uncased'
    # The file name of the trained model.
    save_model = 'new_models/scibert_transformer.h5'

    # You can have multiple datasets per list. All datasets (per list) will be
    # encoded and stacked to form a new combined one.
    datasets_legitimate = dict(train=['data/SciARK.json'], dev=[], test=['data/legitimate_abstracts.json'])
    datasets_fraudulent = dict(train=['data/SciARK.json'], dev=[], test=['data/fraudulent_abstracts.json'])
    train_test_splits = dict(train=0.9, dev=0.05, test=0.05)

    # Transformer block (Context Encoder) hyperparameters.
    transformer = dict(
    embed_dim=768,
    num_heads=8,
    ff_dim=128,
    dropout_rate=0.1
)


def load_datasets_fraudulent(config):
    datasets = dict(
        train=dict(),
        dev=dict(),
        test=dict(),
        X=dict(train=dict(), dev=dict(), test=dict()),
        ids=dict(train=dict(), dev=dict(), test=dict()),
        y=dict(train=dict(), dev=dict(), test=dict()),
        raw=dict(train=dict(), dev=dict(), test=dict()))
    
    # If there is not a separated test set, we have to split the training one.
    if config['datasets_fraudulent']['test'] is None:
        train, dev, test = lib.load_datasets(
            config['datasets_fraudulent'], config['train_test_splits'], config['seed'])
        datasets['train']['train_split'] = train
        datasets['dev']['dev_split'] = dev
        datasets['test']['test_split'] = test
        return datasets

    for dataset in ('train', 'dev', 'test'):
        for src in config['datasets_fraudulent'][dataset]:
            datasets[dataset][src] = load_dataset(src, config['seed'])     
            with open(src, 'r') as f:
                raw_data = json.load(f)
                for abstract in raw_data:
                    raw_data[abstract]['id'] = abstract
            datasets['raw'][dataset][src] = raw_data           
    return datasets


def load_datasets_legitimate(config):
    datasets = dict(
        train=dict(),
        dev=dict(),
        test=dict(),
        X=dict(train=dict(), dev=dict(), test=dict()),
        ids=dict(train=dict(), dev=dict(), test=dict()),
        y=dict(train=dict(), dev=dict(), test=dict()),
        raw=dict(train=dict(), dev=dict(), test=dict()))
    
    # If there is not a separated test set, we have to split the training one.
    if config['datasets_legitimate']['test'] is None:
        train, dev, test = lib.load_datasets(
            config['datasets_legitimate'], config['train_test_splits'], config['seed'])
        datasets['train']['train_split'] = train
        datasets['dev']['dev_split'] = dev
        datasets['test']['test_split'] = test
        return datasets

    for dataset in ('train', 'dev', 'test'):
        for src in config['datasets_legitimate'][dataset]:
            datasets[dataset][src] = load_dataset(src, config['seed'])     
            with open(src, 'r') as f:
                raw_data = json.load(f)
                for abstract in raw_data:
                    raw_data[abstract]['id'] = abstract
            datasets['raw'][dataset][src] = raw_data           
    return datasets


def stack_encoded_datasets(datasets, key, dataset):
    # Stack the encoded datasets.
    stack = None
    if key in datasets:
        if dataset in datasets[key]:
            for corpus in datasets[key][dataset]:
                if stack is None:
                    stack = datasets[key][dataset][corpus]
                else:
                    stack = np.vstack([stack, datasets[key][dataset][corpus]])
    return stack


def encode_datasets(datasets, config, _log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_tokenizer = AutoTokenizer.from_pretrained(config['sentence_encoder'])
    bert_model = AutoModel.from_pretrained(config['sentence_encoder']).to(
        device)
    label2id = config['label2id']
    max_sentences = config['max_sentences']
    # for each document in the data
    for dataset in ('train', 'dev', 'test'):
        for corpus in datasets[dataset]:
            _log.info(f'Encoding {corpus}')
            data = datasets[dataset][corpus]

            # Encode labels to one-hot vectors.
            data['y'] = data.labels.map(lambda row: [label2id[r] for r in row])
            data['y'] = data.y.map(
                lambda row: pad_sequences([row], maxlen=max_sentences)[0])
            y = to_categorical(data.y.to_list())
            datasets['y'][dataset][corpus] = y

            # Encode sentences.
            data['sentence_embeddings'] = data.sentences.progress_map(
                lambda row: [
                    encode_text(sentence, bert_tokenizer, bert_model, device)
                    for sentence in row
                ])
            # Reshape each document's sentence embeddings to:
            # 1 (doc), #sents, 768
            data['X'] = data.sentence_embeddings.map(
                lambda vec: pad_array(np.vstack(vec), max_sentences))

            # data.X.to_numpy() doesn't return a 3D array.
            # It returns (len(data), ). So, I stack all rows and reshape.
            X = np.vstack(data.X.to_numpy()).reshape(
                (len(data), max_sentences, 768))
            datasets['X'][dataset][corpus] = X
         
            # Add ids to the dataset        
            datasets['ids'][dataset][corpus] = data['id'].values
 
            # Pickle encoded data.
            pickled = '{}_{}.p'.format(
                corpus.replace('.json', ''),
                config['sentence_encoder'].split('/')[-1])
            with open(pickled, 'wb') as fp:
                pickle.dump((X, y, data['id'].values), fp)


    X_train = stack_encoded_datasets(datasets, 'X', 'train')
    y_train = stack_encoded_datasets(datasets, 'y', 'train')
    X_dev = stack_encoded_datasets(datasets, 'X', 'dev')
    y_dev = stack_encoded_datasets(datasets, 'y', 'dev')
    X_test = stack_encoded_datasets(datasets, 'X', 'test')
    y_test = stack_encoded_datasets(datasets, 'y', 'test')

    id_train = stack_encoded_datasets(datasets, 'ids', 'train')
    id_dev = stack_encoded_datasets(datasets, 'ids', 'dev')
    id_test = stack_encoded_datasets(datasets, 'ids', 'test')

    encoded_datasets = dict(
        X=dict(train=X_train, dev=X_dev, test=X_test),
        y=dict(train=y_train, dev=y_dev, test=y_test),
        ids=dict(train=id_train, dev=id_dev, test=id_test)
)

    return encoded_datasets



# Argument Quality Evaluation model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_saved_model(model_save_path, base_model):
    model = ModelQualityPredictor(base_model)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()  
    return model

def prepare_input(text, tokenizer, max_length=512):
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    return input_ids, attention_mask


def infer(model, text, tokenizer):
    input_ids, attention_mask = prepare_input(text, tokenizer)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class.item(), probabilities.squeeze().tolist()

# Load the saved model
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
base_model = RobertaForSequenceClassification.from_pretrained('roberta-base')
model_save_path = 'quality_evaluation_model.pt'
loaded_model = load_saved_model(model_save_path, base_model)


def process_datasets(datasets, _config, _log):
    # Collect abstracts into an array
    raw_test_data = []
    for val in datasets['raw']['test'].values():
        for v in list(val.values()):
            raw_test_data.append((v['id'], v['sentences']))
    
    # Encode datasets
    embedded_datasets = encode_datasets(datasets, _config, _log)
    X_train = embedded_datasets['X']['train']
    y_train = embedded_datasets['y']['train']
    X_dev = embedded_datasets['X']['dev']
    
    if X_dev is not None:
        y_dev = embedded_datasets['y']['dev']
        validation_data = (X_dev, y_dev)
    else:
        validation_data = None

    X_test = embedded_datasets['X']['test']
    y_test = embedded_datasets['y']['test']
    id_test = np.array(embedded_datasets['ids']['test'])

    return X_train, y_train, validation_data, X_test, y_test, id_test, raw_test_data

def process_prediction(model, X_test, id_test, raw_test_data):
    logits = model(X_test)
    y_pred_unsorted = softmax(logits).argmax(-1)
    y_pred_unsorted = np.array(y_pred_unsorted)
    
    id_test = np.array(id_test)
    sort_indices = np.argsort(id_test)
    
    logits = np.array(logits)
    logits_sorted = logits[sort_indices]
    
    y_pred = y_pred_unsorted[sort_indices]
    
    raw_test_data.sort(key=lambda x: x[0])  # sorts in-place
    
    return y_pred, logits_sorted, raw_test_data


def evaluate_predictions(y_true, y_pred, label):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Metrics for {label} data:")
    print(f"  Accuracy : {accuracy}")
    print(f"  Precision: {precision}")
    print(f"  Recall   : {recall}")
    print(f"  F1 Score : {f1}\n")
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


@ex.main
def main(_config, _log):
    seed = _config['seed']
    np.random.seed(seed)
    torch.seed = seed
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)

    datasets_fraudulent = load_datasets_fraudulent(_config)
    datasets_legitimate = load_datasets_legitimate(_config)
    
    X_train_fraudulent, y_train_fraudulent, validation_data_fraudulent, \
    X_test_fraudulent, y_test_fraudulent, id_test_fraudulent, raw_test_data_fraudulent = \
    process_datasets(datasets_fraudulent, _config, _log)
    
    X_train_legitimate, y_train_legitimate, validation_data_legitimate, \
    X_test_legitimate, y_test_legitimate, id_test_legitimate, raw_test_data_legitimate = \
    process_datasets(datasets_legitimate, _config, _log)

    # Process fraudulent data
    y_pred_fraudulent, logits_sorted_fraudulent, sorted_raw_test_data_fraudulent = \
    process_prediction(model, X_test_fraudulent, id_test_fraudulent, raw_test_data_fraudulent)
    
    # Process legitimate data
    y_pred_legitimate, logits_sorted_legitimate, sorted_raw_test_data_legitimate = \
    process_prediction(model, X_test_legitimate, id_test_legitimate, raw_test_data_legitimate)

    # Evaluate fraudulent data
    metrics_fraudulent = evaluate_predictions(y_test_fraudulent, y_pred_fraudulent, "Fraudulent")
    
    # Evaluate legitimate data
    metrics_legitimate = evaluate_predictions(y_test_legitimate, y_pred_legitimate, "Legitimate")
    
    # Calculate and display overall scores
    overall_metrics = {}
    for key in metrics_fraudulent.keys():
        overall_metrics[key] = (metrics_fraudulent[key] + metrics_legitimate[key]) / 2
    
    print("Overall Metrics:")
    for key, value in overall_metrics.items():
        print(f"  {key.capitalize()}: {value}")
 

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '-e',
        '--sentence-encoder',
        type=str,
        default=None,
        help='A huggingface 🤗  model to use as a sentence encoder.')
    parser.add_argument(
        '-s',
        '--save-model',
        type=str,
        default='trained_models/bert_based_context_encoder.h5',
        help='The file name for the saved model. '
        'The file name may have a path and *should* have `.h5` extension.')
    parser.add_argument(
        '-t',
        '--train-set',
        nargs='*',
        default=None,
        help='A list of datasets to use as a training set.')
    parser.add_argument(
        '-d',
        '--development-set',
        nargs='*',
        default=None,
        help='A list of datasets to use as a development set.')
    parser.add_argument(
        '-x',
        '--test-set',
        nargs='*',
        default=None,
        help='A list of datasets to use as a test set.')
    args = parser.parse_args()

    config_update = dict()
    if args.sentence_encoder is not None:
        config_update['sentence_encoder'] = args.sentence_encoder
    if args.save_model is not None:
        config_update['save_model'] = args.save_model
        # Create the parent directories.
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
    if args.train_set is not None:
        if 'datasets' not in config_update:
            config_update['datasets'] = dict()
        config_update['datasets']['train'] = args.train_set
    if args.development_set is not None:
        if 'datasets' not in config_update:
            config_update['datasets'] = dict()
        config_update['datasets']['dev'] = args.development_set
    if args.test_set is not None:
        if 'datasets' not in config_update:
            config_update['datasets'] = dict()
        config_update['datasets']['test'] = args.test_set
    ex.run(config_updates=config_update)
