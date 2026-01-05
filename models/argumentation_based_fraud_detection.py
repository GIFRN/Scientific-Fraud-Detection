import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, AutoTokenizer, AutoModel
from quality_model import ModelQualityPredictor
import torch
import csv
import json
import numpy as np
import pandas as pd
import pickle
import lib
import random

# Import TensorFlow and force it to use CPU only (avoids ptxas/nvlink XLA issues)
# PyTorch will still use GPU for sentence encoding
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Hide GPUs from TensorFlow
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
    save_model = 'new_models/scibert_transformer.keras'

    # You can have multiple datasets per list. All datasets (per list) will be
    # encoded and stacked to form a new combined one.
    # Training data for argument mining (claim/evidence extraction)
    datasets_train = dict(train=['../data/SciARK.json'], dev=[], test=[])
    # Test datasets for fraud detection evaluation
    datasets_legitimate = ['../data/legitimate_abstracts.json']
    datasets_fraudulent = ['../data/fraudulent_abstracts.json']
    train_test_splits = dict(train=0.9, dev=0.05, test=0.05)

    # Transformer block (Context Encoder) hyperparameters.
    transformer = dict(
        embed_dim=768,
        num_heads=8,
        ff_dim=128,
        dropout_rate=0.1
    )


def load_training_datasets(config):
    """Load training data for the argument mining model."""
    datasets = dict(
        train=dict(),
        dev=dict(),
        test=dict(),
        X=dict(train=dict(), dev=dict(), test=dict()),
        ids=dict(train=dict(), dev=dict(), test=dict()),
        y=dict(train=dict(), dev=dict(), test=dict()),
        raw=dict(train=dict(), dev=dict(), test=dict()))
    
    for dataset in ('train', 'dev', 'test'):
        for src in config['datasets_train'][dataset]:
            datasets[dataset][src] = load_dataset(src, config['seed'])     
            with open(src, 'r') as f:
                raw_data = json.load(f)
                for abstract in raw_data:
                    raw_data[abstract]['id'] = abstract
            datasets['raw'][dataset][src] = raw_data           
    return datasets


def load_test_dataset(file_path, config):
    """Load a single test dataset file."""
    datasets = dict(
        train=dict(),
        dev=dict(),
        test=dict(),
        X=dict(train=dict(), dev=dict(), test=dict()),
        ids=dict(train=dict(), dev=dict(), test=dict()),
        y=dict(train=dict(), dev=dict(), test=dict()),
        raw=dict(train=dict(), dev=dict(), test=dict()))
    
    datasets['test'][file_path] = load_dataset(file_path, config['seed'])
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
        for abstract in raw_data:
            raw_data[abstract]['id'] = abstract
    datasets['raw']['test'][file_path] = raw_data
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


def encode_test_only(datasets, config, _log):
    """Encode only test data (for when training data is already processed)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_tokenizer = AutoTokenizer.from_pretrained(config['sentence_encoder'])
    bert_model = AutoModel.from_pretrained(config['sentence_encoder']).to(device)
    label2id = config['label2id']
    max_sentences = config['max_sentences']
    
    for corpus in datasets['test']:
        _log.info(f'Encoding {corpus}')
        data = datasets['test'][corpus]

        # Encode labels to one-hot vectors.
        data['y'] = data.labels.map(lambda row: [label2id[r] for r in row])
        data['y'] = data.y.map(
            lambda row: pad_sequences([row], maxlen=max_sentences)[0])
        y = to_categorical(data.y.to_list())
        datasets['y']['test'][corpus] = y

        # Encode sentences.
        data['sentence_embeddings'] = data.sentences.progress_map(
            lambda row: [
                encode_text(sentence, bert_tokenizer, bert_model, device)
                for sentence in row
            ])
        data['X'] = data.sentence_embeddings.map(
            lambda vec: pad_array(np.vstack(vec), max_sentences))

        X = np.vstack(data.X.to_numpy()).reshape(
            (len(data), max_sentences, 768))
        datasets['X']['test'][corpus] = X
        datasets['ids']['test'][corpus] = data['id'].values

    X_test = stack_encoded_datasets(datasets, 'X', 'test')
    y_test = stack_encoded_datasets(datasets, 'y', 'test')
    id_test = stack_encoded_datasets(datasets, 'ids', 'test')

    return dict(X=X_test, y=y_test, ids=id_test)


# Argument Quality Evaluation model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_saved_model(model_save_path, base_model):
    """Load the quality evaluation model.
    
    If the model file doesn't exist, instructions are provided to generate
    training data and train the model using GPT-5-mini.
    """
    model = ModelQualityPredictor(base_model)
    
    if not os.path.exists(model_save_path):
        print(f"⚠ Quality model not found at {model_save_path}")
        print("  To create it, run:")
        print("    cd data && python generate_quality_data.py && python train_quality_model.py")
        print("  Using untrained model - quality predictions will be random")
    else:
        try:
            state_dict = torch.load(model_save_path, map_location=torch.device('cpu'), weights_only=False)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded quality model from {model_save_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load quality model: {e}")
            print("  Using untrained model - quality predictions will be random")
            print("  To retrain: cd data && python generate_quality_data.py && python train_quality_model.py")
    
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


def extract_claims_and_premises(claim_extractor_model, X_test, raw_test_data, id_test):
    """
    Use a SINGLE claim extractor model to extract claims and premises from all test samples.
    This ensures no label leakage - the same model is used regardless of true label.
    """
    logits = claim_extractor_model(X_test)
    y_pred_unsorted = softmax(logits).argmax(-1)
    id_test = np.array(id_test)
    y_pred_unsorted = np.array(y_pred_unsorted)
    sort_indices = np.argsort(id_test)
    y_pred = y_pred_unsorted[sort_indices]
    raw_test_data.sort(key=lambda x: x[0])  # sorts in-place
    
    extracted_texts = []
    
    for idx, prediction in enumerate(y_pred):
        prediction = [i for i in prediction if i != 0]
        raw_text = raw_test_data[idx][1]  # sentences

        claims = ''
        premises = ''
        
        for id_pred, pred in enumerate(prediction):
            if id_pred >= len(raw_text):
                print(id_pred, raw_text)
            elif len(raw_text[id_pred]) > 1:
                if pred == 2:  # EVIDENCE
                    if raw_text[id_pred][-1] == '.':
                        premises += (raw_text[id_pred] + ' ')
                    else:
                        premises += (raw_text[id_pred] + '. ')
                if pred == 3:  # CLAIM
                    if raw_text[id_pred][-1] == '.':
                        claims += (raw_text[id_pred] + ' ')
                    else:
                        claims += (raw_text[id_pred] + '. ')

        text = claims + premises
        extracted_texts.append(text)
    
    return extracted_texts


@ex.main
def main(_config, _log):
    seed = _config['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    tf.random.set_seed(seed)

    # =========================================================================
    # STEP 1: Load and train the claim extractor model ONCE
    # =========================================================================
    _log.info("Loading training data for argument mining model...")
    train_datasets = load_training_datasets(_config)
    
    _log.info("Encoding training data...")
    encoded_train = encode_datasets(train_datasets, _config, _log)
    X_train = encoded_train['X']['train']
    y_train = encoded_train['y']['train']
    X_dev = encoded_train['X']['dev']
    
    if X_dev is not None:
        y_dev = encoded_train['y']['dev']
        validation_data = (X_dev, y_dev)
    else:
        validation_data = None

    # Create and train ONE claim extractor model
    _log.info("Training claim extractor model (SINGLE model for all test data)...")
    claim_extractor = bert_transformer(
        sents_shape=_config['sents_shape'],
        num_classes=_config['num_classes'],
        transformer_params=_config['transformer'])
    print(claim_extractor.summary(100))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=_config['save_model'], save_best_only=True, verbose=1)
    ]
    
    if validation_data is None:
        claim_extractor.fit(
            X_train,
            y_train,
            callbacks=callbacks,
            **_config['model_fit'],
            verbose=2)
    else:
        claim_extractor.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            callbacks=callbacks,
            epochs=_config['model_fit']['epochs'],
            verbose=2)

    # =========================================================================
    # STEP 2: Load the quality evaluation model
    # =========================================================================
    _log.info("Loading quality evaluation model...")
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    base_model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    model_save_path = '../quality_evaluation_model.pt'
    quality_model = load_saved_model(model_save_path, base_model)

    # =========================================================================
    # STEP 3: Process FRAUDULENT test data using the SAME claim extractor
    # =========================================================================
    _log.info("Processing fraudulent test data...")
    fraudulent_datasets = load_test_dataset(_config['datasets_fraudulent'][0], _config)
    
    raw_fraudulent_data = []
    for val in fraudulent_datasets['raw']['test'].values():
        for v in list(val.values()):
            raw_fraudulent_data.append((v['id'], v['sentences']))
    
    encoded_fraudulent = encode_test_only(fraudulent_datasets, _config, _log)
    X_test_fraudulent = encoded_fraudulent['X']
    id_test_fraudulent = encoded_fraudulent['ids']
    
    # Extract claims using the SAME model
    fraudulent_texts = extract_claims_and_premises(
        claim_extractor, X_test_fraudulent, raw_fraudulent_data, id_test_fraudulent)
    
    # Classify using quality model
    fraudulent_predictions = []
    for text in fraudulent_texts:
        predicted_class, _ = infer(quality_model, text, tokenizer)
        fraudulent_predictions.append(predicted_class)

    # =========================================================================
    # STEP 4: Process LEGITIMATE test data using the SAME claim extractor
    # =========================================================================
    _log.info("Processing legitimate test data...")
    legitimate_datasets = load_test_dataset(_config['datasets_legitimate'][0], _config)
    
    raw_legitimate_data = []
    for val in legitimate_datasets['raw']['test'].values():
        for v in list(val.values()):
            raw_legitimate_data.append((v['id'], v['sentences']))
    
    encoded_legitimate = encode_test_only(legitimate_datasets, _config, _log)
    X_test_legitimate = encoded_legitimate['X']
    id_test_legitimate = encoded_legitimate['ids']
    
    # Extract claims using the SAME model (no retraining!)
    legitimate_texts = extract_claims_and_premises(
        claim_extractor, X_test_legitimate, raw_legitimate_data, id_test_legitimate)
    
    # Classify using quality model
    legitimate_predictions = []
    for text in legitimate_texts:
        predicted_class, _ = infer(quality_model, text, tokenizer)
        legitimate_predictions.append(predicted_class)

    # =========================================================================
    # STEP 5: Calculate metrics
    # =========================================================================
    # For fraudulent samples: predicted_class=0 means correctly identified as fraudulent
    # For legitimate samples: predicted_class=1 means correctly identified as legitimate
    
    fraudulent_t = fraudulent_predictions.count(0)  # True positives (fraudulent correctly identified)
    fraudulent_f = fraudulent_predictions.count(1)  # False negatives (fraudulent missed)
    
    legitimate_t = legitimate_predictions.count(1)  # True negatives (legitimate correctly identified)
    legitimate_f = legitimate_predictions.count(0)  # False positives (legitimate misclassified as fraudulent)

    fraudulent_acc = fraudulent_t / len(fraudulent_predictions) if fraudulent_predictions else 0
    legitimate_acc = legitimate_t / len(legitimate_predictions) if legitimate_predictions else 0

    total_samples = fraudulent_t + fraudulent_f + legitimate_t + legitimate_f
    total_acc = (fraudulent_t + legitimate_t) / total_samples if total_samples > 0 else 0
    
    # Precision: TP / (TP + FP)
    precision = fraudulent_t / (fraudulent_t + legitimate_f) if (fraudulent_t + legitimate_f) > 0 else 0
    # Recall: TP / (TP + FN)
    recall = fraudulent_t / (fraudulent_t + fraudulent_f) if (fraudulent_t + fraudulent_f) > 0 else 0
    # F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*60)
    print("RESULTS (Using SINGLE claim extractor - no label leakage)")
    print("="*60)
    print(f"\nNumber of articles correctly classified:")
    print(f"  Fraudulent (True Positives): {fraudulent_t}")
    print(f"  Legitimate (True Negatives): {legitimate_t}")
    print(f"\nNumber of articles falsely classified:")
    print(f"  Fraudulent as Legitimate (False Negatives): {fraudulent_f}")
    print(f"  Legitimate as Fraudulent (False Positives): {legitimate_f}")
    print(f"\nMetrics:")
    print(f"  Fraudulent accuracy (Recall/Sensitivity): {fraudulent_acc:.4f}")
    print(f"  Legitimate accuracy (Specificity): {legitimate_acc:.4f}")
    print(f"  Overall accuracy: {total_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("="*60)

    return {
        'fraudulent_accuracy': fraudulent_acc,
        'legitimate_accuracy': legitimate_acc,
        'overall_accuracy': total_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


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
        default='trained_models/bert_based_context_encoder.keras',
        help='The file name for the saved model. '
        'The file name may have a path and *should* have `.keras` extension.')
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
        if 'datasets_train' not in config_update:
            config_update['datasets_train'] = dict()
        config_update['datasets_train']['train'] = args.train_set
    if args.development_set is not None:
        if 'datasets_train' not in config_update:
            config_update['datasets_train'] = dict()
        config_update['datasets_train']['dev'] = args.development_set
    if args.test_set is not None:
        if 'datasets_train' not in config_update:
            config_update['datasets_train'] = dict()
        config_update['datasets_train']['test'] = args.test_set
    ex.run(config_updates=config_update)
