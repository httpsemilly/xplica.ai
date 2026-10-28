import os
import re
import numpy as np
import spacy
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TOKENIZER_NAME = 'neuralmind/bert-base-portuguese-cased'
MAX_LENGTH = 512
LABELS = ['fake', 'true']
LIME_SEED = 42
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_NUM_FEATURES = 10
MAX_CHARS = 20_000

# Mesma definicao usada no notebook 03_explainability (Secao 6.2 do artigo)
WEEKDAYS = re.compile(r'\b(?:segunda|terça|quarta|quinta|sexta)-feira\b|\bsábado\b|\bdomingo\b', re.IGNORECASE)


def compute_linguistic_features(nlp, text):
  """Caracteristicas linguisticas do artigo (as 10 da analise + noun/loc/misc).

  Separada da classe Analyzer para poder ser usada sem carregar o BERTimbau
  (por exemplo, no calculo das medianas de referencia em compute_baselines.py).
  """
  doc = nlp(text)

  total_tokens = len([t for t in doc if not t.is_punct and not t.is_space])

  if total_tokens == 0:
    return {}

  pos_counts = {pos: sum(1 for t in doc if t.pos_ == pos) for pos in ['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']}

  ent_counts = {'PER': 0, 'ORG': 0, 'LOC': 0, 'MISC': 0}

  for ent in doc.ents:
    if ent.label_ in ent_counts:
      ent_counts[ent.label_] += 1

  sentences = list(doc.sents)
  avg_sentence_length = total_tokens / len(sentences) if sentences else 0

  denominator = pos_counts['NOUN'] + pos_counts['VERB']
  emotiveness = (pos_counts['ADJ'] + pos_counts['ADV']) / denominator if denominator > 0 else 0

  words = re.findall(r'\w+', text.lower())
  stop_words = nlp.Defaults.stop_words

  return {
    'propn_ratio': pos_counts['PROPN'] / total_tokens,
    'noun_ratio': pos_counts['NOUN'] / total_tokens,
    'adj_ratio': pos_counts['ADJ'] / total_tokens,
    'verb_ratio': pos_counts['VERB'] / total_tokens,
    'adv_ratio': pos_counts['ADV'] / total_tokens,
    'per_ratio': ent_counts['PER'] / total_tokens,
    'org_ratio': ent_counts['ORG'] / total_tokens,
    'loc_ratio': ent_counts['LOC'] / total_tokens,
    'misc_ratio': ent_counts['MISC'] / total_tokens,
    'avg_sentence_length': avg_sentence_length,
    'emotiveness': emotiveness,
    'stop_ratio': sum(w in stop_words for w in words) / max(len(words), 1),
    'weekday_ratio': len(WEEKDAYS.findall(text)) / total_tokens,
    'total_tokens': total_tokens
  }


class Analyzer:
  def __init__(self, model_repo=None, device=None):
    model_repo = model_repo or os.environ.get('XPLICA_MODEL_REPO')

    if not model_repo:
        raise RuntimeError(
            'Set XPLICA_MODEL_REPO (e.g. "user/xplica-bertimbau-scenario3") '
            'or pass a local path.'
        )

    self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, do_lower_case=False)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_repo).to(self.device).eval()

    id2label = getattr(self.model.config, 'id2label', None)

    if id2label and not all(str(v).startswith('LABEL_') for v in id2label.values()):
        self.labels = [str(id2label[i]) for i in sorted(id2label, key=int)]
    else:
        self.labels = LABELS

    try:
        self.nlp = spacy.load('pt_core_news_sm')
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'pt_core_news_sm' not found. Install with: "
            "python -m spacy download pt_core_news_sm"
        ) from e

  def predict_proba(self, texts, batch_size=32):
    texts = list(texts)
    all_probs = []

    for i in range(0, len(texts), batch_size):
        inputs = self.tokenizer(
            texts[i:i + batch_size],
            padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

    return np.vstack(all_probs)

  def explain(self, text, num_samples=DEFAULT_NUM_SAMPLES, num_features=DEFAULT_NUM_FEATURES):
    explainer = LimeTextExplainer(class_names=self.labels, random_state=LIME_SEED)
    exp = explainer.explain_instance(text, self.predict_proba, labels=(1,), num_features=num_features, num_samples=num_samples)

    return [{'word': str(word), 'weight': float(weight)} for word, weight in exp.as_list(label=1)]

  def linguistic_features(self, text):
    return compute_linguistic_features(self.nlp, text)

  def analyze(self, text, num_samples=DEFAULT_NUM_SAMPLES, num_features=DEFAULT_NUM_FEATURES):
    text = (text or '').strip()

    if not text:
        raise ValueError('Empty text.')

    if len(text) > MAX_CHARS:
        raise ValueError(f'Text too long (max {MAX_CHARS} characters).')

    probs = self.predict_proba([text])[0]
    n_tokens = len(self.tokenizer(text, truncation=False, add_special_tokens=False)['input_ids'])

    return {
        'label': self.labels[int(probs.argmax())],
        'prob_fake': float(probs[0]),
        'prob_true': float(probs[1]),
        'n_tokens': n_tokens,
        'truncated': n_tokens + 2 > MAX_LENGTH,
        'lime': self.explain(text, num_samples, num_features),
        'linguistic': self.linguistic_features(text)
    }
