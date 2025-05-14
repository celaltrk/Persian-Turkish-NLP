import os
import re
import logging
from tqdm import tqdm
from pathlib import Path
from gensim.models import FastText
from zemberek import (
    TurkishSpellChecker,
    TurkishSentenceNormalizer,
    TurkishSentenceExtractor,
    TurkishMorphology,
    TurkishTokenizer
)
from collections import Counter
from pathlib import Path


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- Configuration ---jpype.startJVM(jpype.getDefaultJVMPath(), "-Djava.class.path=/path/to/zemberek.jar")

DATA_DIR = Path("./data")
MODEL_OUTPUT_DIR = Path("./embedding_models")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


NUM_WORKERS = os.cpu_count() if os.cpu_count() else 4 # Use available CPU cores
NUM_WORKERS = NUM_WORKERS - 3 # prevent excessive load
MIN_WORD_COUNT = 5

logging.info(f"Number of workers is {NUM_WORKERS}")

# stopword list from https://github.com/ahmetax/trstop/blob/master/dosyalar/turkce-stop-words
STOPWORDS = [
    'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'bile', 'bir', 'biraz', 'birçoğu',
    'birçok', 'birisi', 'birkaç', 'birşey', 'biz', 'bizden', 'bize', 'bizi', 'bizim', 'bu',
    'buna', 'bundan', 'bunlar', 'bunları', 'bunların', 'bunu', 'bunun', 'burada', 'böyle',
    'böylece', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep',
    'hepsi', 'her', 'herkes', 'hiç', 'hiçbir', 'için', 'ile', 'ise', 'içinde', 'kadar', 'ki',
    'kim', 'kimse', 'mı', 'mi', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nereye', 'niye',
    'niçin', 'o', 'olan', 'olarak', 'oldu', 'olduğu', 'olmak', 'olmaz', 'olsun', 'on', 'ona',
    'ondan', 'onlar', 'onlardan', 'onları', 'onların', 'onu', 'onun', 'orada', 'sanki',
    'sadece', 'sen', 'senden', 'sende', 'seni', 'senin', 'siz', 'sizden', 'size', 'sizi',
    'sizin', 'şey', 'şu', 'şuna', 'şunda', 'şundan', 'şunları', 'şunlar', 'şunu', 'şunun',
    'ta', 'tamam', 'tüm', 've', 'veya', 'ya', 'yani'
]

try:
    morphology = TurkishMorphology.create_with_defaults()
    normalizer = TurkishSentenceNormalizer(morphology)
    spell_checker = TurkishSpellChecker(morphology)
    extractor = TurkishSentenceExtractor()
    tokenizer = TurkishTokenizer.DEFAULT
    logging.info("Zemberek tools initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Zemberek: {e}")
    morphology = normalizer = spell_checker = extractor = tokenizer = None

def preprocess_turkish_text(text_lines):
    processed_sentences = []
    vocab_counts = Counter()

    for line_num, line in enumerate(tqdm(text_lines, desc="Preprocessing")):
        if not line.strip():
            continue
        
        sentences = extractor.from_paragraph(line.strip())
        unwanted_token_types = ['Punctuation', 'Emoticon', 'UnknownWord', 'Number', 'SpaceTab', 'NewLine', 'RomanNumeral', 'PercentNumeral', 'Time', 'Date', 'URL', 'Email', 'HashTag', 'Mention', 'MetaTag', 'Emoji', 'Emoticon', 'UnknownWord', 'Unknown']
        for sentence in sentences:
            # Tokenization
            tokens = tokenizer.tokenize(sentence)
            token_list = [token.content for token in tokens if token.type_.name not in unwanted_token_types and token.content.lower() not in STOPWORDS]
            if not token_list:
                continue
            
            normalized_text = normalizer.normalize(' '.join(token_list))
            normalized_tokens = tokenizer.tokenize(normalized_text)
            normalized_tokens = [token.content.lower() for token in normalized_tokens if token.type_.name not in unwanted_token_types and token.content.lower() not in STOPWORDS]
            
            # Spell correction
            corrected_tokens = []
            for token in normalized_tokens:
                results = morphology.analyze(token)
                if not results or len(str(results)) < 5:
                    suggestions = spell_checker.suggest_for_word(token)
                    token = suggestions[0] if suggestions else token
                corrected_tokens.append(token)

            # Lemmatization
            lemmatized_tokens = []
            for token in corrected_tokens:
                results = morphology.analyze(token)
                largest_lemma = ''
                largest_lemma_length = 0
                for result in results:
                    analysis_str = str(result)
                    if analysis_str.startswith('['):
                        lemma_section = analysis_str.split(']')[0]
                        lemma = lemma_section.split(':')[0].lstrip('[')
                    else:
                        lemma = analysis_str.split(':')[0]
                    if (len(lemma) > largest_lemma_length):
                        largest_lemma = lemma
                if largest_lemma:
                    lemmatized_tokens.append(largest_lemma.lower())

            final_tokens = [token for token in lemmatized_tokens if token not in STOPWORDS]

            # Update vocabulary counts
            if final_tokens:
                vocab_counts.update(final_tokens)
                processed_sentences.append(final_tokens)

    # Vocabulary pruning
    shared_vocab = {word for word, count in vocab_counts.items() if count >= MIN_WORD_COUNT}
    final_sentences = []
    for sentence in processed_sentences:
        final_sentence = [word if word in shared_vocab else '<UNK>' for word in sentence]
        if final_sentence and any(word != '<UNK>' for word in final_sentence):
            final_sentences.append(final_sentence)

    logging.info(f"Processed {len(final_sentences)} sentences with {len(shared_vocab)} unique words")
    decade_name = 'test'
    preprocessed_file_path = MODEL_OUTPUT_DIR / f"preprocessed_{decade_name}.txt"
    try:
        with open(preprocessed_file_path, 'w', encoding='utf-8') as f:
            for sentence in processed_sentences:
                f.write(' '.join(sentence) + '\n')
        logging.info(f"Preprocessed sentences saved to {preprocessed_file_path}")
    except Exception as e:
        logging.error(f"Could not save preprocessed sentences for {decade_name}: {e}")
        
    return processed_sentences

def preprocess_all_decade_files(decade_files):
    for file_path in decade_files:
        decade_name = file_path.stem
        logging.info(f"--- Preprocessing for {decade_name} ---")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logging.error(f"Could not read {file_path}: {e}")
            continue

        if not lines:
            logging.warning(f"{file_path} is empty. Skipping.")
            continue

        processed_sentences = preprocess_turkish_text(lines)

        if not processed_sentences:
            logging.warning(f"No valid sentences found in {file_path}. Skipping save.")
            continue

        save_path = MODEL_OUTPUT_DIR / f"preprocessed_{decade_name}.txt"
        try:
            with open(save_path, 'w', encoding='utf-8') as out_f:
                for sentence in processed_sentences:
                    out_f.write(' '.join(sentence) + '\n')
            logging.info(f"Saved preprocessed file: {save_path}")
        except Exception as e:
            logging.error(f"Could not save preprocessed file {save_path}: {e}")

DATA_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.is_dir():
    logging.error(f"Data directory not found: {DATA_DIR}")
    logging.error("Please create a 'data' directory and place your decade .txt files (e.g., 1930s.txt) in it.")
    raise FileNotFoundError("Data directory not found")


decade_files = sorted(DATA_DIR.glob("*.txt"))

if not decade_files:
    logging.warning(f"No .txt files found in {DATA_DIR}. Nothing to process.")
    raise FileNotFoundError("No .txt files found in data directory")

logging.info(f"Found {len(decade_files)} decade files to process: {[f.name for f in decade_files]}")

preprocess_all_decade_files(decade_files)