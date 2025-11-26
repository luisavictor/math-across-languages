from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import re
import gc

# ---- Configuration ----
TARGET_LANGUAGE = "German"
LANG_CODE = "de"
NLLB_LANG_CODE = "deu_Latn"  # NLLB language code for German

tok = "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"

# Use NLLB-200-3.3B for high-quality translation
MODEL_NAME = "facebook/nllb-200-3.3B"
INPUT_CSV = "MathNeuro/data/race.csv"
OUTPUT_DIR = "/home/iailab34/selbacht0/Test_Lab/LabTest/"

# ---- Step 1: Load quantized model ----
print("Loading NLLB-200-3.3B model...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    src_lang="eng_Latn",
    token=tok
)
print(f"Tokenizer max length: {tokenizer.model_max_length}")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=tok
)

print("Model loaded successfully!")

# ---- Step 2: Helper function to reset model state ----
def reset_model_state():
    """Reset model's internal state to prevent context carryover"""
    if hasattr(model, 'past_key_values'):
        model.past_key_values = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()

# ---- Step 3: Token monitoring helper ----
def get_token_count(text):
    """Get the number of tokens for a given text"""
    if not text or pd.isna(text):
        return 0
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def log_token_stats(stage, text, token_count):
    """Log token statistics for monitoring"""
    char_count = len(text) if text else 0
    print(f"  [{stage}] Characters: {char_count:,} | Tokens: {token_count:,}")

# ---- Step 4: Translation function with sentence-by-sentence approach ----
def translate_text_chunk(text, max_tokens=1024):
    """Translate a single chunk of text (helper function)"""
    if not text or not text.strip():
        return ""
    
    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        padding=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(NLLB_LANG_CODE),
            max_length=max_tokens,
            num_beams=5,
            early_stopping=True,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation.strip()

def split_into_sentences(text):
    """Split text into sentences for better translation"""
    # First, normalize the text by adding space after punctuation if missing
    # Handle cases like "sentence.Another" -> "sentence. Another"
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # Split on period, exclamation, question mark followed by space and capital letter
    # Also handle quotes and parentheses
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\(\[])', text)
    return [s.strip() for s in sentences if s.strip()]

def translate_text(text):
    """Translate English text to German using NLLB-200 with sentence-by-sentence strategy"""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return ""
    
    # Reset model state BEFORE each translation
    reset_model_state()
    
    # Monitor input text
    input_token_count = get_token_count(text)
    print(f"\n🔍 Translation Input Monitoring:")
    log_token_stats("Input Text", text, input_token_count)
    
    try:
        # ALWAYS use sentence-by-sentence translation for consistency and quality
        print(f"  [Strategy] Sentence-by-sentence translation")
        sentences = split_into_sentences(text)
        print(f"  [Sentences] Split into {len(sentences)} sentences")
        
        translated_sentences = []
        for i, sentence in enumerate(sentences):
            sentence_tokens = get_token_count(sentence)
            print(f"    [Sentence {i+1}/{len(sentences)}] Tokens: {sentence_tokens}")
            
            if sentence_tokens > 800:
                # If a single sentence is too long, split it by clauses
                # Split on commas, semicolons, colons, dashes
                parts = re.split(r'([,;:\-—])\s+', sentence)
                
                # Reconstruct parts with their punctuation
                chunks = []
                current_chunk = ""
                for j, part in enumerate(parts):
                    if current_chunk and get_token_count(current_chunk + part) > 700:
                        chunks.append(current_chunk.strip())
                        current_chunk = part
                    else:
                        current_chunk += part
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                print(f"      [Long sentence] Split into {len(chunks)} chunks")
                
                # Translate each chunk
                translated_parts = []
                for k, chunk in enumerate(chunks):
                    chunk_translation = translate_text_chunk(chunk, max_tokens=1024)
                    translated_parts.append(chunk_translation)
                    print(f"        [Chunk {k+1}/{len(chunks)}] Translated")
                
                translated_sentence = " ".join(translated_parts)
            else:
                translated_sentence = translate_text_chunk(sentence, max_tokens=1024)
            
            translated_sentences.append(translated_sentence)
            
            # Reset model state periodically to prevent memory issues
            if i % 10 == 0 and i > 0:
                reset_model_state()
        
        # Join translated sentences
        translation = " ".join(translated_sentences)
        
        # Monitor output
        output_token_count = get_token_count(translation)
        log_token_stats("Translation Output", translation, output_token_count)
        print(f"  [Ratio] Output/Input tokens: {output_token_count/max(input_token_count, 1):.2f}x")
        print(f"  [Preview] {translation[:200]}...")
        
        return translation.strip()
    except Exception as e:
        print(f"❌ Error during translation: {e}")
        import traceback
        traceback.print_exc()
        return ""
    finally:
        # Reset model state AFTER each translation
        reset_model_state()

def translate_problems_field(problems_text):
    """Parse and translate the problems field which contains a list of dictionaries"""
    if pd.isna(problems_text) or not isinstance(problems_text, str) or not problems_text.strip():
        return ""
    
    print(f"\n📋 Problems Field Processing:")
    
    try:
        # Parse the string representation of the list
        import ast
        problems_list = ast.literal_eval(problems_text)
        
        if not isinstance(problems_list, list):
            print("  ⚠️ Not a list, translating as plain text")
            return translate_text(problems_text)
        
        print(f"  [Problems] Found {len(problems_list)} question(s)")
        
        translated_problems = []
        for i, problem in enumerate(problems_list):
            print(f"    [Problem {i+1}/{len(problems_list)}]")
            
            translated_problem = {}
            
            # Translate the question
            if 'question' in problem:
                print(f"      - Translating question...")
                translated_problem['question'] = translate_text(problem['question'])
            
            # Keep the answer as-is (it's usually a letter like 'A', 'B', 'C', 'D')
            if 'answer' in problem:
                translated_problem['answer'] = problem['answer']
            
            # Translate each option
            if 'options' in problem and isinstance(problem['options'], list):
                print(f"      - Translating {len(problem['options'])} options...")
                translated_options = []
                for j, option in enumerate(problem['options']):
                    translated_option = translate_text(option)
                    translated_options.append(translated_option)
                    print(f"        Option {j+1}: Done")
                translated_problem['options'] = translated_options
            
            translated_problems.append(translated_problem)
        
        # Convert back to string representation
        result = str(translated_problems)
        print(f"  ✅ Translated {len(translated_problems)} problem(s)")
        return result
        
    except Exception as e:
        print(f"  ❌ Error parsing problems field: {e}")
        print(f"  Falling back to plain text translation")
        return translate_text(problems_text)

def translate_qa_field(qa_text, translated_article):
    """Parse QA field, reuse translated article, and translate only Q&A parts"""
    if pd.isna(qa_text) or not isinstance(qa_text, str) or not qa_text.strip():
        return ""
    
    print(f"\n📋 QA Field Processing:")
    
    try:
        # Split the QA text to find the structure
        lines = qa_text.split('\n')
        
        # Find the intro line (usually first line)
        intro = ""
        article_start_idx = -1
        question_start_idx = -1
        answer_choices_idx = -1
        answer_idx = -1
        
        for i, line in enumerate(lines):
            if i == 0 and line.strip() and not line.strip().startswith('Question:'):
                intro = line.strip()
            elif line.strip().startswith('Question:'):
                question_start_idx = i
            elif line.strip().startswith('Answer choices:'):
                answer_choices_idx = i
            elif line.strip().startswith('Answer:'):
                answer_idx = i
        
        # If we can't find the question, fallback to simple translation
        if question_start_idx == -1:
            print("  ⚠️ Could not parse QA structure, using simple translation")
            return translate_text(qa_text)
        
        # Extract components
        # Intro (first line before article)
        if not intro and len(lines) > 0:
            intro = lines[0].strip()
        
        # Question
        question_text = ""
        if question_start_idx >= 0:
            question_line = lines[question_start_idx]
            question_text = question_line.replace('Question:', '').strip()
        
        # Answer choices
        answer_choices_text = ""
        if answer_choices_idx >= 0:
            choices_line = lines[answer_choices_idx]
            answer_choices_text = choices_line.replace('Answer choices:', '').strip()
        
        # Answer
        answer_text = ""
        if answer_idx >= 0:
            answer_line = lines[answer_idx]
            answer_text = answer_line.replace('Answer:', '').strip()
        
        print(f"  [Intro] Found: {intro[:50]}...")
        print(f"  [Question] Found: {question_text[:50]}...")
        print(f"  [Answer choices] Found: {answer_choices_text[:50] if answer_choices_text else 'None'}...")
        print(f"  [Answer] Found: {answer_text[:50] if answer_text else 'None'}...")
        
        # Translate components
        print(f"  [Translating] Intro...")
        intro_translated = translate_text(intro) if intro else ""
        
        # Hard-coded German labels (no model translation needed)
        question_label = "Frage"
        answer_choices_label = "Antwortmöglichkeiten"
        answer_label = "Antwort"
        
        print(f"  [Translating] Question...")
        question_translated = translate_text(question_text) if question_text else ""
        
        # Translate answer choices
        answer_choices_translated = ""
        if answer_choices_text:
            try:
                print(f"  [Translating] Answer choices...")
                import ast
                # Parse the list of choices
                choices_list = ast.literal_eval(answer_choices_text)
                if isinstance(choices_list, list):
                    translated_choices = []
                    for i, choice in enumerate(choices_list):
                        print(f"    - Choice {i+1}/{len(choices_list)}...")
                        translated_choice = translate_text(choice)
                        translated_choices.append(translated_choice)
                    answer_choices_translated = str(translated_choices)
                else:
                    answer_choices_translated = answer_choices_text
            except:
                print(f"  ⚠️ Could not parse answer choices, keeping as-is")
                answer_choices_translated = answer_choices_text
        
        print(f"  [Translating] Answer...")
        answer_translated = translate_text(answer_text) if answer_text else ""
        
        # Reconstruct the QA field with TRANSLATED labels and the ALREADY TRANSLATED article
        result = f"{intro_translated}\n\n{translated_article}\n\n{question_label}: {question_translated}\n\n"
        
        if answer_choices_translated:
            result += f"{answer_choices_label}: {answer_choices_translated}\n\n"
        
        result += f"{answer_label}: {answer_translated}"
        
        print(f"  ✅ QA field reconstructed with translated labels and article")
        return result
        
    except Exception as e:
        print(f"  ❌ Error parsing QA field: {e}")
        import traceback
        traceback.print_exc()
        print(f"  Falling back to simple translation")
        return translate_text(qa_text)

# ---- Step 5: Main processing ----
def main():
    print(f"\n{'='*60}")
    print(f"Processing German translation with NLLB-200-3.3B (sentence-by-sentence)...")
    print(f"{'='*60}")
    
    output_path = Path(OUTPUT_DIR) / f"race_nllb_sentences_{LANG_CODE}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load original data
    print(f"Loading original CSV from {INPUT_CSV}")
    original_df = pd.read_csv(INPUT_CSV)
    
    # Create fresh dataframe for translation
    df = pd.DataFrame()
    df['article'] = ""
    df['problems'] = ""
    df['qa'] = ""
    
    # Copy non-translation columns from original
    for col in original_df.columns:
        if col not in ['article', 'problems', 'qa']:
            df[col] = original_df[col]
    
    # Ensure we have the same number of rows
    df = df.reindex(range(len(original_df)))
    
    # Translate each row
    for idx in tqdm(range(len(df)), desc=f"Translating to {TARGET_LANGUAGE}"):
        row_changed = False
        translated_article = ""
        
        print(f"\n{'─'*60}")
        print(f"Processing Row {idx + 1}/{len(df)}")
        print(f"{'─'*60}")
        
        # Step 1: Translate article (model state reset inside translate_text)
        if idx < len(original_df):
            print(f"\n📰 Step 1: Translating Article")
            original_article = original_df.at[idx, 'article']
            article_tokens = get_token_count(original_article)
            log_token_stats("Original Article", original_article, article_tokens)
            
            translated_article = translate_text(original_article)
            df.at[idx, 'article'] = translated_article
            row_changed = True
        
        # Step 2: Translate problems (model state reset inside translate_text)
        if idx < len(original_df):
            print(f"\n🔢 Step 2: Translating Problems")
            original_problems = original_df.at[idx, 'problems']
            
            translated = translate_problems_field(original_problems)
            df.at[idx, 'problems'] = translated
            row_changed = True
        
        # Step 3: Translate qa (model state reset for each component inside translate_text)
        if idx < len(original_df):
            print(f"\n❓ Step 3: Translating Q&A")
            original_qa = original_df.at[idx, 'qa']
            translated = translate_qa_field(original_qa, translated_article)
            df.at[idx, 'qa'] = translated
            row_changed = True
        
        # Save progress every row
        if row_changed and idx % 1 == 0:
            df.to_csv(output_path, index=False)
            print(f"\n💾 Progress saved")
    
    # Final save
    df.to_csv(output_path, index=False)
    print(f"\n✅ {TARGET_LANGUAGE} translation completed and saved to {output_path}")
    
    print("\n" + "="*60)
    print("Translation completed!")
    print("="*60)

if __name__ == "__main__":
    main()
