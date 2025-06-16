import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and being used.")
else:
    print("No GPU detected or being used by TensorFlow.")

import os
import random
import numpy as np
import pretty_midi
from glob import glob
from tensorflow.keras.models import Sequential, load_model # <-- Added load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding
import time

# Suppress TensorFlow informational messages about oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ===== Step 1: Extract Notes from MIDI Files =====
def extract_notes(midi_path):
    """
    Extracts pitch notes from a given MIDI file.

    Args:
        midi_path (str): The path to the MIDI file.

    Returns:
        list: A list of integer pitches extracted from the MIDI file.
              Returns an empty list if an error occurs during processing.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for instrument in midi.instruments:
            # Only consider non-drum instruments
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append(note.pitch)
        return notes
    except Exception as e:
        print(f"[Error] Failed to process {midi_path}: {e}")
        return []

# ===== Step 2: Prepare Sequences =====
def prepare_sequences(notes, seq_len=50):

    if not notes:
        print("Warning: Input notes list is empty. Cannot prepare sequences.")
        return None, None, None, None

    # Ensure there are enough notes to form at least one sequence
    if len(notes) < seq_len + 1:
        print(f"Warning: Not enough total notes ({len(notes)}) to create sequences of length {seq_len}. Skipping.")
        return None, None, None, None

    unique_notes = sorted(list(set(notes)))
    if not unique_notes:
        print("Warning: No unique notes found after filtering. Cannot prepare sequences.")
        return None, None, None, None

    note_to_int = {note: number for number, note in enumerate(unique_notes)}
    int_to_note = {number: note for note, number in note_to_int.items()}

    sequences = []
    next_notes = []
    for i in range(len(notes) - seq_len):
        seq = notes[i:i + seq_len]
        label = notes[i + seq_len]
        sequences.append([note_to_int[n] for n in seq])
        next_notes.append(note_to_int[label])

    X = np.array(sequences)
    y = np.array(next_notes)

    if X.shape[0] == 0:
        print(f"Warning: No valid sequences could be formed with sequence length {seq_len}. Try reducing seq_len.")
        return None, None, None, None

    return X, y, note_to_int, int_to_note

# ===== Step 3: Build the Model =====
def build_model(vocab_size, seq_len):
    """
    Builds a Sequential Keras model with Embedding and LSTM layers.

    Args:
        vocab_size (int): The total number of unique notes (vocabulary size).
        seq_len (int): The length of input sequences.

    Returns:
        tensorflow.keras.models.Sequential: The compiled Keras model.
    """
    if vocab_size <= 0:
        raise ValueError("Vocabulary size must be greater than 0 to build the model.")
    if seq_len <= 0:
        raise ValueError("Sequence length must be greater than 0 to build the model.")

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, input_length=seq_len, name='embedding_layer'),
        LSTM(256, return_sequences=True, name='lstm_layer_1'),
        LSTM(256, name='lstm_layer_2'),
        Dense(256, activation='relu', name='dense_layer_1'),
        Dense(vocab_size, activation='softmax', name='output_dense_layer')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model

# ===== Main Execution Block =====
if __name__ == "__main__":
    print("--- Starting MIDI LSTM Generator (Subset Mode) ---")

    # --- Configuration ---
    MIDI_DATA_PATH = "midi_dataset/"
    SEQUENCE_LENGTH = 50
    EPOCHS = 10 
    BATCH_SIZE = 64
    MODEL_OUTPUT_DIR = "model_output/"

    # Consistent names for saved files
    MODEL_SAVE_NAME = "music_generation_model_subset.keras"
    NOTE_TO_INT_SAVE_NAME = "note_to_int_subset.npy"
    INT_TO_NOTE_SAVE_NAME = "int_to_note_subset.npy"

    # New configuration for subset selection
    NUM_FILES_TO_PROCESS = 100
    RANDOM_SEED = 42

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    # --- Step 1: Collect all notes from a SUBSET of MIDI files ---
    print(f"\nSearching for MIDI files in: {os.path.abspath(MIDI_DATA_PATH)}")
    all_available_midi_files = glob(os.path.join(MIDI_DATA_PATH, "**/*.mid"), recursive=True)

    if not all_available_midi_files:
        print(f"Error: No MIDI files found in '{MIDI_DATA_PATH}'. Please ensure .mid files are placed inside this directory.")
        print("Exiting.")
        exit()

    print(f"Found {len(all_available_midi_files)} total MIDI files.")

    # Select a random subset of files
    if len(all_available_midi_files) > NUM_FILES_TO_PROCESS:
        midi_files_to_process = random.sample(all_available_midi_files, NUM_FILES_TO_PROCESS)
        print(f"Selected a random subset of {NUM_FILES_TO_PROCESS} files for processing.")
    else:
        midi_files_to_process = all_available_midi_files
        print(f"Using all {len(midi_files_to_process)} available files as it's less than or equal to {NUM_FILES_TO_PROCESS}.")

    print("Extracting notes from selected MIDI files (this might take a while)...")

    all_notes = []
    processed_count = 0
    start_time = time.time()

    for midi_file in midi_files_to_process:
        notes_from_file = extract_notes(midi_file)
        if notes_from_file:
            all_notes.extend(notes_from_file)
        else:
            print(f"Skipping {midi_file} due to extraction issues or no notes found.")

        processed_count += 1
        if processed_count % 50 == 0 or processed_count == len(midi_files_to_process):
            elapsed_time = time.time() - start_time
            if processed_count > 0:
                time_per_file = elapsed_time / processed_count
                remaining_files = len(midi_files_to_process) - processed_count
                estimated_remaining_time = remaining_files * time_per_file
                print(f"Processed {processed_count}/{len(midi_files_to_process)} files. "
                      f"Elapsed: {elapsed_time:.1f}s. "
                      f"Estimated remaining: {estimated_remaining_time:.1f}s")


    if not all_notes:
        print("Error: No notes were successfully extracted from any of the selected MIDI files. Exiting.")
        exit()

    print(f"Successfully extracted {len(all_notes)} notes in total from {len(midi_files_to_process)} files.")

    # --- Step 2: Prepare sequences for training ---
    print(f"\nPreparing sequences with length {SEQUENCE_LENGTH}...")
    X, y, note_to_int, int_to_note = prepare_sequences(all_notes, seq_len=SEQUENCE_LENGTH)

    if X is None or y is None:
        print("Error: Failed to prepare sequences. Exiting.")
        exit()

    current_vocab_size = len(note_to_int) # Renamed to avoid confusion with loaded vocab
    print(f"Prepared {len(X)} input-output pairs (sequences).")
    print(f"Current data vocabulary size (number of unique notes): {current_vocab_size}")
    print(f"Input shape (X): {X.shape}")
    print(f"Target shape (y): {y.shape}")

    
    ## Step 3: Load or Build and Train the Model


    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_SAVE_NAME)
    note_to_int_path = os.path.join(MODEL_OUTPUT_DIR, NOTE_TO_INT_SAVE_NAME)
    int_to_note_path = os.path.join(MODEL_OUTPUT_DIR, INT_TO_NOTE_SAVE_NAME)

    model = None
    loaded_existing_model = False
    
    # Check if a previously trained model and mappings exist
    if os.path.exists(model_path) and \
       os.path.exists(note_to_int_path) and \
       os.path.exists(int_to_note_path):
        print(f"\nFound existing model and mappings in '{MODEL_OUTPUT_DIR}'. Loading for continued training...")
        try:
            model = load_model(model_path)
            # Load the original mappings that the model was trained with
            loaded_note_to_int = np.load(note_to_int_path, allow_pickle=True).item()
            loaded_int_to_note = np.load(int_to_note_path, allow_pickle=True).item()

            if current_vocab_size != len(loaded_note_to_int):
                print(f"WARNING: Vocabulary size mismatch detected!")
                print(f"  New data unique notes: {current_vocab_size}")
                print(f"  Loaded model unique notes: {len(loaded_note_to_int)}")
                print("  This can cause issues. Consider retraining from scratch if vocabularies are significantly different,")
                print("  or ensure your data preparation consistently maps notes if training on varying subsets.")

            vocab_size_for_model = model.output_shape[1] # Get vocab size from the loaded model's output layer
            print(f"Model loaded successfully with output vocabulary size: {vocab_size_for_model}. Summary:")
            model.summary()
            loaded_existing_model = True

        except Exception as e:
            print(f"Error loading model or mappings: {e}")
            print("Building a new model from scratch.")
            # If loading fails, build a new model using the current data's vocabulary
            try:
                model = build_model(current_vocab_size, SEQUENCE_LENGTH)
                model.summary()
            except ValueError as e:
                print(f"Error building new model: {e}")
                print("Exiting.")
                exit()
    else:
        print("\nNo existing model found. Building a new Keras model...")
        try:
            model = build_model(current_vocab_size, SEQUENCE_LENGTH)
            model.summary()
        except ValueError as e:
            print(f"Error building model: {e}")
            print("Exiting.")
            exit()

    print(f"\nStarting model training for {EPOCHS} epochs with batch size {BATCH_SIZE}...")
    # Indicate if it's new training or continued
    if loaded_existing_model:
        print("Training will continue from the loaded model's last state.")
    else:
        print("Training a new model from scratch.")
    print("This will be significantly faster than with the full dataset.")

    try:
        # model.fit() automatically continues training when called on a loaded model.
        history = model.fit(X, y,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=1)

        print("\nTraining complete.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Exiting.")
        exit()


    ## Step 4: Save the Updated Model and Mappings
    

    print(f"\nSaving updated trained model and note mappings to '{MODEL_OUTPUT_DIR}'...")
    try:
        model.save(model_path)
        print(f"Model saved to: {model_path}")

        # Always save the *current* note_to_int and int_to_note,
        # as they reflect the vocabulary used for the *last* training pass
        # and are needed for generation.
        np.save(note_to_int_path, note_to_int)
        np.save(int_to_note_path, int_to_note)
        print(f"Note mappings ({NOTE_TO_INT_SAVE_NAME}, {INT_TO_NOTE_SAVE_NAME}) saved.")
    except Exception as e:
        print(f"Error saving model or mappings: {e}")

    print("\n--- Process Finished ---")
    print("Remember that a model trained on a subset might have limited generative capabilities compared to a full dataset.")
