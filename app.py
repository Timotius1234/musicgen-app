import streamlit as st
import tensorflow as tf
import numpy as np
import pretty_midi
import os
import io
import random # For random seed generation
import librosa # Required for pretty_midi.synthesize to work correctly
import soundfile as sf # To write WAV files

# --- Configuration ---
# Path to the directory where your trained model and mappings are saved.
# This assumes 'model_output' is a subfolder in the same directory as this script.
MODEL_OUTPUT_DIR = "model_output/"
# The sequence length your model was trained with (e.g., 50 from your training script)
SEQUENCE_LENGTH = 50 

# --- Streamlit App Title ---
st.set_page_config(page_title="AI Music Generator", layout="centered")
st.title("🎶 AI Music Generator 🎶")
st.markdown("---")

st.write("""
This application uses a trained LSTM neural network to generate new musical sequences.
Provide a starting melody (seed notes), and the AI will predict the subsequent notes.
""")

# --- Load Model and Mappings (Cached for performance) ---
@st.cache_resource
def load_resources(model_dir):
    """Loads the Keras model and note mappings, caching them for efficiency."""
    try:
        model_path = os.path.join(model_dir, "music_generation_model_subset.keras")
        note_to_int_path = os.path.join(model_dir, "note_to_int_subset.npy")
        int_to_note_path = os.path.join(model_dir, "int_to_note_subset.npy")

        # Load the Keras model
        model = tf.keras.models.load_model(model_path)
        
        # Load the numpy dictionaries
        note_to_int = np.load(note_to_int_path, allow_pickle=True).item()
        int_to_note = np.load(int_to_note_path, allow_pickle=True).item()
        
        st.success("Model and mappings loaded successfully!")
        return model, note_to_int, int_to_note
    except Exception as e:
        st.error(f"Error loading model or mappings: {e}")
        st.warning("Please ensure 'model_output' folder exists and contains 'music_generation_model_subset.keras', 'note_to_int_subset.npy', and 'int_to_note_subset.npy'.")
        return None, None, None

model, note_to_int, int_to_note = load_resources(MODEL_OUTPUT_DIR)

if model is None:
    st.stop() # Stop the app if resources couldn't be loaded

# --- Helper Function for Sampling (Temperature-based) ---
def sample_from_probabilities(predictions, temperature=1.0):
    """
    Samples an index from a probability distribution,
    optionally adjusting for 'creativity' with temperature.
    """
    predictions = np.asarray(predictions).astype('float64')
    # Apply temperature: higher temperature -> more random, lower -> more deterministic
    predictions = np.log(predictions + 1e-8) / temperature # Add small epsilon to avoid log(0)
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    
    # Handle potential NaN or inf values after temperature scaling
    if np.isnan(predictions).any() or np.isinf(predictions).any():
        # Fallback to uniform sampling or argmax if probabilities become invalid
        st.warning("Invalid probabilities after temperature scaling. Falling back to argmax.")
        return np.argmax(predictions) # Or random.choice(range(len(predictions)))
    
    # Ensure probabilities sum to 1 to avoid issues with np.random.choice
    predictions = predictions / predictions.sum()

    # Sample the next note index based on probabilities
    next_index = np.random.choice(len(predictions), p=predictions)
    return next_index

# --- Musical Scale Definitions ---
# Define scales as intervals (semitones) from a root note
SCALES = {
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Minor (Natural)": [0, 2, 3, 5, 7, 8, 10],
    "Minor (Harmonic)": [0, 2, 3, 5, 7, 8, 11],
    "Pentatonic Major": [0, 2, 4, 7, 9],
    "Pentatonic Minor": [0, 3, 5, 7, 10],
    "Chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # All notes
}

# Base MIDI note names (without octave, C=0, C#=1, etc.)
# This is useful for building up the scale root selection.
BASE_MIDI_NOTE_NAMES = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
}
BASE_MIDI_NOTE_NUMBERS = {name: num for num, name in BASE_MIDI_NOTE_NAMES.items()} # Reverse mapping

def get_scale_pitches(root_midi_note, scale_type, min_pitch_limit=0, max_pitch_limit=127):
    """
    Generates a set of allowed MIDI pitches for a given root note and scale type.
    """
    scale_intervals = SCALES[scale_type]
    allowed_pitches = set()
    
    # Iterate through multiple octaves to cover the full MIDI range
    for octave_offset in range(-5, 6): # Cover a wide range of octaves relative to C0
        for interval in scale_intervals:
            # The root_midi_note here is the MIDI number for the *base* note (e.g., 0 for C, 1 for C#)
            # We need to calculate its actual pitch in C0 octave for correct offsetting
            base_pitch_in_octave_0 = (root_midi_note % 12) 
            pitch = base_pitch_in_octave_0 + interval + (octave_offset * 12)
            if min_pitch_limit <= pitch <= max_pitch_limit:
                allowed_pitches.add(pitch)
    return allowed_pitches


# --- Music Generation Function ---
def generate_music_sequence(model, note_to_int, int_to_note, seed_notes, num_generate_notes, seq_len, temperature, min_pitch, max_pitch, root_note_midi, scale_type):
   
    generated_notes_indices = []

    # Calculate allowed pitches based on scale and global pitch limits
    # get_scale_pitches now takes the base MIDI note (0-11) as root_midi_note
    allowed_scale_pitches = get_scale_pitches(root_note_midi, scale_type, min_pitch, max_pitch)

    # Convert seed notes to integer indices, enforcing scale and range
    current_seed_indices = []
    for note_pitch in seed_notes:
        if note_pitch in note_to_int and note_pitch in allowed_scale_pitches:
            current_seed_indices.append(note_to_int[note_pitch])
        else:
            st.warning(f"Seed note {note_pitch} is out of specified pitch/scale range. Mapping to a random valid note from the selected scale and range.")
            valid_notes_in_vocab_and_scale = [p for p in int_to_note.values() if p in allowed_scale_pitches]
            if valid_notes_in_vocab_and_scale:
                current_seed_indices.append(note_to_int[random.choice(valid_notes_in_vocab_and_scale)])
            else:
                # Fallback if no notes in vocab match the scale/range (should be rare)
                current_seed_indices.append(random.choice(list(note_to_int.values())))
    
    # Ensure the seed is at least sequence_length long for initial prediction
    if len(current_seed_indices) < seq_len:
        st.warning(f"Seed sequence too short ({len(current_seed_indices)}). Padding with random notes from the selected scale.")
        valid_notes_in_vocab_and_scale = [p for p in int_to_note.values() if p in allowed_scale_pitches]
        while len(current_seed_indices) < seq_len:
            if valid_notes_in_vocab_and_scale:
                current_seed_indices.append(note_to_int[random.choice(valid_notes_in_vocab_and_scale)])
            else:
                current_seed_indices.append(random.choice(list(note_to_int.values())))
    
    input_sequence = list(current_seed_indices[-seq_len:])
    generated_notes_indices.extend(input_sequence)

    for _ in range(num_generate_notes - seq_len):
        input_reshaped = np.array(input_sequence).reshape(1, seq_len)
        predictions = model.predict(input_reshaped, verbose=0)[0] 

        constrained_predictions = np.copy(predictions)
        for i, prob in enumerate(predictions):
            predicted_pitch = int_to_note.get(i) 
            # Zero out probabilities for notes outside the specified pitch range OR not in the selected scale
            if predicted_pitch is None or not (min_pitch <= predicted_pitch <= max_pitch and predicted_pitch in allowed_scale_pitches):
                constrained_predictions[i] = 0.0 
        
        # Re-normalize probabilities after zeroing out values
        if np.sum(constrained_predictions) == 0:
            st.warning("No valid notes left to choose from after pitch/scale constraint. Falling back to unconstrained sampling.")
            next_note_index = sample_from_probabilities(predictions, temperature) # Use original predictions
        else:
            next_note_index = sample_from_probabilities(constrained_predictions, temperature)
        
        generated_notes_indices.append(next_note_index)
        input_sequence.append(next_note_index)
        input_sequence = input_sequence[1:]

    generated_pitches = [int_to_note[idx] for idx in generated_notes_indices]
    return generated_pitches

# --- Function to Create MIDI File and Audio ---
def create_midi_and_audio_file(notes, tempo=120, sample_rate=44100):
    """
    Creates a MIDI file from a list of note pitches and synthesizes a WAV audio.

    Args:
        notes (list): A list of integer pitches.
        tempo (int): The tempo for the MIDI file.
        sample_rate (int): Sample rate for audio synthesis.

    Returns:
        tuple: (midi_bytes, audio_bytes)
               midi_bytes: BytesIO object containing the MIDI data.
               audio_bytes: BytesIO object containing the WAV audio data.
    """
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    current_time = 0.0
    note_duration = 60.0 / tempo / 2 # Example: eighth note duration

    for pitch in notes:
        # Ensure pitch is within MIDI range (0-127) for pretty_midi
        if 0 <= pitch <= 127:
            note = pretty_midi.Note(
                velocity=100,  # MIDI velocity (volume), 1-127
                pitch=pitch,
                start=current_time,
                end=current_time + note_duration
            )
            piano.notes.append(note)
            current_time += note_duration
        else:
            st.warning(f"Skipping invalid pitch: {pitch}")

    midi.instruments.append(piano)

    # --- Create MIDI Bytes ---
    midi_file_bytes = io.BytesIO()
    midi.write(midi_file_bytes)
    midi_file_bytes.seek(0)

    # --- Create Audio (WAV) Bytes ---
    audio_data = midi.synthesize(fs=sample_rate)
    audio_file_bytes = io.BytesIO()
    sf.write(audio_file_bytes, audio_data, sample_rate, format='WAV')
    audio_file_bytes.seek(0)
    
    return midi_file_bytes, audio_file_bytes

# --- Streamlit UI Elements ---

st.header("Generation Settings")

# User input for seed notes
st.markdown("#### 1. Seed Melody")
st.info("Enter 3-5 musical note names (e.g., `C4, D4, E4` for Middle C, D, E). Use `#` for sharps (e.g., `C#5`) and `b` for flats (e.g., `Bb3`).")
seed_note_names_input = st.text_input("Seed Notes (Note Names):", "C4, D4, E4, F4")

# Convert seed note names to MIDI pitch numbers
seed_notes = []
try:
    seed_note_name_list = [s.strip() for s in seed_note_names_input.split(',') if s.strip()]
    for note_name in seed_note_name_list:
        try:
            midi_pitch = pretty_midi.note_name_to_number(note_name)
            seed_notes.append(midi_pitch)
        except ValueError:
            st.error(f"Invalid note name in seed: '{note_name}'. Please check spelling (e.g., C4, D#5).")
            seed_notes = [] # Clear if any invalid notes
            break
    
    if not seed_notes and seed_note_name_list: # If input was given but all invalid
        st.warning("No valid seed notes found. Falling back to default C4 (60).")
        seed_notes = [60]
    elif not seed_notes: # If input was empty
        seed_notes = [60]

except Exception as e:
    st.error(f"An unexpected error occurred parsing seed notes: {e}")
    seed_notes = [60] # Fallback


if len(seed_notes) > SEQUENCE_LENGTH:
    st.warning(f"Seed notes ({len(seed_notes)}) are longer than the model's sequence length ({SEQUENCE_LENGTH}). Using the last {SEQUENCE_LENGTH} notes as seed.")
    seed_notes = seed_notes[-SEQUENCE_LENGTH:]
elif len(seed_notes) < 1:
    st.warning("At least one seed note is required. Using default: C4 (60).")
    seed_notes = [60]


# User input for number of notes to generate
st.markdown("#### 2. Generation Length")
num_generate_notes = st.slider(
    "Total notes to generate (including seed):",
    min_value=SEQUENCE_LENGTH + 1, # Must be at least sequence length + 1
    max_value=500,
    value=100,
    step=10
)

# User input for tempo
st.markdown("#### 3. Tempo (BPM)")
tempo_bpm = st.slider(
    "Tempo (Beats Per Minute):",
    min_value=60,
    max_value=240,
    value=120,
    step=10
)


# User input for pitch range
st.markdown("#### 4. Pitch Range Constraint (in MIDI notes)")
st.info("Limit the generated notes to a specific octave range. For a 2-octave range, set Max Pitch = Min Pitch + 24.")
col1, col2 = st.columns(2)
with col1:
    min_pitch = st.slider(
        "Min MIDI Pitch (0-127):",
        min_value=0,
        max_value=127,
        value=60, # Default to Middle C
        step=1
    )
with col2:
    max_pitch = st.slider(
        "Max MIDI Pitch (0-127):",
        min_value=0,
        max_value=127,
        value=84, # Default to C6 (2 octaves above Middle C)
        step=1
    )

if min_pitch >= max_pitch:
    st.error("Minimum pitch must be less than maximum pitch.")
    min_pitch = 60
    max_pitch = 84

# --- Musical Scale Controls ---
st.markdown("#### 5. Musical Scale Constraint")
st.info("Further guide generation to notes within a chosen root note and scale type.")
col_scale1, col_scale2 = st.columns(2)
with col_scale1:
    root_note_name = st.selectbox(
        "Root Note for Scale:",
        list(BASE_MIDI_NOTE_NAMES.values()),
        index=0 # Default to C
    )
    root_note_midi = BASE_MIDI_NOTE_NUMBERS[root_note_name]

with col_scale2:
    scale_type = st.selectbox(
        "Scale Type:",
        list(SCALES.keys()),
        index=0 # Default to Major
    )

# User input for temperature (creativity)
st.markdown("#### 6. Creativity (Temperature)")
st.info("Lower temperature (e.g., 0.2) makes the output more predictable and similar to training data. Higher temperature (e.g., 1.0+) makes it more random and 'creative'.")
temperature = st.slider(
    "Temperature:",
    min_value=0.1,
    max_value=2.0,
    value=0.8,
    step=0.1
)

st.markdown("---")

# Generate Button
if st.button("Generate Music"):
    if model and note_to_int and int_to_note:
        with st.spinner("Generating music... This might take a moment."):
            try:
                generated_pitches = generate_music_sequence(
                    model,
                    note_to_int,
                    int_to_note,
                    seed_notes,
                    num_generate_notes,
                    SEQUENCE_LENGTH,
                    temperature,
                    min_pitch, 
                    max_pitch,
                    root_note_midi, 
                    scale_type     
                )
                
                if generated_pitches:
                    st.success(f"Generated {len(generated_pitches)} notes!")
                    
                    # Create MIDI and Audio files
                    midi_bytes, audio_bytes = create_midi_and_audio_file(generated_pitches, tempo=tempo_bpm)
                    
                    st.subheader("Listen to your generated music:")
                    st.audio(audio_bytes, format='audio/wav')
                    st.info("Note: Direct playback uses basic sine wave sounds. For richer instrument sounds, download the MIDI file.")

                    st.download_button(
                        label="Download MIDI File",
                        data=midi_bytes,
                        file_name="generated_music.mid",
                        mime="audio/midi"
                    )
                    
                    st.markdown("---")
                    st.subheader("Generated Notes (Pitches):")
                    st.write(generated_pitches) 
                else:
                    st.error("Music generation failed. No notes were generated.")
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
                st.info("Please check your input parameters and try again. Ensure `librosa` and `soundfile` are installed (`pip install librosa soundfile`).")
    else:
        st.error("Model or mappings not loaded. Cannot generate music.")

st.markdown("---")
st.markdown("Built with ❤️ and Streamlit")
