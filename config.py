# ==================== Hyperparameters ====================

# --- Tokenizer ---
TOKENIZER_VOCAB_SIZE = 10000
TOKENIZER_DIR = "./my_tokenizer"

# --- Model ---
D_MODEL   = 256
N_LAYERS  = 4
N_HEADS   = 4
D_FF      = 1024
MAX_SEQ_LEN = 128  # Sequence length after tokenization (data is truncated/padded to this)

# --- Data Preprocessing ---
TOKENIZE_SUBSET    = 51000
TRAIN_SUBSET    = 11000

# --- Training ---
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-4
EPOCHS          = 2000
EVAL_EVERY      = 50
SAVE_EVERY      = 500 #epochs
PROMPT_LEN      = 32    # Must be < MAX_SEQ_LEN (prompt is kept unmasked, rest is generated)
NUMBER_WORKER   = 0 

# --- Validation ---
VAL_SPLIT       = 0.1   # Fraction of training subset held out for validation
VAL_EVERY       = 50    # Validate every N epochs
VAL_PASSES      = 3     # Number of eval passes to average over (reduces masking variance)

# --- Diffusion ---
DIFFUSION_STEPS = 128

# --- Paths ---
DATA_DIR        = "./data"
CHECKPOINT_DIR  = "./checkpoints"