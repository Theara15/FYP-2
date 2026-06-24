import streamlit as st
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
import numpy as np

# Page configuration
st.set_page_config(
    page_title="RoBERTa Model Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 RoBERTa Model Demonstration")
st.markdown("---")

# ============ SIDEBAR CONFIGURATION ============
st.sidebar.header("⚙️ Model Configuration")

# Device selection
device = st.sidebar.selectbox(
    "Select Computing Device:",
    ["CPU", "CUDA (GPU)"]
)
device = "cuda" if device == "CUDA (GPU)" and torch.cuda.is_available() else "cpu"

# Model path
model_path = st.sidebar.text_input(
    "Model File Path:",
    value="roberta_best.pt",
    help="Path to your saved RoBERTa model file"
)

# ============ MODEL LOADING ============
@st.cache_resource
def load_model(model_path, device):
    """Load the RoBERTa model with caching"""
    try:
        # Load tokenizer (standard RoBERTa tokenizer)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Load pre-trained RoBERTa model
        model = RobertaModel.from_pretrained('roberta-base')
        
        # Load your trained weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # If checkpoint is a state_dict, load it directly
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            # If it's the entire model, use it directly
            model = checkpoint
        
        model = model.to(device)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model
st.sidebar.write(f"📍 Device: **{device.upper()}**")
with st.spinner("Loading model..."):
    model, tokenizer = load_model(model_path, device)

if model is None or tokenizer is None:
    st.error("❌ Failed to load model. Please check the file path.")
    st.stop()

st.sidebar.success("✅ Model loaded successfully!")

# ============ MAIN INTERFACE ============
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Processing", "Model Info"])

# ============ TAB 1: SINGLE PREDICTION ============
with tab1:
    st.header("Single Text Analysis")
    
    input_text = st.text_area(
        "Enter text to analyze:",
        placeholder="Type or paste your text here...",
        height=120
    )
    
    if st.button("Analyze Text", key="single_analyze"):
        if input_text.strip():
            with st.spinner("Processing..."):
                # Tokenize
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get model output
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state
                    pooled_output = outputs.pooler_output
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Results")
                    st.write(f"**Input Length:** {len(input_text)} characters")
                    st.write(f"**Tokens:** {inputs['input_ids'].shape[1]}")
                    st.write(f"**Embedding Dimension:** {embeddings.shape[-1]}")
                
                with col2:
                    st.subheader("🔢 Output Shape")
                    st.write(f"**Sequence Embeddings:** {tuple(embeddings.shape)}")
                    st.write(f"**Pooled Output:** {tuple(pooled_output.shape)}")
                
                # Show token-level analysis
                st.subheader("🏷️ Token-Level Analysis")
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                token_embeddings = embeddings[0].detach().cpu().numpy()
                
                # Calculate token similarity (simple cosine similarity to pooled output)
                pooled_np = pooled_output[0].detach().cpu().numpy()
                similarities = [
                    float(np.dot(token_emb, pooled_np) / (np.linalg.norm(token_emb) * np.linalg.norm(pooled_np) + 1e-8))
                    for token_emb in token_embeddings
                ]
                
                token_df = {
                    "Token": tokens,
                    "Similarity to Pooled": [f"{sim:.4f}" for sim in similarities]
                }
                st.dataframe(token_df, use_container_width=True)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

# ============ TAB 2: BATCH PROCESSING ============
with tab2:
    st.header("Batch Text Processing")
    
    batch_mode = st.radio(
        "Input Method:",
        ["Text Area (one per line)", "Upload CSV File"]
    )
    
    texts = []
    
    if batch_mode == "Text Area (one per line)":
        batch_text = st.text_area(
            "Enter multiple texts (one per line):",
            placeholder="Line 1\nLine 2\nLine 3\n...",
            height=150
        )
        texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
    
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            st.write("Preview:")
            st.dataframe(df.head())
            
            text_column = st.selectbox(
                "Select text column:",
                df.columns
            )
            texts = df[text_column].tolist()
    
    if st.button("Process Batch", key="batch_analyze"):
        if texts:
            with st.spinner(f"Processing {len(texts)} texts..."):
                results = []
                progress_bar = st.progress(0)
                
                for i, text in enumerate(texts):
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        pooled = outputs.pooler_output[0].cpu().numpy()
                    
                    results.append({
                        "Text": text[:50] + "..." if len(text) > 50 else text,
                        "Length": len(text),
                        "Tokens": inputs['input_ids'].shape[1],
                        "Embedding_Vector": pooled
                    })
                    
                    progress_bar.progress((i + 1) / len(texts))
                
                # Display results
                st.subheader("Batch Results")
                results_df = {
                    "Text": [r["Text"] for r in results],
                    "Length": [r["Length"] for r in results],
                    "Tokens": [r["Tokens"] for r in results]
                }
                st.dataframe(results_df, use_container_width=True)
                
                st.success(f"✅ Processed {len(texts)} texts successfully!")
                
                # Option to download embeddings
                embeddings_data = np.array([r["Embedding_Vector"] for r in results])
                st.write(f"Embeddings shape: {embeddings_data.shape}")
                
                if st.button("Download Embeddings as CSV"):
                    import pandas as pd
                    embeddings_df = pd.DataFrame(embeddings_data)
                    csv = embeddings_df.to_csv(index=False)
                    st.download_button(
                        label="Download embeddings.csv",
                        data=csv,
                        file_name="embeddings.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("⚠️ Please provide texts to process.")

# ============ TAB 3: MODEL INFO ============
with tab3:
    st.header("Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "RoBERTa")
        st.metric("Tokenizer", "RoBERTa Base")
    
    with col2:
        st.metric("Device", device.upper())
        total_params = sum(p.numel() for p in model.parameters())
        st.metric("Parameters", f"{total_params:,}")
    
    with col3:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        st.metric("Trainable", f"{trainable_params:,}")
    
    st.divider()
    
    st.subheader("Model Architecture")
    with st.expander("View Full Model Structure"):
        st.write(str(model))
    
    st.subheader("Configuration")
    if hasattr(model, 'config'):
        config_dict = model.config.to_dict()
        st.json(config_dict)
    else:
        st.info("Model configuration not available")
    
    st.subheader("Tokenizer Information")
    st.write(f"Vocabulary Size: {len(tokenizer)}")
    st.write(f"Model Max Length: {tokenizer.model_max_length}")

st.markdown("---")
st.markdown("""
### 📝 Instructions:
1. **Single Prediction**: Analyze individual texts and view token-level embeddings
2. **Batch Processing**: Process multiple texts at once and download embeddings
3. **Model Info**: View detailed model architecture and configuration

### 🔧 Tips:
- Your model runs on **CPU** or **GPU** (CUDA) depending on availability
- Texts are automatically truncated to 512 tokens
- Embeddings are 768-dimensional vectors (standard for RoBERTa-base)
""")