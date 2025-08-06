# Kimi-K2-from-scratch-NanoKimi-K2

## Overview
This repository implements **NanoKimi K2**, a transformer-based language model inspired by Kimi K2, built from scratch. NanoKimi K2 features a **Mixture of Experts (MoE)** architecture with 4 experts, a novel **Latent Attention** mechanism, and a custom **Muon optimizer**, designed to enhance performance in language modeling tasks. The model is trained on 10% of the TinyStories dataset using a SentencePiece tokenizer (BPE, vocab size 4096) for character-level tokenization. The implementation includes training, evaluation, and text generation with top-k sampling, along with an interactive prompt box for generating text in a Jupyter environment.

The code, contained in `NanoKimi.ipynb`, trains NanoKimi K2, computes training and validation losses, and supports interactive text generation. The Muon optimizer stabilizes training for the MoE and Latent Attention components, achieving efficient convergence with lower memory usage, though with increased computational cost compared to standard optimizers like AdamW.

## Setup Instructions
### Prerequisites
- Python 3.8+
- PyTorch (`torch`)
- NumPy (`numpy`)
- SentencePiece (`sentencepiece`)
- Datasets (`datasets`)
- IPython (`ipython`)
- IPyWidgets (`ipywidgets`)
- Jupyter environment (e.g., Jupyter Notebook or Google Colab)
- CUDA-enabled GPU (optional, falls back to CPU if unavailable)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/Kimi-K2-from-scratch-NanoKimi-K2-.git
   cd Kimi-K2-from-scratch-NanoKimi-K2-
   ```
2. Install dependencies:
   ```bash
   pip install torch numpy sentencepiece datasets ipython ipywidgets jupyter
   ```
3. Ensure the TinyStories dataset is accessible via the `datasets` library (automatically downloaded by the code).

### Running the Code
Open `NanoKimi.ipynb` in a Jupyter environment (e.g., Jupyter Notebook or Google Colab) and run all cells to:
- Train the NanoKimi K2 model on 10% of the TinyStories dataset.
- Compute training and validation losses over 5000 steps.
- Enable interactive text generation via a prompt box.
Alternatively, convert the notebook to a Python script and run it:
```bash
jupyter nbconvert --to script NanoKimi.ipynb
python NanoKimi.py
```
**Note**: The interactive prompt box requires a Jupyter environment and will not work in a standard Python script.

The script outputs training progress (train and validation loss every 100 steps) and provides an interactive interface for text generation.

## Code Structure
The code in `NanoKimi.ipynb` is organized into the following components:

### Data Loading and Preprocessing
- Loads 10% of the TinyStories dataset using the `datasets` library.
- Saves the dataset to `corpus.txt` and trains a SentencePiece tokenizer (BPE, vocab size 4096).
- Tokenizes the dataset into a tensor of token IDs.
- Splits data into 90% training and 10% validation sets (`train_data = data[:n]`, `val_data = data[n:]`, where `n = int(0.9 * len(data))`).
- Implements `get_batch` to generate input-target pairs (block size 64, batch size 32).

### Model Architecture
- **NanoKimi K2**: A transformer with 6 layers, each containing a Latent Attention module and an MoE layer (4 experts), with embedding dimension 256.
- **Components**:
  - `SwiGLU`: Activation function splitting input into two parts, applying SiLU to one and multiplying.
  - `Expert`: A feedforward network with SwiGLU activation for MoE.
  - `MoELayer`: Mixture of 4 experts with a gating network for dynamic weighting.
  - `LatentAttention`: Novel attention mechanism using learnable latent vectors for efficient attention computation.
  - `NanoKimiTransformer`: Full model with token/position embeddings, transformer blocks, and a final linear head.

### Muon Optimizer
- Custom `Muon` optimizer inspired by Lion, using momentum (`exp_avg`) and difference tracking (`exp_avg_diff`) for updates.
- **Implementation**:
  ```python
  class Muon(torch.optim.Optimizer):
      def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-2):
          defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
          super().__init__(params, defaults)

      @torch.no_grad()
      def step(self, closure=None):
          for group in self.param_groups:
              lr = group["lr"]
              beta1, beta2 = group["betas"]
              weight_decay = group["weight_decay"]

              for p in group["params"]:
                  if p.grad is None:
                      continue
                  g = p.grad

                  state = self.state[p]
                  if len(state) == 0:
                      state["exp_avg"] = torch.zeros_like(p)
                      state["exp_avg_diff"] = torch.zeros_like(p)

                  exp_avg, exp_avg_diff = state["exp_avg"], state["exp_avg_diff"]

                  update = p - exp_avg_diff
                  exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                  exp_avg_diff.mul_(beta2).add_(update, alpha=1 - beta2)

                  if weight_decay > 0:
                      p.data.mul_(1 - lr * weight_decay)

                  p.add_(exp_avg_diff, alpha=-lr)
  ```
- **Advantages**:
  - Stabilizes training for MoE and Latent Attention layers.
  - Efficient for sparse updates, reducing memory footprint.
  - Trade-off: Higher computational cost due to dual momentum tracking.

### Training and Evaluation
- Trains for 5000 steps, computing cross-entropy loss on training batches.
- Evaluates validation loss every 100 steps using `compute_val_loss`.
- Uses batch size 32 and block size 64.

### Text Generation
- Implements top-k sampling (k=50, temperature=0.8) for text generation.
- Provides an interactive prompt box in Jupyter for user-input prompts.
- **Example Usage**:
  ```python
  def generate_text(prompt, steps=50, temperature=0.8, top_k=50):
      model.eval()
      tokens = torch.tensor([sp.encode(prompt)], dtype=torch.long).to(device)
      for _ in range(steps):
          logits = model(tokens)
          logits = logits[:, -1, :] / temperature
          top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
          probs = F.softmax(top_k_logits, dim=-1)
          next_token = torch.multinomial(probs, 1)
          next_token = top_k_indices.gather(-1, next_token)
          tokens = torch.cat([tokens, next_token], dim=1)
      return sp.decode(tokens[0].tolist())
  ```

## NanoKimi K2: Design and Advantages
NanoKimi K2 is designed for efficient and high-performing language modeling with the following features:

### Mixture of Experts (MoE)
- **Structure**: Each transformer block includes an `MoELayer` with 4 experts, dynamically weighted by a gating network.
- **Benefit**: Enhances modeling capacity by specializing computations, improving performance on complex datasets like TinyStories.
- **Implementation**:
  ```python
  class MoELayer(nn.Module):
      def __init__(self, dim, num_experts=4):
          super().__init__()
          self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
          self.gate = nn.Linear(dim, num_experts)

      def forward(self, x):
          B, T, D = x.shape
          weights = F.softmax(self.gate(x), dim=-1)
          expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
          weights = weights.permute(2, 0, 1).unsqueeze(-1)
          out = (weights * expert_outputs).sum(dim=0)
          return out
  ```

### Latent Attention
- **Structure**: Uses learnable latent vectors (`num_latents=16`, `latent_dim=64`) to compute attention, reducing computational complexity.
- **Benefit**: Efficiently captures long-range dependencies, complementing MoE’s capacity.
- **Implementation**:
  ```python
  class LatentAttention(nn.Module):
      def __init__(self, dim, latent_dim=64, num_latents=16):
          super().__init__()
          self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))
          self.to_q = nn.Linear(latent_dim, latent_dim)
          self.to_kv = nn.Linear(dim, 2 * latent_dim)
          self.to_out = nn.Linear(latent_dim, dim)

      def forward(self, x):
          B, T, _ = x.shape
          latents = self.latents.expand(B, -1, -1)
          q = self.to_q(latents)
          k, v = self.to_kv(x).chunk(2, dim=-1)
          attn = (q @ k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
          attn = F.softmax(attn, dim=-1)
          out = attn @ v
          out = self.to_out(out).mean(dim=1, keepdim=True)
          return x + out.expand(-1, T, -1)
  ```

### Muon Optimizer
- **Purpose**: Optimizes MoE and Latent Attention layers with momentum-based updates inspired by Lion.
- **Advantages**:
  - Stabilizes training for high-dimensional, sparse architectures.
  - Reduces memory usage compared to AdamW.
  - Trade-off: Slower updates due to dual momentum tracking.
- **Key Parameters**:
  - Learning Rate: 1e-3
  - Betas: (0.9, 0.99)
  - Weight Decay: 1e-2

### Key Model Parameters
- Embedding Dimension: 256 (`dim`)
- Layers: 6 (`depth`)
- Experts: 4 (`num_experts`)
- Latent Vectors: 16 (`num_latents`)
- Latent Dimension: 64 (`latent_dim`)
- Vocabulary Size: 4096 (SentencePiece BPE)
- Block Size: 64
- Batch Size: 32

  ### Output
  <img width="1736" height="293" alt="image" src="https://github.com/user-attachments/assets/75e0f214-d64b-4018-860d-76e9684b0da4" />
  Prompt: Once upon a time
  
  Generated: Once upon a time law digent fridge cupboard mon accidentally toldicy drive Snicesccer new enthusiasticikes knockredish mighty countself good piano scooter mov    engine little cap butterflies plan sandwuring raining valuable mor deadilli shone unboard cup So attract fighting mixed bossy meked engine

  Prompt: Once upon a time
  Generated: Once upon a time mommyleephat smile falling F Bl bite cont adventures bushes avoc grab wrote spell splash Joh trying pilind value jœYou already hotel accept      both wild borrowlc being picked pianob After listenedastickeep sorry neighbor Dave mulecast organizether grab facesatoio pant

  Prompt: The owl
  Generated: The owl carrot games worked wet drum deliver trashize solve inside mysterious anywhereittenioamp howob bul lots mommystairsmaz crab pony hook these loop tri ce   squ fier tun ing Mrs wat un Mommy offered tenven shelfophoneboard walksward fire adm Daveachedelly

## Usage
1. **Prepare Environment**: Ensure dependencies are installed and the repository is cloned.
2. **Run Notebook**: Open `NanoKimi.ipynb` in Jupyter Notebook or Google Colab, and execute all cells to:
   - Train the model for 5000 steps.
   - Monitor training and validation loss.
   - Use the interactive prompt box to generate text (e.g., enter "Once upon a time" and click "Generate").
3. **Generate Text**: Input a prompt in the text box, click the "Generate" button, and view the generated text (50 steps, top-k=50, temperature=0.8).
4. **Analyze Output**: Check console output for loss metrics and generated text.

## Example Output
**Prompt**: "Once upon a time"
**Generated**: (Example output depends on training and dataset, e.g., a continuation of a story-like sequence based on TinyStories patterns.)

## Limitations and Future Work
- **Dataset Size**: Uses only 10% of TinyStories, limiting generalization. Future work could scale to the full dataset.
- **Optimization**: Implement top-k gating in `MoELayer` or reduce `num_experts` for faster training.
- **Resource Limitation**: Due to computational constraints (e.g., limited GPU memory and processing power), NanoKimi K2’s architecture (e.g., 6 layers, 4 experts) and training (5000 steps) are scaled down, potentially limiting performance compared to larger models.
- **Interactive Interface**: The prompt box is Jupyter-specific. A web-based interface could improve accessibility.

## Contributing
Contributions are welcome! Please submit issues or pull requests for bug fixes, optimizations, or additional features.
