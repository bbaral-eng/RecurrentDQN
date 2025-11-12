# Recurrent DQN (R2D1)

This repository contains an implementation of a Recurrent DQN agent based on the R2D1 (Recurrent Replay Distributed DQN) architecture. This is a streamlined variant that combines key improvements to vanilla DQN without the full Rainbow suite:

**Core Components:**
- **Double DQN**: Reduces overestimation bias by using the online network to select actions and the target network to evaluate them
- **Prioritized Experience Replay (PER)**: Samples important transitions more frequently based on TD-error
- **Dueling Architecture**: Separates state value and action advantage streams for better learning
- **N-step Learning**: Uses multi-step returns to reduce variance and improve convergence
- **Custom Feature Extractor**: Handles dictionary observations from the environment

The code is structured to be run in a Jupyter notebook environment with specific dependencies and Python version requirements.

## Prerequisites

- Python 3.12
- Git (for cloning the repository)

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/bbaral-eng/Recurrent_Rainbow_DQN.git
   cd RecurrentDQN
   ```

2. Create a virtual environment:
   ```bash
   python3.12 -m venv venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

1. Ensure your virtual environment is activated (you should see `(venv)` in your terminal prompt)

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the `RecurrentDQN.ipynb` notebook

4. Run all cells in the notebook sequentially from top to bottom

**Alternative: Run via CLI (for hyperparameter tuning)**

For hyperparameter sweeps, use the CLI script with specific hyperparameters:

```bash
# Activate your venv
source venv/bin/activate

# Run a single experiment
python train_coreweave.py \
  --lr 2.5e-4 \
  --gamma 0.99 \
  --n_step 1 \
  --alpha 0.1 \
  --beta 0.6 \
  --prior_eps 1e-6 \
  --seed 42 \
  --total_timesteps 40000
```

For Kubernetes/CoreWeave distributed runs:

```bash
# Generate job manifests
python launch_coreweave_experiments.py --generate-only

# Apply to cluster (requires kubeconfig)
kubectl apply -f k8s_jobs/
```

## Important Notes

- Make sure to use Python 3.12 specifically, as the code has been tested and verified with this version
- The notebook must be run in sequential order as later cells depend on the execution of previous cells
- Do not skip any cells as this may cause dependency or initialization issues
- If you encounter any memory issues, you may need to restart the kernel and run all cells again
- For distributed training on CoreWeave, ensure Docker and kubectl are configured properly

## Hyperparameter Tuning

The repository supports both local and distributed hyperparameter optimization:

**Local tuning (small grid):**
```bash
source venv/bin/activate
python train_coreweave.py --lr 1e-4 --gamma 0.99 --n_step 1 --alpha 0.0 --beta 0.6 --prior_eps 1e-6 --seed 42 --total_timesteps 40000
```

**Distributed tuning (CoreWeave):**
1. Update `launch_coreweave_experiments.py` with your Docker image and namespace
2. Generate job YAMLs: `python launch_coreweave_experiments.py --generate-only`
3. Deploy: `kubectl apply -f k8s_jobs/`
4. Monitor results via W&B dashboard

## Troubleshooting

If you encounter any issues:
1. Verify that you're using Python 3.12
2. Confirm all dependencies are installed correctly
3. Try restarting the Jupyter kernel
4. Ensure all cells are run in order from top to bottom

## Project Structure

- `RecurrentDQN.ipynb`: Main notebook containing the R2D1 implementation
- `train_coreweave.py`: CLI script for single or batch hyperparameter experiments
- `launch_coreweave_experiments.py`: Kubernetes job launcher for distributed training
- `Dockerfile`: Container specification for CoreWeave deployment
- `CustomFeatureExtractor.py`: Environment-specific feature extraction
- `Env.py`: Custom training environment
- `segment_tree.py`: Efficient segment tree for prioritized replay
- `requirements.txt`: Python dependencies
- `k8s_jobs/`: Generated Kubernetes job manifests (created by launch script)

## References

This implementation is based on:
- Deep Recurrent Q-Learning for Partially Observable MDPs (Hausknecht & Stone, 2017)
- Recurrent Experience Replay in Distributed Reinforcement Learning (Kapturowski et al., 2019)
- Implementation inspiration: https://github.com/Curt-Park/rainbow-is-all-you-need

