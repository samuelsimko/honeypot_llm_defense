==================================================
HONEYPOT LLM DEFENSE
==================================================

A framework for studying, generating, and defending against *honeypot attacks* and *alignment-breaking behaviors* in large language models (LLMs).  
The repository provides utilities for **data generation**, **soft prompt optimization**, **alignment fine-tuning**, and **automated harmfulness evaluation** using judge models.

--------------------------------------------------
📁 Repository Structure
--------------------------------------------------

::

  honeypot_llm_defense/
  ├── attacks/
  │   ├── behavior_targets/
  │   │   └── augment.py         # Roleplay & stylistic augmentations for harmful task prompts
  │   └── embedding/
  │       └── embedding_attack.py  # Main soft prompt optimization & attack script
  ├── data/
  │   ├── harmbench_behaviors_text_*.csv  # HarmBench behavior datasets
  │   └── harmbench_targets_*.json        # Target outputs for optimization
  ├── defenses/
  │   ├── honeypot_hinge.py               # Honeypot hinge loss training
  │   ├── honeypot_simple.py              # Simple honeypot alignment defense
  │   └── train_*.py                      # Finetuning utilities
  ├── defended_model_sft*/                # Saved supervised fine-tuned models (LoRA adapters)
  ├── generation/                         # Data generation & category utilities
  ├── judges/
  │   ├── strong_reject/                  # StrongReject classifier for model refusals
  │   ├── harmbench_judge.py              # HarmBench Llama-2 13B classifier interface
  │   └── embedding_attack.py             # Alternate evaluation integration
  ├── scripts/
  │   ├── finetune_oss.sh                 # Finetune script for GPT-OSS / LLaMA models
  │   ├── run_embedding_attack.sh         # Runs soft prompt optimization benchmark
  │   ├── run_honeypot_hinge.sh           # Trains the honeypot hinge defense
  │   ├── run_honeypot_simple.sh          # Trains the simple honeypot model
  │   ├── test_model.py                   # Quick model interaction test script
  │   └── train_new.sh                    # New defense training helper
  ├── results/                            # Attack & evaluation logs
  ├── eval.py                             # Model evaluation helper script
  ├── defense_finetune.py                 # Core alignment fine-tuning pipeline
  ├── tamper.py                           # Analysis / tampering utilities
  ├── outputs.jsonl / outputs.csv         # Cached experiment outputs
  └── requirements.txt                    # Python environment dependencies


--------------------------------------------------
⚙️ Environment Setup
--------------------------------------------------

**Requirements**
- Python 3.10
- CUDA 12.x compatible GPU

Install dependencies:

.. code-block:: bash

   git clone --recursivee_submodules https://github.com/samuelsimko/honeypot_llm_defense.git
   cd honeypot_llm_defense
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

--------------------------------------------------
🧩 Key Components
--------------------------------------------------

**1️⃣ Honeypot Defenses**

- ``defenses/honeypot_hinge.py``  
  Trains a model to recognize and neutralize honeypot triggers through a hinge-style loss function.

- ``defenses/honeypot_simple.py``  
  Lightweight defense model for baseline comparisons.

Run training:

.. code-block:: bash

   source scripts/run_honeypot_hinge.sh
   # or
   source scripts/run_honeypot_simple.sh


**2️⃣ Soft Prompt Optimization (Attacks)**

Implements gradient-based embedding attacks to find adversarial prefix embeddings that make a model produce a target harmful output.

Main script:
``attacks/embedding/embedding_attack.py``

Run via:

.. code-block:: bash

   source scripts/run_embedding_attack.sh

The attack will:
 - Load a base model (e.g., Meta-Llama-3-8B, GPT-OSS-20B)
 - Optimize a soft prompt embedding for each harmful task
 - Evaluate outputs with:
    - **HarmBench judge** (`cais/HarmBench-Llama-2-13b-cls`)
    - **StrongReject** refusal classifier
 - Save variant-by-variant results to `results/`


**3️⃣ Data Augmentation**

Augment prompt–target pairs for robustness testing using:
``attacks/behavior_targets/augment.py``

The augmentation layer adds:
 - Roleplay or persona modifiers
 - Random capitalization / leetspeak noise
 - Side questions or headers
 - Fake “godmode” jailbreak patterns

Used automatically by `embedding_attack.py` during each optimization run.

**4️⃣ Evaluation Utilities**

Run quick response tests:

.. code-block:: bash

   python scripts/test_model.py --model-path defended_model_sft
