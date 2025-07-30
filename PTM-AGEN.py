# -*- coding: utf-8 -*-
"""
A Multi-Objective, Biologically-Informed Framework for Protein PTM Site Data Augmentation
using Generative Language Models and Evolutionary Computing.

This script implements an advanced data augmentation pipeline for small PTM datasets.
Key features:
1.  ESM-3 Inpainting for context-aware positive sample generation.
2.  NSGA-II multi-objective optimization for balancing competing biological objectives.
3.  A sophisticated, multi-modal fitness function including:
    - PTM motif scoring.
    - Structural viability (RSA, Disorder) via a local NetSurfP-3.0 wrapper.
    - Sequence plausibility via ESM pseudo-perplexity.
4.  BLOSUM62-informed mutation operator for biologically relevant exploration.
5.  Hard negative mining to create a challenging training set for downstream models.
"""

import torch
import pandas as pd
import numpy as np
import random
import re
import os
import subprocess
import time
from tqdm import tqdm
from deap import base, creator, tools, algorithms
from transformers import AutoTokenizer, EsmForConditionalGeneration, GenerationConfig
from Bio.Align import substitution_matrices
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors

'''Post-Translational Modification Augmentation via Generative Evolutionary Network(PTM-AGEN)'''
# =======================================
# 1. Configuration Class
# =======================================
class Config:
    """Stores all hyperparameters and configuration settings."""
    # --- Data Parameters ---
    INPUT_CSV = "protein_data.csv"
    OUTPUT_CSV = "enhanced_protein_data.csv"
    SEQ_LENGTH = 33
    CENTER_POS = 16  # 0-indexed

    # --- PTM Parameters ---
    PTM_TYPE = "phosphorylation"
    PTM_AA = {"S", "T", "Y"}
    # Example motif for phosphorylation:..P..
    # This can be made more sophisticated, e.g., using regex or PSSMs
    MOTIF_REGEX = re.compile(r".{14}.{2}P.{2}.{10}")

    # --- Generation & Augmentation Parameters ---
    GENERATION_FACTOR = 5  # Total dataset size multiplier
    INPAINTING_MASK_RATIO = 0.4  # Percentage of window to mask for inpainting
    INPAINTING_WINDOW_SIZE = 7  # +/- residues around the center

    # --- ESM Model Parameters ---
    ESM_MODEL_NAME = "facebook/esm3_650M"
    QUANT_4BIT = True  # Use 4-bit quantization for memory saving
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- NSGA-II Parameters ---
    POPULATION_SIZE = 100
    GENERATIONS = 50
    MUTATION_PROB = 0.2
    CROSSOVER_PROB = 0.9
    # Fitness weights: (Motif, Structure, Plausibility). Signs denote min/max.
    # 1.0 for maximization, -1.0 for minimization.
    FITNESS_WEIGHTS = (1.0, 1.0, -1.0)  # Maximize Motif, Maximize Structure, Minimize Perplexity

    # --- External Tool Parameters ---
    # Assumes NetSurfP-3.0 is installed locally. Update this path.
    NETSURFP_PATH = "./netsurfp-3.0/run_netsurfp.py"
    NETSURFP_MODEL_PATH = "./netsurfp-3.0/models/NSP3_1_CNN_1_LSTM_1_ESM1b_1.pt"  # Example path

    # --- Hard Negative Mining Parameters ---
    HARD_NEGATIVE_K = 5  # Number of hard negatives to find for each positive
    NEGATIVE_POOL_RATIO = 0.5  # Ratio of hard negatives in the final negative set


# =======================================
# 2. ESM-3 Manager Class
# =======================================
class ESM3Manager:
    """Handles all interactions with the ESM-3 model."""

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.ESM_MODEL_NAME)
        self.model = self._load_model()
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    def _load_model(self):
        """Loads the ESM-3 model with optional quantization."""
        model = EsmForConditionalGeneration.from_pretrained(
            self.config.ESM_MODEL_NAME,
            device_map="auto",
            load_in_4bit=self.config.QUANT_4BIT,
            torch_dtype=torch.float16 if self.config.QUANT_4BIT else torch.float32
        )
        return model

    def inpaint_sequence(self, seed_seq: str) -> str:
        """Generates a new sequence by inpainting a masked version of the seed."""
        seq_list = list(seed_seq)
        window_start = max(0, self.config.CENTER_POS - self.config.INPAINTING_WINDOW_SIZE)
        window_end = min(self.config.SEQ_LENGTH, self.config.CENTER_POS + self.config.INPAINTING_WINDOW_SIZE + 1)

        indices_in_window =
        num_to_mask = int(len(indices_in_window) * self.config.INPAINTING_MASK_RATIO)
        indices_to_mask = random.sample(indices_in_window, k=num_to_mask)

        for i in indices_to_mask:
            seq_list[i] = self.tokenizer.mask_token

        masked_prompt = "".join(seq_list)

        inputs = self.tokenizer(masked_prompt, return_tensors="pt").to(self.config.DEVICE)

        # Use a simple generation config for inpainting
        gen_config = GenerationConfig(
            do_sample=True,
            temperature=1.0,
            top_k=50,
            max_length=self.config.SEQ_LENGTH + 2  # Add space for special tokens
        )

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=gen_config)

        decoded_seq = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        # Clean up the output
        cleaned_seq = decoded_seq.replace(" ", "").replace(self.tokenizer.pad_token, "")
        return cleaned_seq

    def get_embedding(self, sequences: list[str]) -> np.ndarray:
        """Gets sequence-level embeddings (CLS token of the last hidden layer)."""
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.config.SEQ_LENGTH
        ).to(self.config.DEVICE)

        with torch.no_grad():
            outputs = self.model.base_model.encoder(**inputs)
            # Embedding is the first token () of the last layer
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings

    def calculate_pseudo_perplexity(self, sequence: str) -> float:
        """Calculates the pseudo-perplexity of a sequence."""
        # This is a simplified implementation. A more rigorous one would average over all masked positions.
        # Here we mask one by one and average the log-likelihoods.
        total_neg_log_likelihood = 0.0

        for i in range(len(sequence)):
            original_char = sequence[i]
            if original_char not in self.amino_acids: continue

            temp_seq = list(sequence)
            temp_seq[i] = self.tokenizer.mask_token
            masked_input = "".join(temp_seq)

            inputs = self.tokenizer(masked_input, return_tensors="pt").to(self.config.DEVICE)
            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Get the log probability of the original token
            original_token_id = self.tokenizer.convert_tokens_to_ids(original_char)
            log_probs = torch.log_softmax(logits[0, i + 1, :], dim=-1)  # +1 for BOS token
            token_log_prob = log_probs[original_token_id].item()
            total_neg_log_likelihood -= token_log_prob

        avg_neg_log_likelihood = total_neg_log_likelihood / len(sequence)
        perplexity = np.exp(avg_neg_log_likelihood)
        return perplexity


# =======================================
# 3. Structure Predictor Class
# =======================================
class StructurePredictor:
    """Wrapper for the local NetSurfP-3.0 tool."""

    def __init__(self, config: Config):
        self.config = config
        if not os.path.exists(config.NETSURFP_PATH):
            raise FileNotFoundError(f"NetSurfP-3.0 script not found at: {config.NETSURFP_PATH}")

    def predict(self, sequence: str) -> dict | None:
        """Runs NetSurfP-3.0 on a single sequence and parses the output."""
        # Create a temporary FASTA file
        fasta_content = f">temp_seq\n{sequence}\n"
        temp_fasta_path = "temp_seq.fasta"
        temp_output_dir = "temp_netsurfp_out"

        with open(temp_fasta_path, "w") as f:
            f.write(fasta_content)

        # Construct and run the command
        command =

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)

            # Parse the CSV output
            output_csv = os.path.join(temp_output_dir, "temp_seq.csv")
            df = pd.read_csv(output_csv)

            # Extract properties for the center position
            center_props = df.iloc
            return {
                "rsa": center_props["rsa"],
                "disorder": center_props["disorder"]
            }
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error running NetSurfP-3.0: {e}")
            return None
        finally:
            # Clean up temporary files
            if os.path.exists(temp_fasta_path):
                os.remove(temp_fasta_path)
            if os.path.exists(temp_output_dir):
                import shutil
                shutil.rmtree(temp_output_dir)


# =======================================
# 4. NSGA-II Optimizer Class
# =======================================
class NSGA2Optimizer:
    """Manages the multi-objective optimization using NSGA-II."""

    def __init__(self, config: Config, esm_manager: ESM3Manager, structure_predictor: StructurePredictor):
        self.config = config
        self.esm_manager = esm_manager
        self.structure_predictor = structure_predictor
        self.toolbox = self._setup_toolbox()
        self.amino_acids_list = list(self.esm_manager.amino_acids)
        self._setup_blosum_matrix()

    def _setup_creator(self):
        """Sets up DEAP creator for fitness and individuals."""
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=self.config.FITNESS_WEIGHTS)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

    def _setup_blosum_matrix(self):
        """Loads and processes BLOSUM62 for mutation."""
        blosum62 = substitution_matrices.load("BLOSUM62")
        self.mutation_probs = {}
        for aa_from in self.amino_acids_list:
            scores = [blosum62[aa_from, aa_to] for aa_to in self.amino_acids_list]
            # Convert log-odds scores to probabilities via softmax
            probs = softmax(np.array(scores))
            self.mutation_probs[aa_from] = probs

    def _setup_toolbox(self):
        """Configures the DEAP toolbox with operators."""
        self._setup_creator()
        toolbox = base.Toolbox()

        # Attribute and Individual generation
        toolbox.register("attr_char", random.choice, self.amino_acids_list)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_char, self.config.SEQ_LENGTH)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Operators
        toolbox.register("evaluate", self.evaluate_fitness)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate_blosum)
        toolbox.register("select", tools.selNSGA2)

        return toolbox

    def evaluate_fitness(self, individual: list) -> tuple:
        """Evaluates the three fitness objectives for an individual."""
        sequence = "".join(individual)

        # Objective 1: PTM Motif Score (Maximization)
        motif_score = 1.0 if self.config.MOTIF_REGEX.match(sequence) else 0.0

        # Objective 2: Structural Fitness Score (Maximization)
        struct_props = self.structure_predictor.predict(sequence)
        if struct_props:
            # We want high RSA and high disorder
            # RSA is 0-1, disorder is 0-1. Simple average for now.
            structure_score = (struct_props["rsa"] + struct_props["disorder"]) / 2.0
        else:
            structure_score = 0.0  # Penalize if prediction fails

        # Objective 3: Sequence Plausibility (Minimization of Perplexity)
        perplexity = self.esm_manager.calculate_pseudo_perplexity(sequence)

        return motif_score, structure_score, perplexity

    def mutate_blosum(self, individual: list) -> tuple:
        """Mutates an individual using BLOSUM62-derived probabilities."""
        for i in range(len(individual)):
            if random.random() < self.config.MUTATION_PROB:
                # Ensure center position remains a valid PTM AA
                if i == self.config.CENTER_POS:
                    individual[i] = random.choice(list(self.config.PTM_AA))
                else:
                    original_aa = individual[i]
                    if original_aa in self.mutation_probs:
                        new_aa = random.choices(self.amino_acids_list, weights=self.mutation_probs[original_aa], k=1)
                        individual[i] = new_aa
        return individual,

    def run(self, initial_population: list[str]) -> list[str]:
        """Runs the NSGA-II optimization algorithm."""
        # Convert initial string sequences to DEAP individuals
        population = [creator.Individual(list(seq)) for seq in initial_population]

        # Evaluate the initial population
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is the first front of the initial population
        population = self.toolbox.select(population, k=len(population))

        # Begin the generational process
        for gen in tqdm(range(self.config.GENERATIONS), desc="NSGA-II Optimization"):
            # Vary the population and select the next generation
            offspring = algorithms.varAnd(population, self.toolbox, self.config.CROSSOVER_PROB,
                                          self.config.MUTATION_PROB)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population from parents and offspring
            population = self.toolbox.select(population + offspring, k=self.config.POPULATION_SIZE)

        # Return the final Pareto front
        pareto_front = tools.selBest(population, k=self.config.POPULATION_SIZE)
        optimized_sequences = ["".join(ind) for ind in pareto_front]

        return optimized_sequences


# =======================================
# 5. Hard Negative Miner Class
# =======================================
class HardNegativeMiner:
    """Mines hard negative samples from a pool of negatives."""

    def __init__(self, config: Config, esm_manager: ESM3Manager):
        self.config = config
        self.esm_manager = esm_manager

    def find_hard_negatives(self, positive_seqs: list[str], negative_pool: list[str]) -> list[str]:
        """Finds hard negatives for each positive sequence."""
        print("Embedding positive and negative pools for hard negative mining...")
        pos_embeddings = self.esm_manager.get_embedding(positive_seqs)
        neg_embeddings = self.esm_manager.get_embedding(negative_pool)

        print(f"Fitting NearestNeighbors model on {len(negative_pool)} negative samples...")
        nn_model = NearestNeighbors(n_neighbors=self.config.HARD_NEGATIVE_K, metric='cosine', algorithm='brute')
        nn_model.fit(neg_embeddings)

        print("Querying for hard negatives...")
        distances, indices = nn_model.kneighbors(pos_embeddings)

        hard_negatives = set()
        for i in range(len(indices)):
            for idx in indices[i]:
                hard_negatives.add(negative_pool[idx])

        print(f"Found {len(hard_negatives)} unique hard negatives.")
        return list(hard_negatives)


# =======================================
# 6. Main Pipeline
# =======================================
def main():
    """Main function to run the entire data augmentation pipeline."""
    config = Config()

    # --- Step 1: Initialization ---
    print("Initializing models and tools...")
    esm_manager = ESM3Manager(config)
    structure_predictor = StructurePredictor(config)
    optimizer = NSGA2Optimizer(config, esm_manager, structure_predictor)
    hard_miner = HardNegativeMiner(config, esm_manager)

    # --- Step 2: Load and Prepare Data ---
    print(f"Loading original data from {config.INPUT_CSV}...")
    try:
        df = pd.read_csv(config.INPUT_CSV)
        df = df[df['seq'].str.len() == config.SEQ_LENGTH].drop_duplicates().reset_index(drop=True)
    except Exception as e:
        print(f"Error loading or processing CSV: {e}")
        return

    positive_df = df[df['label'] == 1]
    negative_df = df[df['label'] == 0]

    original_pos_seqs = positive_df['seq'].tolist()

    # Create a pool of negatives where the center is a modifiable AA
    potential_neg_pool = [
        seq for seq in negative_df['seq'].tolist()
        if len(seq) == config.SEQ_LENGTH and seq in config.PTM_AA
    ]

    if not original_pos_seqs:
        print("No positive sequences found in the input file. Aborting.")
        return

    # --- Step 3: Positive Sample Augmentation ---
    num_to_generate = len(df) * (config.GENERATION_FACTOR - 1)
    num_pos_to_generate = int(num_to_generate * (len(positive_df) / len(df)))

    print(f"\n--- Generating {num_pos_to_generate} new positive sequences via inpainting ---")
    initial_candidates =
    for _ in tqdm(range(num_pos_to_generate), desc="Inpainting"):
        seed_seq = random.choice(original_pos_seqs)
        new_seq = esm_manager.inpaint_sequence(seed_seq)
        if len(new_seq) == config.SEQ_LENGTH and new_seq in config.PTM_AA:
            initial_candidates.append(new_seq)

    print(f"Generated {len(initial_candidates)} initial candidates.")

    print("\n--- Optimizing positive sequences with NSGA-II ---")
    # Run optimization in batches to manage memory
    batch_size = config.POPULATION_SIZE
    optimized_pos_seqs =
    for i in range(0, len(initial_candidates), batch_size):
        batch = initial_candidates[i:i + batch_size]
        if batch:
            optimized_batch = optimizer.run(batch)
            optimized_pos_seqs.extend(optimized_batch)

    # Remove duplicates and ensure correct length
    final_pos_seqs = list(set(optimized_pos_seqs))
    final_pos_seqs =
    print(f"Obtained {len(final_pos_seqs)} unique, optimized positive sequences.")

    # --- Step 4: Negative Sample Augmentation ---
    num_neg_to_generate = num_to_generate - len(final_pos_seqs)
    print(f"\n--- Generating {num_neg_to_generate} new negative sequences ---")

    # Mine hard negatives
    if potential_neg_pool:
        hard_negatives = hard_miner.find_hard_negatives(original_pos_seqs, potential_neg_pool)
    else:
        hard_negatives =
        print("Warning: No suitable negative pool for hard negative mining.")

    # Generate final negative set
    num_hard_negatives = int(num_neg_to_generate * config.NEGATIVE_POOL_RATIO)
    num_easy_negatives = num_neg_to_generate - num_hard_negatives

    final_neg_seqs =
    # Add hard negatives
    final_neg_seqs.extend(random.sample(hard_negatives, min(num_hard_negatives, len(hard_negatives))))
    # Add easy negatives (randomly from the original pool)
    easy_negatives = list(set(negative_df['seq'].tolist()) - set(hard_negatives))
    if easy_negatives:
        final_neg_seqs.extend(random.sample(easy_negatives, min(num_easy_negatives, len(easy_negatives))))

    # Fill up with more random negatives if needed
    while len(final_neg_seqs) < num_neg_to_generate and negative_df['seq'].tolist():
        final_neg_seqs.append(random.choice(negative_df['seq'].tolist()))

    final_neg_seqs = list(set(final_neg_seqs))
    print(
        f"Generated {len(final_neg_seqs)} negative sequences ({len(hard_negatives)} hard, {len(final_neg_seqs) - len(hard_negatives)} easy).")

    # --- Step 5: Final Dataset Assembly ---
    print("\n--- Assembling and saving the final dataset ---")

    new_pos_data = [{"seq": seq, "label": 1} for seq in final_pos_seqs]
    new_neg_data = [{"seq": seq, "label": 0} for seq in final_neg_seqs]

    enhanced_df = pd.concat(, ignore_index = True)
    enhanced_df = enhanced_df.drop_duplicates(subset=['seq']).reset_index(drop=True)

    enhanced_df.to_csv(config.OUTPUT_CSV, index=False)

    print("\n=== Data Augmentation Complete ===")
    print(f"Original dataset size: {len(df)}")
    print(f"  - Positives: {len(positive_df)}")
    print(f"  - Negatives: {len(negative_df)}")
    print(f"Enhanced dataset size: {len(enhanced_df)}")
    print(f"  - Positives: {enhanced_df['label'].sum()}")
    print(f"  - Negatives: {len(enhanced_df) - enhanced_df['label'].sum()}")
    print(f"Data saved to {config.OUTPUT_CSV}")
