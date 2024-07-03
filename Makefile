LANGUAGE := es
DATASET := wiki40b
MODEL := transformer
N_MODEL_TRAIN_TOKENS := -1
TOKENIZER := bpe

DATA_DIR := ./data
CHECKPOINT_DIR := ./checkpoints
RESULTS_DIR := ./results

DATASET_DIR := $(DATA_DIR)/$(DATASET)/$(LANGUAGE)/$(TOKENIZER)/language_modeling/$(MODEL)/$(N_MODEL_TRAIN_TOKENS)
RAW_SURPRISAL_FILE := $(DATASET_DIR)/surprisals.tsv
RAW_FREQUENCY_FILE := $(DATASET_DIR)/freq_tokens.tsv
RAW_STATS_FILE := $(DATASET_DIR)/stats.tsv

CHECKPOINT_DIR_LANG := $(CHECKPOINT_DIR)/$(DATASET)/$(LANGUAGE)/$(TOKENIZER)/$(MODEL)/$(N_MODEL_TRAIN_TOKENS)
DATA_AGGREGATE_FILE := $(CHECKPOINT_DIR_LANG)/agg.tsv
PREDICTIONS_FILE := $(CHECKPOINT_DIR_LANG)/predictions.tsv
# PREDICTIONS_COST_FILE := $(CHECKPOINT_DIR_LANG)/predictions_cost.tsv
# PREDICTIONS_DONE_FILE := $(CHECKPOINT_DIR_LANG)/predictions_done.txt
CORRELATIONS_FILE := $(CHECKPOINT_DIR_LANG)/correlations.tsv
# CORRELATIONS_COST_FILE := $(CHECKPOINT_DIR_LANG)/correlations_cost.tsv

all: aggregate_data predict_length get_correlations 
# predict_length_costs

plot_correlations:
	python src/h03_results/plot_correlations_all.py --checkpoint-dir $(CHECKPOINT_DIR) --dataset $(DATASET) --model $(MODEL)

get_correlations: $(CORRELATIONS_FILE)

predict_length: $(PREDICTIONS_FILE)

# predict_length_costs: $(PREDICTIONS_DONE_FILE)

aggregate_data: $(DATA_AGGREGATE_FILE)

$(CORRELATIONS_FILE): $(PREDICTIONS_FILE)
	python src/h02_analysis/get_correlations.py --language $(LANGUAGE) --src-fname $(PREDICTIONS_FILE) --tgt-fname $(CORRELATIONS_FILE)

# $(PREDICTIONS_DONE_FILE): $(DATA_AGGREGATE_FILE)
# 	python src/h02_analysis/predict_lengths_costs.py --src-fname $(DATA_AGGREGATE_FILE) --tgt-fname $(PREDICTIONS_COST_FILE)
# 	touch $(PREDICTIONS_DONE_FILE)

$(PREDICTIONS_FILE): $(DATA_AGGREGATE_FILE)
	python src/h02_analysis/predict_lengths.py --language $(LANGUAGE) --src-fname $(DATA_AGGREGATE_FILE) --tgt-fname $(PREDICTIONS_FILE)

# Aggregate surprisal and frequency data
$(DATA_AGGREGATE_FILE): $(RAW_SURPRISAL_FILE) $(RAW_FREQUENCY_FILE) $(RAW_STATS_FILE)
	echo "Aggregate data for " $(DATASET)
	mkdir -p $(CHECKPOINT_DIR_LANG)
	python src/h01_data/agg_results.py --src-fname-freqs $(RAW_FREQUENCY_FILE) --src-fname-surps $(RAW_SURPRISAL_FILE) \
		--src-fname-stats $(RAW_STATS_FILE) --tgt-fname $(DATA_AGGREGATE_FILE)
