PYTHON=python
DATA=data/processed/cleaned_data.csv
BASE_CONFIG=configs/model_matrix.yaml
TUNED_CONFIG=configs/model_matrix_tuned.yaml
TRAIN_OUT=outputs/cv/candidates
FORECAST_CSV=outputs/forecasts/forecast_quantiles.csv
MODELS_DIR=outputs/models
PLOTS_DIR=outputs/plots

.PHONY: eda tune train forecast plots all clean

eda:
	@echo "Running EDA..."
	$(PYTHON) -m src.pipeline.run_eda --data-path $(DATA) --output-dir outputs/reports
	@echo "EDA completed. Reports saved to outputs/reports."

tune:
	@echo "Running Hyperparameter Tuning..."
	$(PYTHON) -m src.pipeline.run_tuning --data-path $(DATA) --base-config $(BASE_CONFIG) --output-config $(TUNED_CONFIG) --output-dir outputs/tuning
	@echo "Tuning completed. Tuned config saved to $(TUNED_CONFIG)."

train:
	@echo "Starting Model Training..."
	@mkdir -p $(TRAIN_OUT)
	@if [ -f $(TUNED_CONFIG) ]; then \
		cfg=$(TUNED_CONFIG); \
	else \
		echo "Tuned config not found; using $(BASE_CONFIG)"; \
		cfg=$(BASE_CONFIG); \
	fi; \
	$(PYTHON) -m src.pipeline.run_training --data-path $(DATA) --config $$cfg --output-dir $(TRAIN_OUT) --best-yaml $(TRAIN_OUT)/best_models_composite.yaml
	@echo "Model training completed. Best models saved to $(TRAIN_OUT)/best_models_composite.yaml."

forecast:
	@echo "Generating Forecasts..."
	$(PYTHON) -m src.pipeline.run_forecast --data-path $(DATA) --best-yaml $(TRAIN_OUT)/best_models_composite.yaml --models-dir $(MODELS_DIR) --forecast-path $(FORECAST_CSV)
	@echo "Forecasting completed. Forecasts saved to $(FORECAST_CSV)."

plots:
	@echo "Creating Forecast Plots..."
	$(PYTHON) -m src.pipeline.plot_forecasts --data-path $(DATA) --forecast-path $(FORECAST_CSV) --output-dir $(PLOTS_DIR)
	@echo "Plots created. Saved to $(PLOTS_DIR)."

all: eda train forecast plots

clean:
	rm -rf outputs/models outputs/forecasts outputs/plots outputs/cv outputs/reports outputs/tuning
