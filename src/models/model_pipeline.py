from src.utils.logger import get_logger
from .model_analysis import ModelAnalysis
import os
from tqdm import tqdm
import shutil

class ModelPipeline:
    """
    Coordinates data prep & model training across multiple horizon combos.
    """

    def __init__(self, model_manager, data_handler, horizon_manager):
        self.model_manager = model_manager
        self.data_handler = data_handler
        self.horizon_manager = horizon_manager
        self.logger = get_logger(self.__class__.__name__)

    def filter_low_impact_sentiment(self, df, pos_col="title_positive", neg_col="title_negative", threshold=0.2):
        if pos_col not in df.columns or neg_col not in df.columns:
            self.logger.warning(f"No columns {pos_col}/{neg_col} found. Skipping sentiment filter.")
            return df
        orig_len = len(df)
        mask = (df[pos_col] >= threshold) | (df[neg_col] >= threshold)
        filtered = df[mask].copy()
        self.logger.info(f"Filtered sentiment from {orig_len} to {len(filtered)} w/ threshold={threshold}")
        return filtered

    def train_on_horizons(
        self, X, df, max_combos=1000,
        save_best_only=True,
        filter_sentiment=False,
        sentiment_threshold=0.2
    ):
        # Optionally filter
        if filter_sentiment:
            df = self.filter_low_impact_sentiment(df, threshold=sentiment_threshold)
            X = X.loc[df.index]

        # Generate combos
        combos = self.horizon_manager.generate_horizon_combos()
        combos = combos[:max_combos]

        results = []
        best_mse = float("inf")
        best_info = None  # store best model

        from tqdm import tqdm
        for combo in tqdm(combos, desc="Training across horizons", unit="horizon"):
            gather = combo["gather_name"]
            predict = combo["predict_name"]
            target_col = f"{predict}_change"

            if target_col not in df.columns:
                self.logger.warning(f"Missing target col {target_col} for horizon {predict}")
                continue

            df_f = df.dropna(subset=[target_col])
            if df_f.empty:
                continue

            X_f = X.loc[df_f.index]
            y = df_f[target_col].values
            if X_f.empty:
                continue

            model_name = f"model_{gather}_to_{predict}"
            self.logger.info(f"Training {model_name}")

            # Override model_stage so all plots & data go inside data/models/best_models/...
            old_stage = self.model_manager.model_stage
            horizon_stage = f"{old_stage}/best_models/{gather}_to_{predict}" 
            # e.g. "models/best_models/5_minutes_to_10_minutes"
            self.model_manager.model_stage = horizon_stage

            model, metrics = self.model_manager.train_and_evaluate(X_f.values, y, model_name)

            # restore
            self.model_manager.model_stage = old_stage

            if model is None:
                continue

            if save_best_only:
                # Save model .pt in that subfolder
                best_dir = self._resolve_local_path(horizon_stage)
                os.makedirs(best_dir, exist_ok=True)
                model_path = os.path.join(best_dir, f"{model_name}.pt")
                self.model_manager.save_model(model, model_path)

            # track results
            results.append({
                "gather_horizon": gather,
                "predict_horizon": predict,
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"]
            })

            # see if it's the new best
            if metrics["mse"] < best_mse:
                best_mse = metrics["mse"]
                best_info = {
                    "model_name": model_name,
                    "stage": horizon_stage,
                    "gather": gather,
                    "predict": predict
                }

        if results:
            self._save_summary(results)
        else:
            self.logger.warning("No horizon combos produced any results.")
            return results

        # Save single goated model
        if best_info:
            self._save_goated_model(best_info)
        return results

    def _resolve_local_path(self, stage_name):
        """
        If data_handler is local, get the absolute path to that stage.
        If it's s3, we can still do local to keep code consistent.
        We'll just replicate data_handler's approach for making a local subdir.
        """
        from src.utils.data_handler import DataHandler
        if self.data_handler.storage_mode == "s3":
            # In S3 mode, we can't do a local path. We might skip or do something else
            return os.path.join(self.data_handler.base_data_dir, stage_name)
        else:
            # local
            return os.path.join(self.data_handler.base_data_dir, stage_name)

    def _save_goated_model(self, best_info):
        """
        Copies that single best model's .pt + plots from
        data/models/best_models/{gather}_to_{predict}
        to data/models/goated_models.
        """
        goated_stage = "models/goated_models"
        goated_dir = self._resolve_local_path(goated_stage)
        os.makedirs(goated_dir, exist_ok=True)

        horizon_stage = best_info["stage"]  # e.g. "models/best_models/5_minutes_to_10_minutes"
        horizon_dir = self._resolve_local_path(horizon_stage)

        model_name = best_info["model_name"]
        # copy the .pt
        src_model = os.path.join(horizon_dir, f"{model_name}.pt")
        dst_model = os.path.join(goated_dir, f"{model_name}.pt")

        try:
            shutil.copy2(src_model, dst_model)
            self.logger.info(f"GOATED model copied to {dst_model}")
        except Exception as e:
            self.logger.error(f"Failed to copy goated model .pt: {e}")

        # copy all files starting with model_{X}_to_{Y} (the plots)
        for fname in os.listdir(horizon_dir):
            if fname.startswith(model_name):
                src_file = os.path.join(horizon_dir, fname)
                dst_file = os.path.join(goated_dir, fname)
                try:
                    shutil.copy2(src_file, dst_file)
                    self.logger.info(f"GOATED plot copied: {dst_file}")
                except Exception as e:
                    self.logger.error(f"Failed copying goated plot: {e}")

    def _save_summary(self, results):
        from .model_analysis import ModelAnalysis
        # Save the summary CSV in "models" stage => data/models
        analysis = ModelAnalysis(self.data_handler, model_stage="models")
        analysis.save_summary_table(results)
        self.logger.info("Training summary table saved.")
