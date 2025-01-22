# src/models/model_pipeline.py

import os
import shutil
from tqdm import tqdm

from src.utils.logger import get_logger
from .model_analysis import ModelAnalysis

class ModelPipeline:
    """
    Coordinates data prep & model training across multiple horizon combos.
    """

    def __init__(self, model_manager, data_handler, horizon_manager):
        """
        :param model_manager: Instance of ModelManager for training/eval.
        :param data_handler: For saving data & figures (local or s3).
        :param horizon_manager: For generating horizon combos.
        """
        self.model_manager = model_manager
        self.data_handler = data_handler
        self.horizon_manager = horizon_manager
        self.logger = get_logger(self.__class__.__name__)

    # ----------------------------
    # 1) Advanced Sentiment Filter
    # ----------------------------
    def advanced_filter_sentiment(
        self,
        df,
        sentiment_columns=None,
        threshold=0.2,
        mode="any"
    ):
        """
        Filters the DataFrame to keep only rows where the specified sentiment columns meet a threshold.
        :param df: The DataFrame
        :param sentiment_columns: e.g. ["title_positive", "summary_negative", "expected_sentiment"]
        :param threshold: The minimum value required in those columns
        :param mode: "any" => row is kept if ANY col >= threshold
                     "all" => row is kept if ALL cols >= threshold
        """
        if not sentiment_columns:
            self.logger.warning("No sentiment columns specified for filtering. Returning df unchanged.")
            return df

        missing_cols = [col for col in sentiment_columns if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns {missing_cols}. Not used in sentiment filter.")

        valid_cols = [col for col in sentiment_columns if col in df.columns]
        if not valid_cols:
            self.logger.warning("No valid sentiment columns found. Returning df unchanged.")
            return df

        orig_len = len(df)

        # Create a boolean mask for each sentiment column
        bool_masks = [(df[col] >= threshold) for col in valid_cols]

        if mode == "any":
            combined_mask = bool_masks[0]
            for m in bool_masks[1:]:
                combined_mask = combined_mask | m
        else:
            # "all"
            combined_mask = bool_masks[0]
            for m in bool_masks[1:]:
                combined_mask = combined_mask & m

        filtered_df = df[combined_mask].copy()
        self.logger.info(
            f"Advanced sentiment filter: from {orig_len} -> {len(filtered_df)} "
            f"(mode='{mode}', threshold={threshold}, columns={valid_cols})."
        )
        return filtered_df

    # -----------------------------------------
    # 2) Drop "Future" Horizon Columns
    # -----------------------------------------
    def _drop_future_horizon_columns(self, X, gather_horizon_str):
        """
        Removes columns from X that refer to a horizon bigger than gather_horizon.
        E.g. if gather_horizon_str is "5_minutes", drop columns referencing "15_minutes",
        "30_minutes", etc. This prevents "time travel" features.
        """
        gather_mins = int(gather_horizon_str.split("_")[0])  # e.g. "5_minutes" -> 5
        if not hasattr(X, "columns"):
            # If X is not a DataFrame, just return as is
            return X

        keep_cols = []
        for col in X.columns:
            if "_minutes" in col:
                # e.g. "15_minutes_change" or "20_minutes_percentage_change"
                # parse out the integer portion
                try:
                    col_mins = int(col.split("_minutes")[0])
                    if col_mins <= gather_mins:
                        keep_cols.append(col)
                    # else we skip it
                except ValueError:
                    # If we can't parse an int, keep the column (maybe "title_positive" etc.)
                    keep_cols.append(col)
            else:
                # e.g. "Open", "Close", "embedding_0", "title_positive"
                keep_cols.append(col)

        X_dropped = X[keep_cols]
        dropped_count = X.shape[1] - X_dropped.shape[1]
        self.logger.info(
            f"Dropped {dropped_count} future-horizon columns for gather={gather_horizon_str}. "
            f"Remaining columns={X_dropped.shape[1]}."
        )
        return X_dropped

    # -------------------------------
    # 3) The Main Training Loop
    # -------------------------------
    def train_on_horizons(
        self,
        X,
        df,
        max_combos=1000,
        save_best_only=True,
        filter_sentiment=False,
        sentiment_threshold=0.35,
        sentiment_cols=None,
        sentiment_mode="any"
    ):
        """
        Train models across multiple horizon combos.
        :param X: Feature DataFrame (aligned embeddings).
        :param df: Preprocessed DataFrame with target columns.
        :param max_combos: Max number of horizon combos to train on.
        :param save_best_only: If True, saves the final .pt for each horizon.
        :param filter_sentiment: If True, filter rows by advanced_filter_sentiment.
        :param sentiment_threshold: The threshold used if filter_sentiment=True.
        :param sentiment_cols: The sentiment columns to check (list).
        :param sentiment_mode: "any" or "all".
        """
        # 1) Possibly filter by sentiment
        if filter_sentiment and sentiment_cols:
            df = self.advanced_filter_sentiment(
                df,
                sentiment_columns=sentiment_cols,
                threshold=sentiment_threshold,
                mode=sentiment_mode
            )
            # Align X
            X = X.loc[df.index]

        # 2) Generate horizon combos
        combos = self.horizon_manager.generate_horizon_combos()
        combos = combos[:max_combos]

        results = []
        best_mse = float("inf")
        best_info = None  # keep track of global best model

        for combo in tqdm(combos, desc="Training across horizons", unit="horizon"):
            gather = combo["gather_name"]    # e.g. "5_minutes"
            predict = combo["predict_name"]  # e.g. "10_minutes"
            target_col = f"{predict}_change"

            if target_col not in df.columns:
                self.logger.warning(f"Missing target col '{target_col}' for horizon {predict}. Skipping.")
                continue

            # 3) Filter out rows missing the target
            df_f = df.dropna(subset=[target_col])
            if df_f.empty:
                continue

            X_f = X.loc[df_f.index]
            y = df_f[target_col].values
            if X_f.empty:
                continue

            # 4) Also drop columns referencing a horizon bigger than 'gather'
            X_f = self._drop_future_horizon_columns(X_f, gather)

            # If after dropping columns, we have no features left, skip
            if X_f.shape[1] == 0:
                self.logger.warning(f"No features remain after dropping future horizon columns. Skipping.")
                continue

            model_name = f"model_{gather}_to_{predict}"
            self.logger.info(f"Training {model_name}")

            # Override model_stage => data/models/best_models/gather_to_predict
            old_stage = self.model_manager.model_stage
            horizon_stage = f"{old_stage}/best_models/{gather}_to_{predict}"
            self.model_manager.model_stage = horizon_stage

            model, metrics = self.model_manager.train_and_evaluate(X_f.values, y, model_name)

            # restore original stage
            self.model_manager.model_stage = old_stage

            if model is None:
                continue

            if save_best_only:
                best_dir = self._resolve_local_path(horizon_stage)
                os.makedirs(best_dir, exist_ok=True)
                model_path = os.path.join(best_dir, f"{model_name}.pt")
                self.model_manager.save_model(model, model_path)

            results.append({
                "gather_horizon": gather,
                "predict_horizon": predict,
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"]
            })

            if metrics["mse"] < best_mse:
                best_mse = metrics["mse"]
                best_info = {
                    "model_name": model_name,
                    "stage": horizon_stage,
                    "gather": gather,
                    "predict": predict
                }

        # If no combos produce results, exit
        if not results:
            self.logger.warning("No horizon combos produced any results.")
            return results

        # Save a summary CSV
        self._save_summary(results)

        # Copy the single best model & plots to "goated_models"
        if best_info:
            self._save_goated_model(best_info)

        return results

    # -------------------------------------------
    # 4) Utility: Resolve local path for a stage
    # -------------------------------------------
    def _resolve_local_path(self, stage_name):
        # merges the stage_name with data_handler's base_data_dir
        from src.utils.data_handler import DataHandler
        if self.data_handler.storage_mode == "s3":
            return os.path.join(self.data_handler.base_data_dir, stage_name)
        else:
            return os.path.join(self.data_handler.base_data_dir, stage_name)

    # -------------------------------------------
    # 5) Copy the Single GOATED Model
    # -------------------------------------------
    def _save_goated_model(self, best_info):
        goated_stage = "models/goated_models"
        goated_dir = self._resolve_local_path(goated_stage)
        os.makedirs(goated_dir, exist_ok=True)

        horizon_stage = best_info["stage"]
        horizon_dir = self._resolve_local_path(horizon_stage)

        model_name = best_info["model_name"]
        src_model = os.path.join(horizon_dir, f"{model_name}.pt")
        dst_model = os.path.join(goated_dir, f"{model_name}.pt")

        try:
            shutil.copy2(src_model, dst_model)
            self.logger.info(f"GOATED model copied to {dst_model}")
        except Exception as e:
            self.logger.error(f"Failed to copy goated model .pt: {e}")

        # Also copy the plots
        for fname in os.listdir(horizon_dir):
            if fname.startswith(model_name):
                src_file = os.path.join(horizon_dir, fname)
                dst_file = os.path.join(goated_dir, fname)
                try:
                    shutil.copy2(src_file, dst_file)
                    self.logger.info(f"GOATED plot copied: {dst_file}")
                except Exception as e:
                    self.logger.error(f"Failed copying goated plot: {e}")

    # -------------------------------------------
    # 6) Save a Summary Table
    # -------------------------------------------
    def _save_summary(self, results):
        from .model_analysis import ModelAnalysis
        analysis = ModelAnalysis(self.data_handler, model_stage="models")
        analysis.save_summary_table(results)
        self.logger.info("Training summary table saved.")
