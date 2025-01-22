from src.utils.logger import get_logger
from .model_analysis import ModelAnalysis
import os
from tqdm import tqdm

class ModelPipeline:
    """
    Coordinates data preparation and model training across multiple time horizons.
    """

    def __init__(self, model_manager, data_handler, horizon_manager):
        """
        :param model_manager: Instance of ModelManager for training and evaluation.
        :param data_handler: Handles data loading and saving.
        :param horizon_manager: Instance of TimeHorizonManager for generating combos.
        """
        self.model_manager = model_manager
        self.data_handler = data_handler
        self.horizon_manager = horizon_manager
        self.logger = get_logger(self.__class__.__name__)

    def filter_low_impact_sentiment(
        self, df, pos_col="title_positive", neg_col="title_negative", threshold=0.2
    ):
        """
        Removes rows where neither positive nor negative sentiment is above 'threshold'.
        This helps focus on high-impact sentiment articles.

        :param df: DataFrame with sentiment columns (pos_col, neg_col).
        :param pos_col: Name of the column for positive sentiment probability.
        :param neg_col: Name of the column for negative sentiment probability.
        :param threshold: Minimum sentiment probability required to keep the row.
        :return: Filtered DataFrame.
        """
        if pos_col not in df.columns or neg_col not in df.columns:
            self.logger.warning(
                f"Sentiment columns '{pos_col}' or '{neg_col}' not found. Skipping filter."
            )
            return df

        orig_len = len(df)
        mask = (df[pos_col] >= threshold) | (df[neg_col] >= threshold)
        filtered_df = df[mask].copy()
        self.logger.info(
            f"Filtered low-impact sentiment rows: from {orig_len} to {len(filtered_df)} "
            f"(threshold={threshold})."
        )
        return filtered_df

    def train_on_horizons(
        self,
        X,
        df,
        max_combos=10,
        save_best_only=True,
        filter_sentiment=False,
        sentiment_threshold=0.2
    ):
        """
        Train models across multiple time horizons.

        :param X: Feature DataFrame (aligned embeddings).
        :param df: Preprocessed DataFrame with target columns.
        :param max_combos: Max number of horizon combos to train on.
        :param save_best_only: If True, only saves the best model per horizon in 'models/best_models/'.
        :param filter_sentiment: If True, remove rows with sentiment < 'sentiment_threshold'.
        :param sentiment_threshold: Minimum sentiment level to keep row if filter_sentiment=True.
        :return: List of results with metrics for each horizon.
        """
        # Optionally filter out low-impact sentiment rows
        if filter_sentiment:
            df = self.filter_low_impact_sentiment(df, threshold=sentiment_threshold)
            X = X.loc[df.index]  # Keep features aligned

        # Generate horizon combos
        combos = self.horizon_manager.generate_horizon_combos()
        combos = combos[:max_combos]

        results = []

        # Initialize tqdm progress bar
        for combo in tqdm(combos, desc="Training across horizons", unit="horizon"):
            gather_horizon = combo["gather_name"]
            predict_horizon = combo["predict_name"]
            target_col = f"{predict_horizon}_change"  # or "_percentage_change"

            if target_col not in df.columns:
                self.logger.warning(f"Skipping horizon {predict_horizon}: `{target_col}` missing.")
                continue

            # Filter out rows without target
            df_filtered = df.dropna(subset=[target_col])
            if df_filtered.empty:
                self.logger.warning(
                    f"No valid rows after filtering for target_col={target_col}."
                )
                continue

            X_filtered = X.loc[df_filtered.index]
            y = df_filtered[target_col].values

            if X_filtered.empty:
                self.logger.warning(f"No features left after alignment for {target_col}.")
                continue

            model_name = f"model_{gather_horizon}_to_{predict_horizon}"
            self.logger.info(f"Training model: {model_name}")
            model, metrics = self.model_manager.train_and_evaluate(
                X_filtered.values, y, model_name
            )

            # Save final model to best_models if desired
            if save_best_only and model is not None:
                # E.g., 'models/best_models/10_minutes_to_30_minutes/model_10_minutes_to_30_minutes.pt'
                model_stage_dir = os.path.join("models", "best_models", f"{gather_horizon}_to_{predict_horizon}")
                os.makedirs(model_stage_dir, exist_ok=True)
                model_path = os.path.join(model_stage_dir, f"{model_name}.pt")
                self.model_manager.save_model(model, model_path)

            results.append({
                "gather_horizon": gather_horizon,
                "predict_horizon": predict_horizon,
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"]
            })

        if results:
            # Save summary
            self._save_summary(results)
        else:
            self.logger.warning("No horizon training results were produced.")

        return results


    def _save_summary(self, results):
        analysis = ModelAnalysis(self.data_handler)
        analysis.save_summary_table(results)
        self.logger.info("Training summary saved.")
