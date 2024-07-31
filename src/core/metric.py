import shap
import numpy as np
import seaborn as sns
from typing import List
from matplotlib import pyplot as plt
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')

class Metric:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str, dpi: int = 600, seed: int = 42) -> None:
        self.y_true = y_true
        self.y_pred = y_pred

        self.task_type = task_type
        self.dpi = dpi
        self.seed = seed

        self.labels = sorted(list(set(self.y_true)))

    def evaluate_classification(self) -> dict:
        """
        Evaluate classification performance using various metrics.

        Returns:
            dict: Dictionary containing accuracy, precision, recall, and F1 score.

        Raises:
            ValueError: If y_true and y_pred have different shapes.
        """
        try:
            accuracy = accuracy_score(self.y_true, self.y_pred)
            balanced_accuracy = balanced_accuracy_score(self.y_true, self.y_pred)
            precision = precision_score(self.y_true, self.y_pred, average = 'weighted')
            recall = recall_score(self.y_true, self.y_pred, average = 'weighted')
            f1 = f1_score(self.y_true, self.y_pred, average = 'weighted')

            return {'accuracy': accuracy, 
                    'balanced_accuracy': balanced_accuracy, 
                    'precision': precision,
                    'recall': recall, 
                    'f1': f1}

        except Exception as e:
            raise ValueError(f"Error in evaluating classification performance: {str(e)}")

    def evaluate_regression(self) -> dict:
        """
        Evaluate regression performance using various metrics.

        Returns:
            dict: Dictionary containing MSE, RMSE, MAE, and R2 score.

        Raises:
            ValueError: If y_true and y_pred have different shapes.
        """
        try:
            mse = mean_squared_error(self.y_true, self.y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_true, self.y_pred)
            mdae = median_absolute_error(self.y_true, self.y_pred)
            r2 = r2_score(self.y_true, self.y_pred)

            return {'mse': mse, 
                    'rmse': rmse,
                    'mae': mae, 
                    'mdae': mdae,
                    'r2': r2}

        except Exception as e:
            raise ValueError(f"Error in evaluating regression performance: {str(e)}")
        
    def save_confusion_matrix(self,
                              path: str, 
                              cmap: str = 'viridis') -> None:
        """
        Saves the confusion matrix as a heatmap image.

        Args:
            path (str): Path to save the image.
            cmap (str, optional): Matplotlib colormap. Defaults to 'viridis'.
            dpi (int, optional): Dots per inch for the saved figure. Defaults to 300.

        Raises:
            Exception: If an error occurs during the confusion matrix calculation or plotting.
        """
        try:
            # Calculate the confusion matrix
            cm = confusion_matrix(self.y_true, self.y_pred, labels=self.labels)

            # Create a heatmap using seaborn
            sns.heatmap(cm, annot=True, cmap=cmap, fmt='g', xticklabels=self.labels, yticklabels=self.labels)

            # Add labels to the axes
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')

            # Save the image to the specified path
            plt.savefig(path + '/confusion_matrix.png', dpi=self.dpi, bbox_inches='tight')
            plt.close()

        except Exception as e:
            raise Exception(f"An error occurred during confusion matrix generation: {str(e)}")

    def _plot_feature_importance_tree_and_save(self, 
                                                importances,
                                                indices, 
                                                feature_names, 
                                                top_k_features, 
                                                title, 
                                                save_path):
        indices = indices[:top_k_features]

        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importance")
        # plt.xlim([0, 1.1 * max(importances[indices])])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_feature_importance_tree(self, 
                                     model, 
                                     model_name: str, 
                                     top_k_features: int, 
                                     feature_names: List[str], 
                                     path: str) -> None:
        """
        Plots and saves feature importances for tree-based models.

        Args:
            model: The trained model object.
            model_name (str): Name of the model ('DecisionTree', 'RandomForest', 'XGBoost', 'LightGBM').
            top_k_features (int): Number of top features to plot.
            feature_names (List[str]): Names of features.
            path (str): Path to save the image.
        """
        if(model_name in ['DecisionTree', 'RandomForest', 'XGBoost']):
            path_to_save_image = f'{path}/{model_name}_feature_importance.png'
            feature_importances = model.feature_importances_

            indices = sorted(range(len(feature_importances)), key=lambda i: feature_importances[i], reverse=True)[:top_k_features]

            self._plot_feature_importance_tree_and_save(feature_importances, indices, feature_names, top_k_features, 
                                    f'Top {top_k_features} Feature Importance', path_to_save_image)

        elif(model_name == 'LightGBM'):
            for importance_type in ['gain', 'split']:
                # Plotando importância por ganho e salvando como imagem
                ax1 = lgb.plot_importance(model, 
                                          importance_type = importance_type, 
                                          max_num_features = top_k_features, 
                                          figsize = (7, 6), 
                                          title = f'LightGBM Feature Importance ({importance_type.capitalize()})')
                plt.savefig(path + f'/lgb_{importance_type}_feature_importance.png', dpi=self.dpi, bbox_inches='tight')
                plt.close()  

    def _plot_feature_permutation_importance_and_save(self, 
                                                    importances,
                                                    feature_names, 
                                                    top_k_features, 
                                                    title, 
                                                    save_path):
        
        sorted_idx = importances.argsort()[::-1][:top_k_features]
        top_features = feature_names[sorted_idx]
        top_importances = importances[sorted_idx]

        # Criar o gráfico de barras das top 20 features
        plt.figure(figsize=(10, 8))
        plt.barh(top_features, top_importances, align='center')
        plt.xlabel('Importância Relativa')
        plt.title(title)
        plt.gca().invert_yaxis()  # Inverter para mostrar a mais importante no topo
        plt.tight_layout() 
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_feature_permutation_importance(self, 
                                            model, 
                                            model_name: str, 
                                            X_val,
                                            y_val,
                                            top_k_features: int, 
                                            feature_names: List[str], 
                                            path: str,
                                            n_repeats: int,
                                            encoder = None) -> None:
        if(model_name == 'MLP'):
            print("Feature importance for MLP not implemented!")
            return

        result = permutation_importance(estimator = model, 
                                        X = X_val, 
                                        y = y_val,
                                        n_repeats = n_repeats,
                                        random_state = self.seed,
                                        n_jobs = -1)
        
        importances = result.importances_mean

        self._plot_feature_permutation_importance_and_save(importances,
                                                            feature_names, 
                                                            top_k_features, 
                                                            title = f'Top {top_k_features} Feature Importance using permutation importance', 
                                                            save_path = f'{path}/feature_importance_permutation.png')
        
        if(self.task_type == 'classification'):
            classes = np.unique(y_val)
            for cls in classes:
                mask = (y_val == cls)
                result = permutation_importance(estimator=model, 
                                                X=X_val[mask], 
                                                y=y_val[mask],
                                                n_repeats=n_repeats,
                                                random_state=self.seed,
                                                n_jobs=-1)
                importances = result.importances_mean

                if(model_name == 'XGBoost'):
                    int_to_label = {idx: label for idx, label in enumerate(encoder.get_classes())}
                    cls = int_to_label[cls]

                self._plot_feature_permutation_importance_and_save(importances,
                                                                feature_names, 
                                                                top_k_features, 
                                                                title = f'Top {top_k_features} Feature Importance using permutation importance for class {cls}', 
                                                                save_path = f'{path}/feature_importance_permutation_for_class_{cls}.png')
                
    def plot_feature_importance(self, 
                                model, 
                                model_name: str, 
                                X_val,
                                y_val,
                                top_k_features: int, 
                                feature_names: List[str], 
                                n_repeats: int,
                                path: str,
                                encoder = None) -> None:
        
        self.plot_feature_importance_tree(model = model, 
                                    model_name = model_name, 
                                    top_k_features = top_k_features, 
                                    feature_names = feature_names, 
                                    path = path)
        
        self.plot_feature_permutation_importance(model = model, 
                                                model_name = model_name, 
                                                X_val = X_val,
                                                y_val = y_val,
                                                top_k_features = top_k_features, 
                                                feature_names = feature_names, 
                                                path = path,
                                                n_repeats = n_repeats,
                                                encoder = encoder) 




