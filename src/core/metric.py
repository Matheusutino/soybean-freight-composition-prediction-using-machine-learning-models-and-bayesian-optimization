import numpy as np
import seaborn as sns
from typing import List
from matplotlib import pyplot as plt
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

class Metric:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, dpi: int = 600) -> None:
        self.y_true = y_true
        self.y_pred = y_pred

        self.dpi = dpi

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
            precision = precision_score(self.y_true, self.y_pred, average = 'weighted')
            recall = recall_score(self.y_true, self.y_pred, average = 'weighted')
            f1 = f1_score(self.y_true, self.y_pred, average = 'weighted')

            return {'accuracy': accuracy, 'precision': precision,'recall': recall, 'f1': f1}

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
            r2 = r2_score(self.y_true, self.y_pred)

            return {'mse': mse, 'rmse': rmse,'mae': mae, 'r2': r2}

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

        except Exception as e:
            raise Exception(f"An error occurred during confusion matrix generation: {str(e)}")

    def plot_feature_importance_tree(self, 
                                     model, 
                                     model_name: str, 
                                     top_k_features: int, 
                                     feature_names: List[str], 
                                     path: str):
        """
        Plots and saves feature importances for tree-based models.

        Args:
            model: The trained model object.
            model_name (str): Name of the model ('DecisionTree', 'RandomForest', 'XGBoost', 'LightGBM').
            top_k_features (int): Number of top features to plot.
            feature_names (List[str]): Names of features.
            path (str): Path to save the image.
        
        Raises:
            TypeError: If the model_name is not supported.
        """
        path_to_save_image = f'{path}/{model_name}_feature_importance.png'
        if(model_name in ['DecisionTree', 'RandomForest', 'XGBoost']):
            feature_importances = model.feature_importances_

            indices = sorted(range(len(feature_importances)), key=lambda i: feature_importances[i], reverse=True)[:top_k_features]

            plt.figure(figsize=(10, 6))
            plt.title('Top {} Feature Importance'.format(top_k_features))
            plt.barh(range(len(indices)), feature_importances[indices], align="center")
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel("Importance")
            plt.xlim([0, 1.1 * max(feature_importances[indices])])
            plt.gca().invert_yaxis()  # Inverte o eixo y para que as features mais importantes estejam no topo
            plt.tight_layout()
            plt.savefig(path_to_save_image, dpi=self.dpi, bbox_inches='tight')
            plt.close() 

        elif(model_name == 'LightGBM'):
            # Plotando importância por ganho e salvando como imagem
            ax1 = lgb.plot_importance(model, importance_type='gain', max_num_features = top_k_features, figsize=(7, 6), title='LightGBM Feature Importance (Gain)')
            plt.savefig(path + '/lgb_gain_feature_importance.png', dpi=self.dpi, bbox_inches='tight')
            plt.close()  # Fecha o plot para evitar sobreposição

            # Plotando importância por número de splits e salvando como imagem
            ax2 = lgb.plot_importance(model, importance_type='split', max_num_features = top_k_features, figsize=(7, 6), title='LightGBM Feature Importance (Split)')
            plt.savefig(path + '/lgb_split_feature_importance.png', dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            raise TypeError("Model type not supported for feature importance plotting.")
        
        plt.close()
