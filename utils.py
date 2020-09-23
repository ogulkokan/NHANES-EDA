
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler

# preprocess dataset
def preprocess_dataset(dataset=None, remove_missing=60, remove_empty_rows=True):
    """ Simple preprocessing before PCA and T-SNE
        remove_missing: default value is 60. check all features in the dataset and remove it if
        more than 60 percent is empty.
        remove_empty_rows: check all rows and if it has None value remove row
    """
    print('feature size before dropping:{}'.format(dataset.shape[1]))
    dataset_after_drop = dataset.dropna(thresh=dataset.shape[0]*remove_missing/100, how='all',axis=1)
    print('feature size after dropping:{}'.format(dataset_after_drop.shape[1]))
    print('row size before dropping:{}'.format(dataset_after_drop.shape[0]))
    if remove_empty_rows is True:
        df_final = dataset_after_drop.dropna(inplace=False).reset_index (drop=True)
        print('row size after dropping:{}'.format(df_final.shape[0]))
        print('---------------')
        print('final shape:{}'.format(df_final.shape))
        return df_final
    else:
        return dataset_after_drop

def impute_dataset(dataset=None):
    dataset = dataset.copy()
    imputer = KNN()
    imputed_data = pd.DataFrame(np.round(imputer.fit_transform(dataset)),columns = dataset.columns)
    return imputed_data


def reduce_feature(data=None, variation_percentage=90):
    total_features = data.shape[1]
    n_components = round(total_features / 2)
    
    
    pca_n_comp = PCA(n_components=n_components)
    
    pca_reduced_results = pca_n_comp.fit_transform(data)
    cumulative_variation = np.sum(pca_n_comp.explained_variance_ratio_)
    
    print('Cumulative explained variation for {} principal components: {}'.format(n_components, cumulative_variation))
    
    print(cumulative_variation)
    if variation_percentage - 1 <= cumulative_variation * 100 <= variation_percentage + 1:
        return pca_reduced_results
    
    elif cumulative_variation * 100 < variation_percentage:
        add_number = (total_features + n_components) / n_components
        n_components = round(n_components + add_number)
        
        pca_n_comp = PCA(n_components=round(n_components))
    
        pca_reduced_results = pca_n_comp.fit_transform(data)
        cumulative_variation = np.sum(pca_n_comp.explained_variance_ratio_)
        print('Cumulative explained variation for {} principal components: {}'.format(n_components, cumulative_variation))
        return pca_reduced_results
        
    elif cumulative_variation * 100 > variation_percentage:
            add_number = (total_features + n_components) / n_components
            n_components = round(n_components - add_number)

            pca_n_comp = PCA(n_components=n_components)

            pca_reduced_results = pca_n_comp.fit_transform(data)
            cumulative_variation = np.sum(pca_n_comp.explained_variance_ratio_)
            print('Cumulative explained variation for {} principal components: {}'.format(n_components, cumulative_variation))
            return pca_reduced_results


def prepare_dataset(dataset=None, desired_class=None, with_values=True):
    """ Choose one class from dataset and split and scale before
    fitting for PCA and t-sne

        desired_class: string
    """
    if with_values is True:

        y = dataset[desired_class].values
        X = dataset.drop([desired_class], axis=1).values
        target_names = np.unique(y)
        
        # Scale Data
        X_std = StandardScaler().fit_transform(X)
        return X_std, y, target_names
    else:

        y2 = dataset[desired_class]
        X2 = dataset.drop([desired_class], axis=1)
        target_names = np.unique(y)

        # Scale Data
        X_std = StandardScaler().fit_transform(X2)
        return X2, y2, target_names


def fit_PCA_tsne(X=None, y=None, n_components=None, perplexity=50, n_iter=2000, metric=None, plot=True):

    pca = PCA(n_components=n_components)
    pca_2d = pca.fit_transform(X)

    # Invoke the TSNE method
    tsne = TSNE(n_components=n_components, verbose=1, perplexity=perplexity, n_iter=n_iter, metric=metric)
    tsne_result = tsne.fit_transform(X)
    
    if plot is True:
        compare_plots(y, pca_result=pca_2d, tsne_result=tsne_result)


def compare_plots(y, pca_result=None, tsne_result=None, ):

    plt.figure(figsize = (16,11))
    plt.subplot(121)
    plt.scatter(pca_result[:,0],pca_result[:,1], c = y, 
                cmap = "coolwarm", edgecolor = "None", alpha=0.35)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.colorbar()
    plt.title('PCA Scatter Plot')
    plt.subplot(122)
    plt.scatter(tsne_result[:,0],tsne_result[:,1],  c = y, 
                cmap = "coolwarm", edgecolor = "None", alpha=0.35)
    plt.colorbar()
    plt.legend()
    plt.title('TSNE Scatter Plot')
    plt.show()