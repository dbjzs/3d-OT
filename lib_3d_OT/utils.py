import scanpy as sc
import numpy as np
from typing import List
import anndata
from typing import Optional, Union
from sklearn.neighbors import kneighbors_graph
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csc_matrix, csr_matrix
from typing import List
import os
os.environ['R_HOME'] = '/home/dbj/anaconda3/envs/r/lib/R'
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
robjects.r.library("mclust")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import torch
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

class Graph:
    def __init__(self, edges, edge_feats, k_neighbors, size, pcloud, express,name,truth):
        self.edges = edges
        self.size = tuple(size)
        self.edge_feats = edge_feats
        self.k_neighbors = k_neighbors
        self.pcloud = pcloud
        self.express = express
        self.name = name
        self.truth = truth 
    def to(self, device):
        self.edges = self.edges.to(device)
        self.edge_feats = self.edge_feats.to(device)
        self.pcloud = self.pcloud.to(device)
        self.express = self.express.to(device)
        return self

    @staticmethod
    def construct_graph(pcloud, nb_neighbors, expr,names,truth):
        nb_points = pcloud.shape[1]
        size_batch = pcloud.shape[0]

        distance_matrix = torch.sum(pcloud**2, -1, keepdim=True)
        distance_matrix = distance_matrix + distance_matrix.transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(
            pcloud, pcloud.transpose(1, 2)
        )

        neighbors = torch.argsort(distance_matrix, -1)[..., :nb_neighbors]
        effective_nb_neighbors = neighbors.shape[-1]
        neighbors = neighbors.reshape(size_batch, -1)

        idx = torch.arange(nb_points, device=distance_matrix.device).long()
        idx = torch.repeat_interleave(idx, effective_nb_neighbors)

        edge_feats = []
        for ind_batch in range(size_batch):
            edge_feats.append(
                pcloud[ind_batch, neighbors[ind_batch]] - pcloud[ind_batch, idx]
            )
        edge_feats = torch.cat(edge_feats, 0)

        for ind_batch in range(1, size_batch):
            neighbors[ind_batch] = neighbors[ind_batch] + ind_batch * nb_points
        neighbors = neighbors.view(-1)
        pclouds = pcloud
        express = expr
        
        graph = Graph(
            neighbors,
            edge_feats,
            effective_nb_neighbors,
            [size_batch * nb_points, size_batch * nb_points],
            pclouds,
            express,
            names,
            truth,
        )
        return graph



def prepare_data(adata, location, nb_neighbors=10):

    pcloud = adata.obsm[location][:, [0, 1]]  
    
    if issparse(adata.obsm['feat']):
        express = torch.tensor(adata.obsm['feat'].toarray(), dtype=torch.float)  # 转为密集张量
    else:
        express = torch.tensor(adata.obsm['feat'], dtype=torch.float)
        
    pcloud = torch.tensor(pcloud, dtype=torch.float)
    pcloud_batched = pcloud.unsqueeze(0)
    signal = express.unsqueeze(0)

    names = list(adata.obs_names)
    truth = adata.obs['truth'].values

    graph = Graph.construct_graph(pcloud_batched, nb_neighbors, signal,names,truth)

    return graph


def prepare_data_pca(adata, location, nb_neighbors=10):

    pcloud = adata.obsm[location][:, [0, 1]]

    if issparse(adata.obsm['feat']):
        express = torch.tensor(adata.obsm['feat'].toarray(), dtype=torch.float) 
    else:
        express = torch.tensor(adata.obsm['feat'] , dtype=torch.float)

    pcloud = torch.tensor(pcloud, dtype=torch.float)
    pcloud_batched = pcloud.unsqueeze(0)
    signal = express.unsqueeze(0)

    names = list(adata.obs_names)

    truth = adata.obs['truth'].values 
    graph = Graph.construct_graph(pcloud_batched, nb_neighbors, signal,names,truth)

    return graph





def dpca(adatas: List,join, n_comps: int = 50) -> List:
    adata_all = adatas[0].concatenate(adatas[1], join="inner")
    sc.pp.highly_variable_genes(adata_all, n_top_genes=12000, flavor="seurat_v3")
    adata_all = adata_all[:, adata_all.var.highly_variable]
    sc.pp.normalize_total(adata_all)
    sc.pp.log1p(adata_all)

    adata_1 = adata_all[adata_all.obs["batch"] == "0"]
    adata_2 = adata_all[adata_all.obs["batch"] == "1"]
    sc.pp.scale(adata_1)
    sc.pp.scale(adata_2)

    X = adata_1.X.toarray() if isinstance(adata_1.X, (csc_matrix, csr_matrix)) else adata_1.X
    Y = adata_2.X.toarray() if isinstance(adata_2.X, (csc_matrix, csr_matrix)) else adata_2.X

    cor_var = X @ Y.T 
    U, S, Vh = randomized_svd(cor_var, n_components=n_comps, random_state=0)
 
    Z_x = U @ np.diag(np.sqrt(S))
    Z_y = Vh.T @ np.diag(np.sqrt(S)) 
    adata_1.obsm['feat'] = Z_x
    adata_2.obsm['feat'] = Z_y
    
    return adata_1, adata_2
    




def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    #import sklearn
    #import scipy
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def pca(adata, use_reps=None, n_comps=10):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps,random_state=42)

    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca







def mclust_R(adata, num_cluster, modelNames='EEE', used_rep='emb_pca', random_seed=0):
    """
    Clustering using the mclust algorithm with data from adata.X.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    

    data = adata.obsm[used_rep] if used_rep in adata.obsm else adata.X.toarray()


    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(data_scaled), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, radius=50, key='X', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False,random=0,n_comp=30):
    """
    Spatial clustering using adata.X or other specified representations.
    """
    if key == 'X':  
        if not isinstance(adata.X, np.ndarray):
            data = adata.X.toarray()
        else:
            data = adata.X
    else:
        print(f"Using {key} representation for clustering...")
        data = adata.obsm[key]
    
    # PCA 降维
    pca = PCA(n_components=n_comp, random_state=2024)
    embedding = pca.fit_transform(data)
    adata.obsm['emb_pca'] = embedding   

    if method == 'mclust':
        adata = mclust_R(adata, used_rep='emb_pca', num_cluster=n_clusters,random_seed=random)
        adata.obs['3d-OT'] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain']
       

    if refinement:  
        new_type = refine_label(adata, radius, key='3d-OT')
        adata.obs['3d-OT'] = new_type

def refine_label(adata, radius=90, key='3d-OT'):
    """
    Refine clustering labels based on spatial neighbors.
    """
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    position = adata.obsm['spatial']
    from scipy.spatial.distance import cdist
    distance = cdist(position, position, metric='euclidean')     
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    return new_type


def chamfer_distance(points_a, points_b):
    dists_a_to_b = np.min(np.linalg.norm(points_a[:, np.newaxis] - points_b, axis=2), axis=1)
    dists_b_to_a = np.min(np.linalg.norm(points_b[:, np.newaxis] - points_a, axis=2), axis=1)
    
    return np.mean(dists_a_to_b**2) + np.mean(dists_b_to_a**2)