"""Principal component analysis"""
import numpy as np
from kneed import KneeLocator

def locate_elbow(x, y, S=10, min_elbow=0,
                 curve='convex', direction='decreasing', online=False,
                 **kwargs):
    """Detect knee points

    Parameters
    ----------
    x : `array_like`
        x values
    y : `array_like`
        y values
    S : `float`, optional (default: 10)
        Sensitivity
    min_elbow: `int`, optional (default: 0)
        The minimum elbow location
    curve: `str`, optional (default: 'convex')
        Choose from {'convex','concave'}
        If 'concave', algorithm will detect knees,
        If 'convex', algorithm will detect elbows.
    direction: `str`, optional (default: 'decreasing')
        Choose from {'decreasing','increasing'}
    online: `bool`, optional (default: False)
        kneed will correct old knee points if True,
        kneed will return first knee if False.
    **kwargs: `dict`, optional
        Extra arguments to KneeLocator.

    Returns
    -------
    elbow: `int`
        elbow point
    """
    kneedle = KneeLocator(x[int(min_elbow):], y[int(min_elbow):],
                          S=S, curve=curve,
                          direction=direction,
                          online=online,
                          **kwargs,
                          )
    if(kneedle.elbow is None):
        elbow = len(y)
    else:
        elbow = int(kneedle.elbow)
    return elbow

def select_pcs(adata,
               n_pcs=None,
               S=1,
               curve='convex',
               direction='decreasing',
               online=False,
               min_elbow=None,
               **kwargs):
    """select top PCs based on variance_ratio

    Parameters
    ----------
    n_pcs: `int`, optional (default: None)
        If n_pcs is None,
        the number of PCs will be automatically selected with "`kneed
        <https://kneed.readthedocs.io/>`__"
    S : `float`, optional (default: 1)
        Sensitivity
    min_elbow: `int`, optional (default: None)
        The minimum elbow location
        By default, it is n_components/10
    curve: `str`, optional (default: 'convex')
        Choose from {'convex','concave'}
        If 'concave', algorithm will detect knees,
        If 'convex', algorithm will detect elbows.
    direction: `str`, optional (default: 'decreasing')
        Choose from {'decreasing','increasing'}
    online: `bool`, optional (default: False)
        kneed will correct old knee points if True,
        kneed will return first knee if False.
    **kwargs: `dict`, optional
        Extra arguments to KneeLocator.
    Returns

    """
    if(n_pcs is None):
        n_components = adata.obsm['X_pca'].shape[1]
        if(min_elbow is None):
            min_elbow = n_components/10
        n_pcs = locate_elbow(range(n_components),
                             adata.uns['pca']['variance_ratio'],
                             S=S,
                             curve=curve,
                             min_elbow=min_elbow,
                             direction=direction,
                             online=online,
                             **kwargs)
        adata.uns['pca']['n_pcs'] = n_pcs
    else:
        adata.uns['pca']['n_pcs'] = n_pcs

def select_pcs_features(adata,
                        S=1,
                        curve='convex',
                        direction='decreasing',
                        online=False,
                        min_elbow=None,
                        **kwargs):
    """select features that contribute to the top PCs

    Parameters
    ----------
    S : `float`, optional (default: 10)
        Sensitivity
    min_elbow: `int`, optional (default: None)
        The minimum elbow location.
        By default, it is #features/6
    curve: `str`, optional (default: 'convex')
        Choose from {'convex','concave'}
        If 'concave', algorithm will detect knees,
        If 'convex', algorithm will detect elbows.
    direction: `str`, optional (default: 'decreasing')
        Choose from {'decreasing','increasing'}
    online: `bool`, optional (default: False)
        kneed will correct old knee points if True,
        kneed will return first knee if False.
    **kwargs: `dict`, optional
        Extra arguments to KneeLocator.
    Returns
    -------
    """
    n_pcs = adata.uns['pca']['n_pcs']
    n_features = adata.uns['pca']['PCs'].shape[0]
    if(min_elbow is None):
        min_elbow = n_features/6
    adata.uns['pca']['features'] = dict()
    ids_features = list()
    for i in range(n_pcs):
        elbow = locate_elbow(range(n_features),
                             np.sort(
                                 np.abs(adata.uns['pca']['PCs'][:, i],))[::-1],
                             S=S,
                             min_elbow=min_elbow,
                             curve=curve,
                             direction=direction,
                             online=online,
                             **kwargs)
        ids_features_i = \
            list(np.argsort(np.abs(
                adata.uns['pca']['PCs'][:, i],))[::-1][:elbow])
        adata.uns['pca']['features'][f'pc_{i}'] = ids_features_i
        ids_features = ids_features + ids_features_i
        # print(f'#features selected from PC {i}: {len(ids_features_i)}')
    adata.var['top_pcs'] = False
    adata.var.loc[adata.var_names[np.unique(ids_features)], 'top_pcs'] = True
    print(f'#features in total: {adata.var["top_pcs"].sum()}')
