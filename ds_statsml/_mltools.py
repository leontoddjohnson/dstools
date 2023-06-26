import json
import pickle as pkl
import gzip
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, current_process
from sklearn.tree._tree import TREE_LEAF
from glob import glob


def get_file(filename):
    try:
        with gzip.open(filename, 'rb') as f:
            data = pkl.load(f)
    except:
        with open(filename, 'rb') as f:
            data = pkl.load(f)

    print(f'[{current_process().pid}] Opened estimator in {filename}.')

    return data


def _save_tree(work):
    i, est, modeldir = work

    print(f'[{current_process().pid}] Saving {modeldir}/estimator_{i + 1}.pkl ...')
    with gzip.open(f"{modeldir}/estimator_{i + 1}.pkl", "wb") as f:
        pkl.dump(est, f, protocol=2)


def get_tree_mapping_pred(work):
    X, json_fpath = work
    print(f"[{current_process().pid}] Predicting for {json_fpath} ...")
    mapping = TreeMapping()
    preds = mapping.predict_from_rules(X, json_fpath)

    return preds


def save_tree_mapping(work):
    i, tree_mapping_dir, estimator, max_bit, round_to_int = work
    mapping = TreeMapping(estimator, max_bit, round_to_int)
    mapping.get_tree_mapping(tree_mapping_dir + f"/estimator_{i + 1}.json")
    print(f"[{current_process().pid}] Mapping for estimator {i + 1} saved in {tree_mapping_dir}/estimator_{i + 1}.json.")


class LoadedTrees(object):

    def __init__(self, model=None, modeldir=None, n_jobs=None):
        if n_jobs is not None and n_jobs < 1:
            n_jobs = None

        if model is not None:
            self.estimators_ = model.estimators_

        elif modeldir is not None:
            self.files = [modeldir + "/" + f for f in os.listdir(modeldir)]
            with Pool(n_jobs) as p:
                self.estimators_ = p.map(get_file, self.files)

        else:
            raise AttributeError("Either `model` or `modeldir` must be defined.")

        self.n_jobs = n_jobs

    def __del__(self):
        del self.estimators_

    def predict(self, X, tree_mapping_dir=None):
        if tree_mapping_dir is not None:
            allwork = zip([X] * len(glob(tree_mapping_dir + "/*")), glob(tree_mapping_dir + "/*"))

            with Pool(self.n_jobs) as p:
                results = p.map(get_tree_mapping_pred, list(allwork))

            return np.vstack(results).mean(axis=0).round(1)

        ensemble_mean = np.zeros(X.shape[0])
        for est in self.estimators_:
            mean = est.predict(X, return_std=False)
            ensemble_mean += mean

        ensemble_mean /= len(self.estimators_)

        return np.round(ensemble_mean, 1)

    def save_trees(self, modeldir='.', modelname='mondrian'):

        modeldir = modeldir + '/' + modelname
        try:
            os.mkdir(modeldir)
        except FileExistsError:
            os.mkdir(modeldir + "_" + pd.Timestamp.today().strftime("%Y-%m-%d_%H%M%S"))
            modeldir = modeldir + "_" + pd.Timestamp.today().strftime("%Y-%m-%d_%H%M%S")

        ests = self.estimators_

        if self.n_jobs > 1:
            with Pool(self.n_jobs if self.n_jobs > 0 else None) as p:
                _ = p.map(_save_tree, list(zip(range(len(ests)), ests, [modeldir] * len(ests))))
        else:
            for i, est in enumerate(list(zip(range(len(ests)), ests, [modeldir] * len(ests)))):
                _save_tree(est)

        print(f"Pickled model estimators to the folder {modeldir}.")

    def save_tree_mappings(self, tree_mapping_dir, max_bit=32, round_to_int=True):

        n_estimators = len(self.estimators_)
        allwork = zip(range(n_estimators),
                      [tree_mapping_dir] * n_estimators,
                      self.estimators_,
                      [max_bit] * n_estimators,
                      [round_to_int] * n_estimators)

        print(f"Saving tree mappings for {n_estimators} estimators in {tree_mapping_dir}.")
        with Pool(self.n_jobs) as p:
            p.map(save_tree_mapping, allwork)


class TreeMapping(object):

    def __init__(self, dtree=None, max_bit=32, round_to_int=False):
        if dtree is not None:
            self.tree_ = dtree.tree_
            self.n_features = self.tree_.n_features
            self._inf = 2 ** (max_bit - 1)
            self.round_to_int = round_to_int

    def _get_leaf_rules(self, child_id):
        '''
        If the rules for feature_i are [a, b], then a < feature_i <= b.
        '''
        c_left = self.tree_.children_left
        c_right = self.tree_.children_right
        s_feature = self.tree_.feature
        s_thresh = self.tree_.threshold
        round_to_int = self.round_to_int

        root_id = 0 if not hasattr(self.tree_, 'root') else self.tree_.root

        leaf_rules = [[-self._inf, self._inf - 1] for i in range(self.n_features)]

        def recurse(child_id, leaf_rules):
            child_type = -1 if child_id in c_left else 1  # -1 is a left child and 1 is a right child
            parent_id = np.where(c_left == child_id)[0][0] if child_type < 0 else np.where(c_right == child_id)[0][0]

            if child_type < 0 and s_thresh[parent_id] < leaf_rules[s_feature[parent_id]][1]:
                leaf_rules[s_feature[parent_id]][1] = int(round(s_thresh[parent_id])) if round_to_int else s_thresh[parent_id]

            if child_type > 0 and leaf_rules[s_feature[parent_id]][0] < s_thresh[parent_id]:
                leaf_rules[s_feature[parent_id]][0] = int(round(s_thresh[parent_id])) if round_to_int else s_thresh[parent_id]

            if parent_id == root_id:
                return None

            recurse(parent_id, leaf_rules)

        recurse(child_id, leaf_rules)

        return leaf_rules

    def get_tree_mapping(self, json_fpath=None):
        c_left = self.tree_.children_left
        n_value = self.tree_.value

        leaf_ids = np.where(c_left == TREE_LEAF)[0]
        self.leaf_values = [x[0][0] for x in n_value[leaf_ids]]
        self.tree_rules = [self._get_leaf_rules(leaf_id) for leaf_id in leaf_ids]

        if json_fpath is None:
            return {"rules": self.tree_rules, "values": self.leaf_values}

        else:
            if os.path.isfile(json_fpath):
                fpath = json_fpath[:json_fpath.rfind('.')]
                json_fpath = fpath + "_" + pd.Timestamp.today().strftime("%Y-%m-%d_%H%M%S") + ".json"

            with open(json_fpath, "w") as f:
                json.dump({"rules": self.tree_rules, "values": self.leaf_values}, f)

    def predict_from_rules(self, X, json_fpath=None):

        if json_fpath is None:
            tree_rules = self.tree_rules
            leaf_values = self.leaf_values
            n_features = self.n_features
        else:
            with open(json_fpath, "r") as f:
                tree_mapping = json.load(f)

            tree_rules = tree_mapping['rules']
            leaf_values = tree_mapping['values']
            n_features = len(tree_mapping['rules'][0])

        preds = []

        for x in X:
            for rule, value in zip(tree_rules, leaf_values):
                if np.sum(~np.array([rule[i][0] < x[i] <= rule[i][1] for i in range(n_features)])) == 0:
                    preds.append(value)
                    break

        return np.array(preds)