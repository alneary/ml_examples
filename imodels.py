# Databricks notebook source
from imodels import FIGSClassifierCV
import warnings
warnings.simplefilter("once")

# COMMAND ----------

import dtreevis
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
from imodels.tree.viz_utils import extract_sklearn_tree_from_figs

def view_tree(model, tree_num, x, y):
    dt = extract_sklearn_tree_from_figs(model, tree_num = tree_num, n_classes = 2)
    shadow_dtree = ShadowSKDTree(dt, x, y, x.columns, 'title', [0,1])
    
    viz_model = dtreeviz.model(shadow_dtree, x, y, x.columns, "title", [0,1])
    displayHTML(viz_model.view(scale = 1.5).svg())

# COMMAND ----------

# MAGIC %md
# MAGIC Build a cv model

# COMMAND ----------

for s in ["f1", "precision"]:
    figs_cv = FIGSClassifierCV(scoring = s)
    pred = figs_cv.fit(xtr, ytr)
    
    print("scoring", s)
    print(classification_report(ytr, figs_cv.predict(xtr))
