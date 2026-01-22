# 2.5.2 Supervised learning approaches

Supervised binary classification analyses were conducted for all pairwise diagnostic comparisons (AD vs CBS, AD vs PSP, and CBS vs PSP) using both ML and DL methods. For all models, we implemented a robust **Nested 5-fold Cross-Validation (5×5)** framework to ensure unbiased performance estimation[[IBG1]](#_msocom_1).

In the outer loop, data were stratified and split at the subject level into an **outer training set (80%)** and a held-out **outer test set (20%)**. The test set was kept completely independent and was not used during model training, hyperparameter tuning[[ET2]](#_msocom_2), or data augmentation. For ML analyses, Random Forest (RF), K-Nearest Neighbors (KNN), and Gradient Boosting (GB) classifiers were trained using either UMAP-derived features or atlas-based summary metrics.

Hyperparameters were optimized within the outer training set using an inner 5-fold grid-search cross-validation. The search space included key model-specific parameters controlling complexity, such as ensemble depth and learning dynamics (for RF and GB), or neighborhood size and distance weighting (for KNN). To assess statistical significance, permutation tests (1,000 runs) were conducted on the best-performing models to compare observed accuracy against a null distribution.

For voxel-wise AN maps, we implemented a 3D CNN framework. Prior to training, images were normalized using Min–Max scaling.

To satisfy the high data requirements of voxel-wise CNNs, **data augmentation** was applied during training. By splitting the normative HCP cohort into 10 fixed subgroups, we computed a separate average AN map for each subgroup. This procedure yielded 10 augmented AN maps per patient, all derived from the same atrophy seed but reflecting slightly different normative connectivity profiles. This approach leverages natural variability in the healthy connectome to satisfy the data demand of voxel-wise models and improve generalization. Augmented maps were used exclusively during training, whereas validation and test sets included only non-augmented maps to ensure unbiased evaluation and avoid data leakage.

**Multiple CNN architectures were evaluated, including 3D ResNet-18 (He et al., 2015), 3D AlexNet (Krizhevsky et al., 2012), and 3D VGG16 (Simonyan, 2014).**

Hyperparameters for all models were tuned using an inner grid-search cross-validation procedure. The search space included **learning rate ($10^{-2}, 10^{-3}$)**, **L2 weight decay ($10^{-4}$)**, and **training epochs (up to 50 with early stopping)**. Consistent with the original architecture proposals, all models were trained using **SGD with Momentum (0.9)**. The **batch size was fixed at 8** due to computational constraints associated with 3D volumes. Each model configuration was optimized independently, and the best-performing setting identified during the inner cross-validation was selected for retraining on the full outer training set and final evaluation on the held-out test set. *_Manca il miglior modello, dopo analisi si saprà_*.

Classification performance was evaluated using accuracy, precision, recall, F1 score, and area under the receiver operating characteristic curve (AUC–ROC). Final performance metrics are reported as mean ± standard deviation across the 5 outer folds, and confusion matrices were generated to analyze misclassifications.
