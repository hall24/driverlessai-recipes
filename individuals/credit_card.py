"""Custom Final Individual 0 from Experiment butugewi """

# EXAMPLE USE CASES THAT REQUIRE MINOR MODIFICATIONS TO RECIPE:
# 1) FROZEN INDIVIDUALS: By default, the custom individual acts like a normal internal DAI individual,
#    which has its features and model hyperparameters mutated.
#    However, mutation of features and model hyperparameters can be avoided, and new features or models can be avoided.
#    This can be achieved by setting self.params values:
#    prob_perturb_xgb = prob_add_genes = prob_prune_genes = prob_prune_by_features = prob_addbest_genes = prob_prune_by_features = 0.0
#    leading to a "frozen" individual that is not mutated.
# 2) ENSEMBLING INDIVIDUALS: If all individuals in an experiment are frozen, then no tuning or evolution is performed.
#    One can set expert toml fixed_ensemble_level to the number of such individuals to include in an ensemble.

from h2oaicore.ga import CustomIndividual


class Indiv_butugewi_finalTrue_id0(CustomIndividual):
    """ 
    Custom wrapper class used to construct DAI Individual,
    which contains all information related to model type, model parameters, feature types, and feature parameters.

    _params_valid: dict: items that can be filled for individual-level control of parameters (as opposed to experiment-level)
                         If not set (i.e. not passed in self.params), then new experiment's value is used
                         Many of these parameters match experiment dials or are like expert tomls with a similar name
                         Dict keys are paramters
                         Dict values are the types (or values if list) for each parameter
    _from_exp: dict: parameters that are pulled from experiment-level (if value True)
 """

    """ 
    Individual instances contain structures and methods for:
        1) Feature data science types (col_dict)
        2) Transformers and their parameters (genomes, one for each layer in multi-layer feature pipeline)
        3) Model hyperparameters
        4) Feature importances and description tables
        5) Mutation control parameters
        An individual is the basis of a Population.
        Each individual is a unit, separate from the experiment that created it, that is stored (on disk) as a "brain" artifact,
        which can be re-used by new experiments if certain matching conditions are met.
        A Custom Individual is a class instance used to generate an Individual.
 """

    ###########################################################################
    #
    # Type of Individual and Origin of Recipe
    _regression = False
    _binary = True
    _multiclass = False
    _unsupervised = False
    _description = 'Indiv_butugewi_finalTrue_id0'
    _display_name = 'Indiv_butugewi_finalTrue_id0'

    # Original Experiment ID
    _experiment_id_origin = '9c066842-7a4f-11ec-973c-00d861553ebb'
    # Original Experiment Description
    _experiment_description_origin = 'butugewi'

    def set_params(self):
        """
        
        Function to set individual-level parameters.
        If don't set any parameters, the new experiment's values are used.
        :return:
        
        """

        ###########################################################################
        #
        # BEGIN: VARIABLES ARE INFORMATIVE, MUST BE KEPT FOR ACCEPTANCE TESTING

        # Was best in population
        self.final_best = True
        # Was final population
        self.final_pop = True
        # Was in final model
        self.is_final = True

        # Which individual by hash
        self.hash = '304278b5-290f-4e7a-b2ac-4e6d6856c8b9'
        # Which parent individual by hash
        self.parent_hash = 'a6c81c46-28cc-4c00-b4e4-4ba7d7f4db4f'

        # Score function's (hashed) name
        self.score_f_name = 'AUC'
        # Score (if is_final=True, then this is the final base model out-of-fold score)
        self.score = 0.77823543501091
        # Score standard deviation (if folds or repeats or bootstrapping)
        self.score_sd = 0.00697769066829106
        # Tournament Score (penalized by features counts or interpretability)
        self.tournament_score = 0.7735551732326832
        # Score history during tuning and evolution
        self.score_list = [0.7689483059295176,
                           0.7689483059295176,
                           0.7697663511600694,
                           0.7723728355388477,
                           0.7714305925186463,
                           0.7714305925186463,
                           0.7728476112519861,
                           0.7728476112519861,
                           0.7735551732326832,
                           0.7735551732326832,
                           0.7735551732326832,
                           0.7735551732326832,
                           0.7735551732326832,
                           0.7735551732326832,
                           0.7735551732326832,
                           0.7735551732326832,
                           0.7735551732326832]
        # Score standard deviation history during tuning and evolution
        self.score_sd_list = [0.006660869584380485,
                              0.006660869584380485,
                              0.006760576123036179,
                              0.00684494599008233,
                              0.007146588278819641,
                              0.007146588278819641,
                              0.007277502007947496,
                              0.007277502007947496,
                              0.007140342995562222,
                              0.007140342995562222,
                              0.007140342995562222,
                              0.007140342995562222,
                              0.007140342995562222,
                              0.007140342995562222,
                              0.007140342995562222,
                              0.007140342995562222,
                              0.007140342995562222]

        # Number of classes if supervised
        self.num_classes = 2
        # Labels if classification, None for regression
        self.labels = [0, 1]

        # Shape of training frame (may include target)
        self.train_shape = (23999, 25)
        # Shape of validation frame (may include target)
        self.valid_shape = None
        # Cardinality for each column
        self.cardinality_dict = {'AGE': 55,
                                 'EDUCATION': 7,
                                 'LIMIT_BAL': 79,
                                 'MARRIAGE': 4,
                                 'PAY_0': 11,
                                 'PAY_2': 11,
                                 'PAY_3': 11,
                                 'PAY_4': 11,
                                 'PAY_5': 10,
                                 'PAY_6': 10,
                                 'SEX': 2}

        # Target column
        self.target = 'default payment next month'
        # Label counts for target column
        self.label_counts = [18630.0, 5369.0]
        # Imbalanced ratio
        self.imbalance_ratio = 3.469919910597877

        # Weight column
        self.weight_column = None
        # Time column
        self.time_column = None

        # Number of validation splits
        self.num_validation_splits = 1
        # Seed for individual
        self.seed = 148610952
        # factor of extra genes added during activation
        self.default_factor = 1
        # Ensemble level
        self.ensemble_level = 1
        #
        # END: VARIABLES ARE INFORMATIVE, MUST BE KEPT FOR ACCEPTANCE TESTING
        ###########################################################################

        ###########################################################################
        #
        # BEGIN: PARAMETERS SET FOR CUSTOM INDIVIDUAL, self.params MAY BE SET
        #
        # Explanation of entries in self.params
        self._params_doc = {'accuracy': 'accuracy dial',
                            'do_te': "Whether to support target encoding (TE) (True, False, 'only', "
                                     "'catlabel')\n"
                                     "True means can do TE, False means cannot do TE, 'only' means only "
                                     'have TE\n'
                                     "'catlabel' is special mode for LightGBM categorical handling, to "
                                     'only use that categorical handling',
                            'explore_anneal_factor': 'Explore anneal factor',
                            'explore_model_anneal_factor': 'Explore anneal factor for models',
                            'explore_model_prob': 'Explore Probability for models\n'
                                                  'Exploration vs. Exploitation of Genetic Algorithm '
                                                  'model hyperparameter is controlled via\n'
                                                  'explore_model_prob = max(explore_model_prob_lowest, '
                                                  'explore_model_prob * explore_model_anneal_factor)',
                            'explore_model_prob_lowest': 'Lowest explore probability for models',
                            'explore_prob': 'Explore Probability\n'
                                            'Exploration vs. Exploitation of Genetic Algorithm feature '
                                            'exploration is controlled via\n'
                                            'explore_prob = max(explore_prob_lowest, explore_prob * '
                                            'explore_anneal_factor)',
                            'explore_prob_lowest': 'Lowest explore probability',
                            'grow_anneal_factor': 'Annealing factor for growth',
                            'grow_prob': 'Probability to grow genome\n'
                                         'Fast growth of many genes at once is controlled by chance\n'
                                         'grow_prob = max(grow_prob_lowest, grow_prob * '
                                         'grow_anneal_factor)',
                            'grow_prob_lowest': 'Lowest growth probability',
                            'interpretability': 'interpretability dial',
                            'nfeatures_max': 'maximum number of features',
                            'nfeatures_min': 'minimum number of features',
                            'ngenes_max': 'maximum number of genes',
                            'ngenes_min': 'minimum number of genes',
                            'num_as_cat': 'whether to treat numeric as categorical',
                            'output_features_to_drop_more': 'list of features to drop from overall genome '
                                                            'output',
                            'prob_add_genes': 'Unnormalized probability to add genes',
                            'prob_addbest_genes': 'Unnormalized probability to add best genes',
                            'prob_perturb_xgb': 'Unnormalized probability to change model hyperparameters',
                            'prob_prune_by_features': 'Unnormalized probability to prune features',
                            'prob_prune_genes': 'Unnormalized probability to prune genes',
                            'random_state': 'random seed for individual',
                            'time_tolerance': 'time dial'}
        #
        # Valid types for self.params
        self._params_valid = {'accuracy': 'int',
                              'do_te': "[True, False, 'only', 'catlabel']",
                              'explore_anneal_factor': 'float',
                              'explore_model_anneal_factor': 'float',
                              'explore_model_prob': 'float',
                              'explore_model_prob_lowest': 'float',
                              'explore_prob': 'float',
                              'explore_prob_lowest': 'float',
                              'grow_anneal_factor': 'float',
                              'grow_prob': 'float',
                              'grow_prob_lowest': 'float',
                              'interpretability': 'int',
                              'nfeatures_max': 'int',
                              'nfeatures_min': 'int',
                              'ngenes_max': 'int',
                              'ngenes_min': 'int',
                              'num_as_cat': 'bool',
                              'output_features_to_drop_more': 'list',
                              'prob_add_genes': 'float',
                              'prob_addbest_genes': 'float',
                              'prob_perturb_xgb': 'float',
                              'prob_prune_by_features': 'float',
                              'prob_prune_genes': 'float',
                              'random_state': 'int',
                              'time_tolerance': 'int'}
        #
        # Parameters that may be set
        self.params = {'accuracy': 5,
                       'do_te': True,
                       'explore_anneal_factor': 0.9,
                       'explore_model_anneal_factor': 0.9,
                       'explore_model_prob': 0.2952450000000001,
                       'explore_model_prob_lowest': 0.1,
                       'explore_prob': 0.2657205000000001,
                       'explore_prob_lowest': 0.1,
                       'grow_anneal_factor': 0.5,
                       'grow_prob': 0.2,
                       'grow_prob_lowest': 0.05,
                       'interpretability': 6,
                       'nfeatures_max': 200,
                       'nfeatures_min': 1,
                       'ngenes_max': 200,
                       'ngenes_min': 1,
                       'num_as_cat': True,
                       'output_features_to_drop_more': [],
                       'prob_add_genes': 0.5,
                       'prob_addbest_genes': 0.5,
                       'prob_perturb_xgb': 0.25,
                       'prob_prune_by_features': 0.25,
                       'prob_prune_genes': 0.5,
                       'random_state': 148610952,
                       'time_tolerance': 4}
        #
        # END: PARAMETERS SET FOR CUSTOM INDIVIDUAL, MAY BE SET
        #
        ###########################################################################

        ###########################################################################
        #
        # BEGIN: CONTROL IF PARAMETERS COME FROM EXPERIMENT (True) OR CustomIndividual (False), self._from_exp MAY BE SET
        #
        self._from_exp_doc = """ 
                    "_from_exp" dictionary have keys as things that will be set from the experiment (True),
                      which then overwrites the custom individual values assigned to self. of False means use custom individual value.
                     Or "_from_exp" values can be forced to come from the self attributes in the CustomIndividual (False).
                     * False is a reasonable possible option for key 'columns', to ensure the exact column types one desires are used
                       regardless of experiment-level column types.
                     * False is default for 'seed' and 'default_factor' to reproduce individual fitting behavior as closely as possible
                       even if reproducible is not set.
                     * False is not currently supported except for 'columns', 'seed', 'default_factor'.
                     One can override the static var value in the constructor or any function call before _from_exp is actually used
                     when calling make_indiv.
 """
        #
        # The values of _from_exp are persisted for this individual when doing refit/restart of experiments
        #
        self._from_exp = {'cardinality_dict': True,
                          'columns': True,
                          'default_factor': False,
                          'ensemble_level': True,
                          'imbalance_ratio': True,
                          'label_counts': True,
                          'labels': True,
                          'num_classes': True,
                          'num_validation_splits': True,
                          'score_f': True,
                          'seed': False,
                          'target': True,
                          'target_transformer': True,
                          'time_column': True,
                          'train_shape': True,
                          'tsgi': True,
                          'valid_shape': True,
                          'weight_column': True}
        #
        # END: CONTROL IF PARAMETERS COME FROM EXPERIMENT (True) OR CustomIndividual (False), self._from_exp MAY BE SET
        #
        ###########################################################################

        ###########################################################################
        #
        # BEGIN: CONTROL SOME VALUES IN THE CONFIG.TOML FILE (AND EXPERT SETTINGS), MAY BE SET
        #
        # config_dicts are python dictionary of config.toml keys and values that should be loadable with toml.loads()
        # Tomls appear in auto-generated code only if different than DAI factory defaults.
        #
        # Any tomls placed into self.config_dict will be enforced for the entire experiment.
        # Some config tomls like time_series_causal_split_recipe must be set for acceptance testing to pass
        # if experiment ran with time_series_causal_split_recipe=false
        # Tomls added to this list in auto-generated code may be required to be set for the individual to function properly,
        # and any experiment-level tomls can be added here to control the experiment independent from
        # config.toml or experiment expert settings.
        #
        self.config_dict = {}
        #
        # self.config_dict_individual contains tomls that may be requried for the individual to behave correctly.
        # self.config_dict are applied at experiment level, while self.config_dict_individual are not.
        # E.g. monotonicity_constraints_dict can be addeed to self.config_dict_individual to only control
        # this individual's transformed features' monotonicity.
        # One can set cols_to_force_in and cols_to_force_in_sanitized to force in a feature at the experiment or individual level,
        # or one can pass force=True to the entire gene in add_transformer() below in set_genes()
        #
        self.config_dict_individual = {'allowed_coltypes_for_tgc_as_features': [],
                                       'glm_optimal_refit': False,
                                       'included_transformers': ['AutovizRecommendationsTransformer',
                                                                 'BERTTransformer',
                                                                 'CVCatNumEncodeTransformer',
                                                                 'CVTECUMLTransformer',
                                                                 'CVTargetEncodeTransformer',
                                                                 'CatOriginalTransformer',
                                                                 'CatTransformer',
                                                                 'ClusterDistCUMLDaskTransformer',
                                                                 'ClusterDistCUMLTransformer',
                                                                 'ClusterDistTransformer',
                                                                 'ClusterIdAllNumTransformer',
                                                                 'ClusterTETransformer',
                                                                 'DBSCANCUMLDaskTransformer',
                                                                 'DBSCANCUMLTransformer',
                                                                 'DateOriginalTransformer',
                                                                 'DateTimeDiffTransformer',
                                                                 'DateTimeOriginalTransformer',
                                                                 'DatesTransformer',
                                                                 'EwmaLagsTransformer',
                                                                 'FrequentTransformer',
                                                                 'ImageOriginalTransformer',
                                                                 'ImageVectorizerTransformer',
                                                                 'InteractionsTransformer',
                                                                 'IsHolidayTransformer',
                                                                 'IsolationForestAnomalyAllNumericTransformer',
                                                                 'IsolationForestAnomalyNumCatAllColsTransformer',
                                                                 'IsolationForestAnomalyNumCatTransformer',
                                                                 'IsolationForestAnomalyNumericTransformer',
                                                                 'LagsAggregatesTransformer',
                                                                 'LagsInteractionTransformer',
                                                                 'LagsTransformer',
                                                                 'LexiLabelEncoderTransformer',
                                                                 'MeanTargetTransformer',
                                                                 'NumCatTETransformer',
                                                                 'NumToCatTETransformer',
                                                                 'NumToCatWoEMonotonicTransformer',
                                                                 'NumToCatWoETransformer',
                                                                 'OneHotEncodingTransformer',
                                                                 'OriginalTransformer',
                                                                 'RawTransformer',
                                                                 'StandardScalerTransformer',
                                                                 'StringConcatTransformer',
                                                                 'TSNECUMLTransformer',
                                                                 'TextBiGRUTransformer',
                                                                 'TextCNNTransformer',
                                                                 'TextCharCNNTransformer',
                                                                 'TextLinModelTransformer',
                                                                 'TextOriginalTransformer',
                                                                 'TextTransformer',
                                                                 'TimeSeriesTargetEncTransformer',
                                                                 'TruncSVDAllNumTransformer',
                                                                 'TruncSVDCUMLDaskTransformer',
                                                                 'TruncSVDCUMLTransformer',
                                                                 'TruncSVDNumTransformer',
                                                                 'UMAPCUMLDaskTransformer',
                                                                 'UMAPCUMLTransformer',
                                                                 'WeightOfEvidenceTransformer'],
                                       'one_hot_encoding_cardinality_threshold': 11,
                                       'prob_default_lags': 0.2,
                                       'prob_lag_non_targets': 0.1,
                                       'prob_lagsaggregates': 0.2,
                                       'prob_lagsinteraction': 0.2}
        #
        # Some transformers and models may be inconsistent with experiment's config.toml or expert config toml state,
        # such as OHE for LightGBM when config.enable_one_hot_encoding in ['auto', 'off'], yet have no other adverse effect.
        # Leaving the default of self.enforce_experiment_config=False
        # will allow features and models even if they were disabled in experiment settings.
        # This avoid hassle of setting experiment config tomls to enable transformers used by custom individual.
        # Also, if False, then self.config_dict_individual are applied for this custom individual when performing
        # operations on just the individual related to feature types, feature parameters, model types, or model parameters.
        # E.g. if enable_lightgbm_cat_support was True when individual was made, that is set again for the
        # particular custom individual even if experiment using the custom individual did not set it.
        # Setting below to self.enforce_experiment_config=True will enforce consistency checks on custom individual,
        # pruning inconsistent models or transformers according to the experiment's config toml settings.
        # The value of enforce_experiment_config is persisted for this individual,
        # when doing refit/restart of experiments.
        #
        self.enforce_experiment_config = False
        #
        # Optional config toml items that are allowed and maybe useful to control.
        # By default, these are ignored, so that config.toml or experiment expert settings can control them.
        # However, these are tomls one can manually copy over to self.config_dict to enforce certain experiment-level values,
        # regardless of experiment settings.
        # Included lists like included_models and included_scorers will be populated with all allowed values,
        # if no changes to defaults were made.
        #
        self.config_dict_experiment = {'application_id': 'dai_29024',
                                       'debug_log': True,
                                       'debug_print': True,
                                       'debug_print_server': True,
                                       'hard_asserts': True,
                                       'included_models': ['Constant',
                                                           'DecisionTree',
                                                           'FTRL',
                                                           'GLM',
                                                           'ImageAuto',
                                                           'ImbalancedLightGBM',
                                                           'ImbalancedXGBoostGBM',
                                                           'LightGBM',
                                                           'LightGBMDask',
                                                           'RFCUML',
                                                           'RFCUMLDask',
                                                           'RuleFit',
                                                           'TensorFlow',
                                                           'TextALBERT',
                                                           'TextBERT',
                                                           'TextCamemBERT',
                                                           'TextDistilBERT',
                                                           'TextMultilingualBERT',
                                                           'TextRoBERTa',
                                                           'TextXLM',
                                                           'TextXLMRoberta',
                                                           'TextXLNET',
                                                           'TorchGrowNet',
                                                           'XGBoostDart',
                                                           'XGBoostDartDask',
                                                           'XGBoostGBM',
                                                           'XGBoostGBMDask',
                                                           'XGBoostRF',
                                                           'XGBoostRFDask'],
                                       'included_scorers': ['ACCURACY',
                                                            'AUC',
                                                            'AUCPR',
                                                            'F05',
                                                            'F1',
                                                            'F2',
                                                            'FDR',
                                                            'FNR',
                                                            'FOR',
                                                            'FPR',
                                                            'GINI',
                                                            'LOGLOSS',
                                                            'MACROAUC',
                                                            'MACROF1',
                                                            'MACROMCC',
                                                            'MCC',
                                                            'NPV',
                                                            'PRECISION',
                                                            'RECALL',
                                                            'TNR'],
                                       'recipe_activation': {'data': [],
                                                             'individuals': [],
                                                             'models': [],
                                                             'scorers': [],
                                                             'transformers': []},
                                       'threshold_scorer': 'F1'}
        #
        # For new/continued experiments with this custom individual,
        # to avoid any extra auto-generated individuals (that would compete with this custom individual) set
        # enable_genetic_algorithm = 'off' and set
        # fixed_num_individuals equal to the number of indivs to allow in any GA competition.
        # This is done in expert settings or add that to self.config_dict.
        # If all features and model parameters are frozen (i.e. prob_perturb_xgb, etc. are all 0), then:
        # * if 1 individual, the genetic algorithm is changed to Optuna if 'auto', else 'off'.
        # * if >1 individuals, then the genetic algorithm is disabled (set to 'off').
        # For any individuals, for the frozen case, the number of individuals is set to the number of custom individuals.
        # To disable this automatic handling of frozen or 1 custom individual,
        # set toml change_genetic_algorithm_if_one_brain_population to false.

        # For refit/retrained experiments with this custom individual or for final ensembling control,
        # to avoid any extra auto-generated individuals (that would compete with this custom individual) set
        # fixed_ensemble_level equal to the number of custom individuals desired in the final model ensembling.
        # These tomls can be set in expert settings or added to the experiment-level self.config_dict.
        #
        # To ensemble N custom individuals, set config.fixed_ensemble_level = config.fixed_num_individuals = N
        # to avoid auto-generating other competing individuals and refit/retrain a final model
        #
        # END: CONTROL SOME CONFIG TOML VALUES, MAY BE SET
        #
        ###########################################################################

    def set_model(self):
        """
        
        Function to set model and its parameters
        :return:
        
        """

        ###########################################################################
        #
        # MODEL TYPE, MUST BE SET
        #
        # Display name corresponds to hashed (for custom recipes) display names to ensure exact match
        # One can also provide short names if only one recipe
        self.model_display_name = 'LightGBM'

        ###########################################################################
        #
        # MODEL PARAMETERS, MUST BE SET
        #
        # Some system-related parameters are overwritten by DAI, e.g. gpu_id, n_jobs for xgboost
        # Monotonicity constraints remain determined by expert toml settings,
        #  e.g. monotonicity_constraints_dict can be used to constrain feature names at experiment-level of individual level.
        #  To really set own constraints in model parameters for XGBoost and LightGBM, one can set them here,
        #  but then set monotonicity_constraints_interpretability_switch toml to 15 to avoid automatic override
        #  of those monotone_constraints params
        # Some parameters like categorical_feature for LightGBM are specific to that recipe, and automatically
        #  get handled for features that use CatTransformer
        # Some parameters like learning_rate and n_estimators are specific to that recipe, and automatically
        #  are adjusted for setting of accuracy dial.  A custom recipe wrapper could be written and one could set
        #  the static var _gbm = False to avoid such changes to learning rate and n_estimators.
        # Some parameters like disable_gpus are internal to DAI but appear in the model parameters, but they are only
        #  for information purposes and do not affect the model.
        import numpy as np
        nan = np.nan
        self.model_params = {'bagging_seed': 148610954,
                             'booster': 'lightgbm',
                             'boosting_type': 'gbdt',
                             'categorical_feature': '',
                             'class_weight': None,
                             'colsample_bytree': 0.8000000000000002,
                             'deterministic': False,
                             'device_type': 'cpu',
                             'disable_gpus': False,
                             'early_stopping_rounds': 150,
                             'early_stopping_threshold': 0.0,
                             'enable_early_stopping_rounds': True,
                             'eval_metric': 'binary',
                             'feature_fraction_seed': 148610953,
                             'gamma': 0.001,
                             'gpu_device_id': 0,
                             'gpu_platform_id': 0,
                             'gpu_use_dp': False,
                             'grow_policy': 'lossguide',
                             'importance_type': 'gain',
                             'label_counts': [18630, 5369],
                             'labels': [0, 1],
                             'learning_rate': 0.05000000000000001,
                             'max_bin': 249,
                             'max_delta_step': 1.7349599552989385,
                             'max_depth': 40,
                             'max_leaves': 16,
                             'min_child_samples': 20,
                             'min_child_weight': 0.001,
                             'min_data_in_bin': 3,
                             'min_split_gain': 0.0,
                             'monotonicity_constraints': False,
                             'n_estimators': 1000,
                             'n_gpus': 1,
                             'n_jobs': 8,
                             'num_class': 1,
                             'num_classes': 2,
                             'num_leaves': 16,
                             'num_threads': 8,
                             'objective': 'binary',
                             'random_state': 148610952,
                             'reg_alpha': 2.0,
                             'reg_lambda': 2.0,
                             'scale_pos_weight': 1.0,
                             'score_f_name': 'AUC',
                             'seed': 148610952,
                             'silent': True,
                             'subsample': 0.6,
                             'subsample_for_bin': 200000,
                             'subsample_freq': 1,
                             'verbose': -1}

        ###########################################################################
        #
        # ADJUST FINAL GBM PARAMETERS, MAY BE SET
        #
        # A list of model hyperparameters to adjust back to defaults for tuning or final model building
        #  If empty list, then no changes to model parameters will be made unless a tuning stage mutation on model parameters is done
        #  For each item in list, set_default_params() will be used to fill those parameters for GA
        #  If _is_gbm=True for the class, then these parameters also will be changed for the final model based upon DAI dails
        #  _is_gbm = True is set for model_classes based upon LightGBM, XGBoost, CatBoost, etc.
        #   E.g. for _is_gbm=True these will be changed:
        #    * learning_rate
        #    * early_stopping_rounds
        #    * n_estimators (_fit_by_iteration in general if not None, if _fit_by_iteration=True),
        # After experiment is completed, the new individual in any restart/refit will not use this parameter,
        #  so tree parameters will adapt to the dial values as well as being in tuning vs. final model.
        self.adjusted_params = ['learning_rate', 'early_stopping_rounds', 'n_estimators']

        # To prevent mutation of the model hyperparameters (frozen case), in self.params set:
        # prob_perturb_xgb = 0.0

        ###########################################################################
        #
        # MODEL ORIGIN, VARIABLE IS INFORMATIVE, NO NEED TO SET
        #
        self.model_origin = 'FINAL BASE MODEL 0'


    def set_target_transformer(self):
        """
        
        Function to set target transformer.
        If don't set any target transformer, the new experiment's values are used.  E.g. this is valid for classification.
        self.target_transformer_name = "None" applies to classification
        self.target_transformer_params = {} applies to non-time-series target transformers, only for informative purposes
        :return:
        
        """

        ###########################################################################
        #
        # TARGET TRANSFORMER, MAY BE SET
        #
        # The target-transformer name is controlled here for non-time-series cases
        # For time-series cases, the config toml choices still control outcome
        self.target_transformer_name = 'None'

        ###########################################################################
        #
        # TARGET TRANSFORMER PARAMETERS, MAY BE SET
        #
        # Target transformer parameters are only for informative purposes for time-series,
        #  for which the target transformer is re-generated from experiment settings and config toml,
        #  if a time-series-based target transformation
        self.target_transformer_params = {}

    def set_genes(self):
        """
        
        Function to set genes/transformers
        :return:
        
        """

        import numpy as np
        nan = np.nan
        from collections import OrderedDict, defaultdict

        ###########################################################################
        #
        # ORIGINAL VARIABLE IMPORTANCE, VARIABLE IS INFORMATIVE, NO NEED TO SET
        #
        self.importances_orig = {'AGE': 0.01376146630276018,
                                 'BILL_AMT1': 0.04062245627467092,
                                 'BILL_AMT2': 0.03632686748965984,
                                 'BILL_AMT3': 0.0,
                                 'BILL_AMT4': 0.0,
                                 'BILL_AMT5': 0.0,
                                 'BILL_AMT6': 0.0,
                                 'EDUCATION': 0.0,
                                 'LIMIT_BAL': 0.04670962128298108,
                                 'MARRIAGE': 0.0,
                                 'PAY_0': 0.6085515602876342,
                                 'PAY_2': 0.0,
                                 'PAY_3': 0.03658255351015567,
                                 'PAY_4': 0.0,
                                 'PAY_5': 0.03670872691097043,
                                 'PAY_6': 0.0,
                                 'PAY_AMT1': 0.04002763826600479,
                                 'PAY_AMT2': 0.031720066189950366,
                                 'PAY_AMT3': 0.035155449575717694,
                                 'PAY_AMT4': 0.021805273633914563,
                                 'PAY_AMT5': 0.017160569708117912,
                                 'PAY_AMT6': 0.02081728464795037,
                                 'SEX': 0.014050465919511973}

        ###########################################################################
        #
        # COLUMN TYPES, CAN BE SET
        #
        # By default self._from_exp['columns'] = True and so this is only informative
        # If set self._from_exp['columns'] = False, then the below col_dict is used
        # This allows one to control the data types for each column in dataset.
        # NOTE: The transformers may only use subset of columns,
        #  in which case "columns" controls any new transformers as well.
        # NOTE: If any genes consume columns that are not in the given column types,
        #  then they will be automatically added.

        self.columns = {'all': ['AGE',
                                'BILL_AMT1',
                                'BILL_AMT2',
                                'BILL_AMT3',
                                'BILL_AMT4',
                                'BILL_AMT5',
                                'BILL_AMT6',
                                'EDUCATION',
                                'LIMIT_BAL',
                                'MARRIAGE',
                                'PAY_0',
                                'PAY_2',
                                'PAY_3',
                                'PAY_4',
                                'PAY_5',
                                'PAY_6',
                                'PAY_AMT1',
                                'PAY_AMT2',
                                'PAY_AMT3',
                                'PAY_AMT4',
                                'PAY_AMT5',
                                'PAY_AMT6',
                                'SEX'],
                        'any': ['AGE',
                                'BILL_AMT1',
                                'BILL_AMT2',
                                'BILL_AMT3',
                                'BILL_AMT4',
                                'BILL_AMT5',
                                'BILL_AMT6',
                                'EDUCATION',
                                'LIMIT_BAL',
                                'MARRIAGE',
                                'PAY_0',
                                'PAY_2',
                                'PAY_3',
                                'PAY_4',
                                'PAY_5',
                                'PAY_6',
                                'PAY_AMT1',
                                'PAY_AMT2',
                                'PAY_AMT3',
                                'PAY_AMT4',
                                'PAY_AMT5',
                                'PAY_AMT6',
                                'SEX'],
                        'categorical': ['AGE',
                                        'EDUCATION',
                                        'LIMIT_BAL',
                                        'MARRIAGE',
                                        'PAY_0',
                                        'PAY_2',
                                        'PAY_3',
                                        'PAY_4',
                                        'PAY_5',
                                        'PAY_6',
                                        'SEX'],
                        'catlabel': ['AGE',
                                     'EDUCATION',
                                     'LIMIT_BAL',
                                     'MARRIAGE',
                                     'PAY_0',
                                     'PAY_2',
                                     'PAY_3',
                                     'PAY_4',
                                     'PAY_5',
                                     'PAY_6',
                                     'SEX'],
                        'date': [],
                        'datetime': [],
                        'id': [],
                        'image': [],
                        'numeric': ['AGE',
                                    'BILL_AMT1',
                                    'BILL_AMT2',
                                    'BILL_AMT3',
                                    'BILL_AMT4',
                                    'BILL_AMT5',
                                    'BILL_AMT6',
                                    'EDUCATION',
                                    'LIMIT_BAL',
                                    'MARRIAGE',
                                    'PAY_0',
                                    'PAY_2',
                                    'PAY_3',
                                    'PAY_4',
                                    'PAY_5',
                                    'PAY_6',
                                    'PAY_AMT1',
                                    'PAY_AMT2',
                                    'PAY_AMT3',
                                    'PAY_AMT4',
                                    'PAY_AMT5',
                                    'PAY_AMT6',
                                    'SEX'],
                        'ohe_categorical': ['EDUCATION',
                                            'MARRIAGE',
                                            'PAY_0',
                                            'PAY_2',
                                            'PAY_3',
                                            'PAY_4',
                                            'PAY_5',
                                            'PAY_6',
                                            'SEX'],
                        'raw': [],
                        'text': [],
                        'time_column': []}

        ###########################################################################
        #
        # GENOME, MUST BE SET
        #
        # All valid parameters for genes should be provided, except:
        # * output_features_to_drop need not be passed if empty list
        # * Mutations need not be provided if want to use default values
        # Mutations or valid parameters are not shown if none, like for OriginalTransformer
        # 'gene_index' is optional, except if use:
        # *) transformed feature names in (e.g.) monotonicity_constraints_dict toml
        # *) multiple layers with specific col_dict per layer for layer > 0
        # * 'col_type' argument to add_transformer() is used in some cases to get unique DAI transformer,
        #  and it does not need to be changed or set independently of the transformer in most cases
        # * 'labels' parameter, if present in valid parameters, is handled internally by DAI and does not need to be set
        # NOTE: While some importance variable data is provided, the newly-generated individual has freshly-determined importances
        # NOTE: For custom recipes, experiments use full hashed names for transformers,
        #       which includes the file name,
        #       but if the recipe version is not important or there is only one version,
        #       then just the name of the transformer is sufficient.

        # To prevent mutation of the genome for this individual (frozen case), in self.params set:
        # prob_add_genes = prob_prune_genes = prob_prune_by_features = prob_addbest_genes = prob_prune_by_features = 0.0

        # Doc string for add_transformer():
        """
        
        transformer collector
        :obj: Transformer display name
        :gene_index: int : index to use for gene and transformed feature name
        :layer: Pipeline layer, 0 (normal single layer), 1, ... n - 1 for n layers
        :forced: Whether forcing in gene/transformer instance to avoid pruning at gene level and any derived feature level
        :mono: Whether making feature monotonic.
               False means no constraint
               True means automatic mode done by DAI
               +1, -1, 0 means specific choice
               'experiment' means depend upon only experiment settings
               Only relevant for transformed features that go into the model,
               e.g. for multi-layer case only last layer is relevant.
        :params: parameters for Transformer
        params should have every mutation key filled, else default parameters used for missing ones

        NOTE: column names are sanitized, which means characters like spaces are not allowed or special internal characters are not allowed.
        The function sanitize_string_list(column_names) can be used to convert known column names into sanitized form, however
        if there are multiple features that differ only by a sanitized string, the de-dup process is dependent on the python-sorted order of column names.
        The function sanitize_string_list can be imported as: `from h2oaicore.transformer_utils import sanitize_string_list`.
        In DAI data handling during experiment, the sanitize_string_list is called on all columns, including:
        target, cols_to_drop, weight_column, fold_column, time_groups_columns, and training/validation/test frames.
        :return:
        
        """

        # Gene Normalized Importance: 0.026744
        # Transformed Feature Names and Importances: {'0_AGE': 0.02674410678446293}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['AGE'], 'random_state': 148610952}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=0, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.078946
        # Transformed Feature Names and Importances: {'1_BILL_AMT1': 0.07894589751958847}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['BILL_AMT1'], 'random_state': 148610953}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=1, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.070598
        # Transformed Feature Names and Importances: {'2_BILL_AMT2': 0.0705978274345398}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['BILL_AMT2'], 'random_state': 148610954}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=2, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.06347
        # Transformed Feature Names and Importances: {'8_LIMIT_BAL': 0.06346997618675232}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['LIMIT_BAL'], 'random_state': 148610960}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=8, forced=False, mono=False, **params)

        # Gene Normalized Importance:       1
        # Transformed Feature Names and Importances: {'10_PAY_0': 1.0}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_0'], 'random_state': 148610962}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=10, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.071095
        # Transformed Feature Names and Importances: {'12_PAY_3': 0.07109472900629044}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_3'], 'random_state': 148610964}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=12, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.07134
        # Transformed Feature Names and Importances: {'14_PAY_5': 0.0713399350643158}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_5'], 'random_state': 148610966}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=14, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.07779
        # Transformed Feature Names and Importances: {'16_PAY_AMT1': 0.07778992503881454}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT1'], 'random_state': 148610968}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=16, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.061645
        # Transformed Feature Names and Importances: {'17_PAY_AMT2': 0.06164494529366493}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT2'], 'random_state': 148610969}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=17, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.068321
        # Transformed Feature Names and Importances: {'18_PAY_AMT3': 0.06832128763198853}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT3'], 'random_state': 148610970}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=18, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.042376
        # Transformed Feature Names and Importances: {'19_PAY_AMT4': 0.04237648472189903}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT4'], 'random_state': 148610971}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=19, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.03335
        # Transformed Feature Names and Importances: {'20_PAY_AMT5': 0.03334994241595268}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT5'], 'random_state': 148610972}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=20, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.040456
        # Transformed Feature Names and Importances: {'21_PAY_AMT6': 0.04045642167329788}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT6'], 'random_state': 148610973}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=21, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.054612
        # Transformed Feature Names and Importances: {'29_ClusterTEClusterID50:LIMIT_BAL:SEX.0': 0.05461150035262108}
        # Valid parameters: ['num_cols', 'clust_num', 'unique_vals', 'l2_norm', 'num_folds', 'cv_type', 'inflection_point', 'steepness', 'min_rows', 'multi_class', 'random_state', 'output_features_to_drop', 'labels']
        # Allowed parameters and mutations (first mutation in list is default): {'clust_num': [5, 10, 20, 50], 'unique_vals': [False], 'l2_norm': [False, True], 'num_folds': [5], 'random_state': [42], 'cv_type': ['KFold'], 'inflection_point': [10, 20, 100], 'steepness': [3, 1, 5, 10], 'min_rows': [10, None, 20, 100], 'multi_class': [False]}
        params = {'clust_num': 50,
                  'cv_type': 'KFold',
                  'inflection_point': 10,
                  'l2_norm': False,
                  'min_rows': 10,
                  'multi_class': False,
                  'num_cols': ['LIMIT_BAL', 'SEX'],
                  'num_folds': 5,
                  'random_state': 1480602788,
                  'steepness': 10,
                  'unique_vals': False}
        self.add_transformer('ClusterTETransformer', col_type='numeric', gene_index=29, forced=False, mono=False, **params)

        # Gene Normalized Importance: 0.18266
        # Transformed Feature Names and Importances: {'32_CVTE:PAY_0.0': 0.18266233801841736}
        # Valid parameters: ['cat_cols', 'num_folds', 'cv_type', 'inflection_point', 'steepness', 'min_rows', 'multi_class', 'random_state', 'output_features_to_drop', 'labels']
        # Allowed parameters and mutations (first mutation in list is default): {'num_folds': [5], 'random_state': [42], 'cv_type': ['KFold'], 'inflection_point': [10, 20, 100], 'steepness': [3, 1, 5, 10], 'min_rows': [10, None, 20, 100], 'multi_class': [False]}
        params = {'cat_cols': ['PAY_0'],
                  'cv_type': 'KFold',
                  'inflection_point': 100,
                  'min_rows': 10,
                  'multi_class': False,
                  'num_folds': 5,
                  'random_state': 4180992656,
                  'steepness': 1}
        self.add_transformer('CVTargetEncodeTransformer', col_type='categorical', gene_index=32, forced=False, mono=False, **params)


        ###########################################################################
        #
        # TIME SERIES GROUP INFO, VARIABLES ARE FOR ACCEPTANCE TESTING ONLY, NO NEED TO CHANGE
        #
        from h2oaicore.timeseries_support import LagTimeSeriesGeneInfo, NonLagTimeSeriesGeneInfo, \
            NonTimeSeriesGeneInfo, EitherTimeSeriesGeneInfoBase
        from h2oaicore.timeseries_support import DateTimeLabelEncoder
        from h2oaicore.timeseries_support import TimeSeriesProperties

        # Note: tsgi will use tsp and encoder, and tsp will use encoder
        self.tsgi_params = {'date_format_strings': {},
                           'encoder': None,
                           'overall_shift_auc': 0.9999428055555556,
                           'shifted_features': OrderedDict([('ID', 0.9999902777777778),
                                                            ('SEX', 0.5646249999999999),
                                                            ('PAY_AMT3', 0.5629289027777779)]),
                           'target': None,
                           'tgc': None,
                           'time_column': None,
                           'tsp': None,
                           'ufapt': []}
        self.tsgi = NonTimeSeriesGeneInfo(**self.tsgi_params)

        self.tsp_params = {}
        self.tsp = None

        self.encoder_params = {}
        self.encoder = None

    @staticmethod
    def is_enabled():
        """Return whether recipe is enabled. If disabled, recipe will be completely ignored."""
        return True

    @staticmethod
    def do_acceptance_test():
        """
        Return whether to do acceptance tests during upload of recipe and during start of Driverless AI.

        Acceptance tests try to make internal DAI individual out of the python code
        """
        return True

    @staticmethod
    def acceptance_test_timeout():
        """
        Timeout in minutes for each test of a custom recipe.
        """
        from h2oaicore.systemutils import config
        return config.acceptance_test_timeout
