"""AUCPR for the top 5 percentile of predictions"""

import typing
import numpy as np
import datatable as dt
from datatable import f
from h2oaicore.metrics import CustomScorer
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning, loggerdata
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, auc


class top_5_percentile_aucpr_Binary(CustomScorer):
    _description = "AUCPR for the top 5 percentile of predictions"
    _binary = True
    _regression = False
    _maximize = True # whether a higher score is better
    _percentile = 0.05
    _display_name = "AUCPR5P"
    _perfect_score = 1 # the ideal score, used for early stopping once validation score achieves this value
    _supports_sample_weight = False  # whether the scorer accepts and uses the sample_weight input

    def __init__(self):
        CustomScorer.__init__(self)

    # avoid acceptance testing
    @staticmethod
    def do_acceptance_test():
        """
        Whether to enable acceptance tests during upload of recipe and during start of Driverless AI.
        Acceptance tests perform a number of sanity checks on small data, and attempt to provide helpful instructions
        for how to fix any potential issues. Disable if your recipe requires specific data or won't work on random data.
        """
        return False

    @property
    def logger(self):
        from h2oaicore import application_context
        from h2oaicore.systemutils import exp_dir
        # Don't assign to self, not picklable
        return make_experiment_logger(experiment_id=application_context.context.experiment_id, tmp_dir=None,
                                      experiment_tmp_dir=exp_dir())

    """
    Try to get the AUCPR for the top quantile
    """
    def score(self, 
            actual: np.array,
            predicted: np.array,
            sample_weight: typing.Optional[np.array] = None,
            labels: typing.Optional[np.array] = None,
            **kwargs):

        """
        :param actual:          Ground truth (correct) target values. Requires actual > 0.
        :param predicted:       Estimated target values. Requires predicted > 0.
        :param sample_weight:   weights
        :param labels:          not used
        :return: aucpr
        """

        """Initialize logger to print additional info in case of invalid inputs(exception is raised) and to enable debug prints"""
        logger = self.logger

        '''Check if any element of the arrays is nan'''
        if np.isnan(np.sum(actual)):
            loggerinfo(logger, 'Actual:%s' % str(actual))
            loggerinfo(logger, 'Nan values index:%s' % str(np.argwhere(np.isnan(actual))))
            raise RuntimeError(
                'Error during AUCPR5P score calculation. Invalid actuals values. Expecting only non-nan values')
        if np.isnan(np.sum(predicted)):
            loggerinfo(logger, 'Predicted:%s' % str(predicted))
            loggerinfo(logger, 'Nan values index:%s' % str(np.argwhere(np.isnan(predicted))))
            raise RuntimeError(
                'Error during AUCPR5P score calculation. Invalid predicted values. Expecting only non-nan values')

        if labels is not None:
            '''Check if any element of the labels array is nan'''
            if np.isnan(np.sum(labels)):
                loggerinfo(logger, 'labels:%s' % str(labels))
                loggerinfo(logger, 'labels Nan values index:%s' % str(np.argwhere(np.isnan(labels))))
                raise RuntimeError(
                    'Error during AUCPR5P score calculation. Invalid labels values. Expecting only non-nan values')

        # label actual values as 1 or 0
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        if labels is not None:
            '''Check if any element of the labels array is nan'''
            if np.isnan(np.sum(labels)):
                loggerinfo(logger, 'labels:%s' % str(labels))
                loggerinfo(logger, 'labels Nan values index:%s' % str(np.argwhere(np.isnan(labels))))
                raise RuntimeError(
                    'Error during AUCPR5P score calculation. Issue encoding labels')

        # Calculate the indices that would sort the predicted probabilities in descending order
        sorted_indices = np.argsort(predicted)[::-1]
        if len(sorted_indices) == 0 :
            loggerinfo(logger, 'sorted_indices is empty' )
            raise RuntimeError(
                'Error during AUCPR5P score calculation. Problem when sorting the predicted probabilities into sorted_indices')

        # Calculate the number of samples in the top 5 percentile
        top_5_percentile_count = int(self.__class__._percentile * len(predicted))
        if not(top_5_percentile_count > 0):
            loggerinfo(logger, 'top_5_percentile_count is not > 0' )
            raise RuntimeError(
                'Error during AUCPR5P score calculation. Problem fetching split index into top_5_percentile_count')

        # Select the top 5 percentile of samples
        actual_top_5_percentile = actual[sorted_indices[:top_5_percentile_count]].astype(int)
        predicted_top_5_percentile = predicted[sorted_indices[:top_5_percentile_count]]
        if len(actual_top_5_percentile) == 0 :
            loggerinfo(logger, 'actual_top_5_percentile is empty' )
            raise RuntimeError(
                'Error during AUCPR5P score calculation. Problem using index filter on actual_top_5_percentile')
        if len(predicted_top_5_percentile) == 0 :
            loggerinfo(logger, 'predicted_top_5_percentile is empty' )
            raise RuntimeError(
                'Error during AUCPR5P score calculation. Problem using index filter on predicted_top_5_percentile')

        
        try:
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true=actual_top_5_percentile, 
                                                          probas_pred = predicted_top_5_percentile #, pos_label = labels[1]
                                                        )

            # Calculate AUCPR
            aucpr = auc(recall, precision)
            if not((aucpr > 0) & (aucpr < 1)):
                loggerinfo(logger, 'aucpr is not between 0 and 1' )
                raise RuntimeError(
                    'Error during AUCPR5P score calculation. aucpr is out of range (0,1)')
            else:
                loggerinfo(logger, f'aucpr:{aucpr}')

        except Exception as e:
            aucpr = 0.0
            '''Print error message into DAI log file'''
            loggerinfo(logger, 'Error during AUCPR5P score calculation. Setting AUCPR to 0.0. Exception raised: %s' % str(e))
            raise
            
        return aucpr
