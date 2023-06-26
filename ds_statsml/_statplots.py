import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cycler import cycler
from sklearn.linear_model import lars_path, LinearRegression
from sklearn.preprocessing import scale, label_binarize, MinMaxScaler
from sklearn.metrics import confusion_matrix, auc, roc_curve, r2_score, mean_absolute_error, median_absolute_error, recall_score, \
    average_precision_score, f1_score, accuracy_score
from numpy import interp
from itertools import cycle
from ._alt_metrics import *
from ._stattools import smooth


def abline(slope, intercept, color, axis=None):
    """Plot a line from slope and intercept"""
    if axis is None:
        axis = plt.gca()
    x_vals = np.array(axis.get_xlim())
    y_vals = intercept + slope * x_vals
    axis.plot(x_vals, y_vals, '--', color=color)


# Regression Plots
def report_scores(df_full, df_sample, actual_col, pred_col, scoring=None, name=None, plt_score=sAPE, ax=None):

    if not scoring:
        scoring = {'R2': r2_score,
                   'RMSE': RMSE,
                   'MAE': mean_absolute_error,
                   'MdAE': median_absolute_error,
                   'sMAPE': sMAPE,
                   'sMAPE2': MAAPE_inv,
                   'MAAPE': MAAPE}

    data = {score: scoring[score](df_full[actual_col], df_full[pred_col]) for score in scoring.keys()}

    if not name:
        name = 'model'
    if ax:
        axis = ax
    else:
        fig, axis = plt.subplots()
        fig.set_size_inches(10, 8)

    textstr = 'Metric Scores:\n' + '\n'.join([score + ':   ' + str(np.round(data[score], 3)) for score in scoring.keys()])

    axis.scatter(df_sample[pred_col], plt_score(df_sample[actual_col], df_sample[pred_col]), label=name, s=3)

    props = dict(boxstyle='round', alpha=0.9)

    # place a text box in upper left in axes coords
    axis.text(0.01, 0.65, s=textstr, transform=ax.transAxes, fontsize=14, color='yellow', bbox=props)
    axis.set_title(name)

    return axis


def diagnose_reg(y, X=None, model=None, y_preds=None, title=None, fitlines=False, figsize=(22, 6)):
    '''
    This plots three diagnostic plots for *linear* regression, given X (features) and *one* target, y. It looks at
    the relationship between predicted and target y values, and two residual plots.
    :param model:
    :param X: Features
    :param y: Target
    :param title: Overall title of the plot (to go on topsies).
    :return: matplotlib.figure
    '''
    if y_preds is None:
        if model and X:
            y_preds = model.predict(X)

        elif model or X:
            print("If you're going to define an X data frame, you need to also define a model, and vice versa.")
            return None

        else:
            print('Make sure to define either y_preds, or both a model and X.')
            return None

    lm_resids = y_preds - y

    fig, axes = plt.subplots(1, 3)

    g1 = sns.regplot(y_preds, y, ci=False, line_kws={'color': 'blue'}, ax=axes[0], fit_reg=fitlines, scatter_kws={'s': 5})
    g1.set_xlabel('Predicted Target')
    g1.set_ylabel('True Target')
    abline(1, 0, 'gray', axes[0])

    g2 = sns.regplot(y_preds, lm_resids, fit_reg=fitlines, lowess=fitlines, ax=axes[1], line_kws={'color': 'red'}, scatter_kws={'s': 5})
    g2.set_xlabel('Predicted Target')
    g2.set_ylabel('Residual')
    g2.axhline(y=0, color='gray', linestyle='--')

    g3 = sns.regplot(y_preds, np.abs(lm_resids), fit_reg=fitlines, lowess=fitlines, ax=axes[2], line_kws={'color': 'orange'}, scatter_kws={'s': 5})
    g3.set_xlabel('Predicted Target')
    g3.set_ylabel('|Residual|')

    fig.suptitle(title)
    fig.set_size_inches(figsize[0], figsize[1])

    return fig


def larspath(X_train, y_train, columns=()):
    '''
    Plot the LARS path for values in the X_train data frame.
    :param X_train: pd.DataFrame, Features. If not a Pandas data frame, you need to pass the column names (all) to
    columns.
    :param y_train: array-like, The target.
    :param columns: array-like, Column names, if X_train is not Pandas Dataframe.
    :return: matplotlib.figure
    '''
    X_train_scaled = scale(X_train)  # We need to scale the data because this method uses LASSO regularization
    alphas, _, coefs = lars_path(X_train_scaled, y_train, method='lasso', verbose=True)

    # colormap = plt.cm.gist_ncar
    # plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]

    if len(columns) > 0:
        features = columns

    else:
        features = X_train.columns

    fig = plt.figure(figsize=(15, 8))
    plt.set_cmap('Set1')
    cy_style = cycler('linestyle', ['-', '--', '-.'] * 4)
    cy_color = cycler('color', [plt.cm.Accent(i / X_train.shape[1]) for i in range(12)])
    plt.gca().set_prop_cycle(cy_style + cy_color)
    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed')
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('LASSO Path')
    plt.axis('tight')
    plt.legend(features, bbox_to_anchor=(1, 0.8))
    plt.show()

    return fig


def diagnose_lin_factor(factor, coeffs, X_train, y_train, intercept=0, featurecols=None, logistic=False, figsize=(18, 6)):
    '''
    Get diagnostic plots for a single factor in a regression model.
    :param coeffs: intercept (0 if there isn't one), and then the features (e.g., [0, 2, 4.3] for features 'const', 'var1', and 'var2'
    :param factor: str, The factor of interest
    :param X_train: pd.DataFrame, all the data you're interested as factors
    :param y_train: array-like, Target
    :param logistic: bool, Whether the factor of interest is logistc.
    :return: (fig, axes)
    '''

    if not hasattr(X_train, 'columns'):
        features = list(featurecols)
    else:
        features = list(X_train.columns)

    # Marginal Plot
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(figsize[0], figsize[1])

    coeff = coeffs[features.index(factor)]

    y_preds = coeff * X_train[factor] + intercept
    gm = sns.regplot(x=X_train[factor], y=y_train, lowess=not logistic, ci=False, ax=axes[0],
                     line_kws={'color': 'red', 'label': 'Best Fit'})
    axes[0].plot(X_train[factor], y_preds, color='orange', label='Model Prediction')
    axes[0].set_title('Marginal Plot')
    gm.figure.legend()

    # Added Variable
    # Define the model y = Beta*X + Gamma*z + err
    notfactor = [col for col in X_train.columns if col != factor]
    z = X_train[factor]
    X = X_train[notfactor]

    # Regress y on X, and get the residuals
    lm_yX = LinearRegression(fit_intercept=intercept != 0)
    lm_yX.fit(X, y_train)
    resid_yX = lm_yX.predict(X=X) - y_train

    # Regress z on X, and get residuals
    lm_zX = LinearRegression(fit_intercept=intercept != 0)
    lm_zX.fit(X, z)
    resid_zX = lm_zX.predict(X=X) - z

    # Plot the residuals of z on X against the residuals for y on X for the influence of z on the model
    # If you get a straight line, it means that the factor doesn't add much to the model (that is, no matter
    # the value of the residual of y on X, the difference z would add is unchanging. If it changes in a linear
    # way, then as the residuals get worse, z could rectify and add in a good way.
    sns.regplot(x=resid_zX, y=resid_yX, lowess=True, ax=axes[1], line_kws={'color': 'red'})
    axes[1].set_xlabel('Residuals of z on X')
    axes[1].set_ylabel('Residuals of y on X')
    axes[1].set_title('Added Variable Plot')
    fig.suptitle(f'Factor Analysis of {factor}')

    return fig, axes


def plot_performances(df_all, actual_col, report_cols, sample=None, score_func=sAPE):
    '''
    This will plot residual values given actual and predicted values to report on.
    :param df_all: pd.DataFrame, All the data, including actual values and values to report on.
    :param actual_col: str, The column name for the actual values
    :param report_cols: {str, list}, the column name(s) to report on. This will only plot up to three different factors.
    :param sample: {int, float in (0, 1)}, how big of a sample do you want to plot. Default == None, and this will take all the data. If you want
    to save on computing power, you can set this to be some value less than the original length of the data, depending on how big it is.
    :param score_func: str, a function that takes actuals and predictions and gives error values as an array.
    :return: matplotlib.pyplot.figure
    '''

    if np.sum(df_all.isna().values) > 0:
        print('You have some NA values ...')
        return None

    if isinstance(report_cols, str):
        report_cols = [report_cols]

    if sample is None:
        df_sample = df_all.copy()
    elif isinstance(sample, int):
        df_sample = df_all.sample(n=sample)
    elif 0 < sample < 1:
        df_sample = df_all.sample(frac=sample)
    else:
        df_sample = df_all.copy()

    df_sample.reset_index(inplace=True, drop=True)

    if len(report_cols) == 1:

        fig, axes = plt.subplots()

        _ = report_scores(df_all, df_sample, actual_col, report_cols[0], ax=axes, name=report_cols[0], plt_score=score_func)

        fig.set_size_inches(12, 8)

    else:
        report_cols = report_cols[:3]
        fig, axes = plt.subplots(1, len(report_cols), sharey=True)

        for i, col in enumerate(report_cols):
            _ = report_scores(df_all, df_sample, actual_col, col, ax=axes[i], name=col, plt_score=score_func)

        fig.set_size_inches(9 * len(report_cols), 8)

    return fig


def plot_report_by_col(df_all, actual_col, report_cols, bin_column, score_metrics=('MdAE', 'sMAPE'), scale_metric='count',
                       sortby='values', xticklabel_spacing=1):
    '''
    This function is for plotting some regression error metrics (at least two) given some predictions and actual values. This will report on the
    error metrics as they change based on some binning (class) column. So, basically, we ask the question 'For each class in <bin_column>,
    how well are <report_cols> doing at predicting <actual_col>?' ... * This could be updated to include classification models ... Just work on the
    metrics.
    :param df_all: pd.DataFrame, This contains all the data for <actual_col> and <report_cols>
    :param actual_col: str, The name of the column that contains the actual values which you'd like to compare against.
    :param report_cols: {str, list}, The column(s) that you'd s to compare against the actual column. This will have each error metric reported
    on (one plot for each metric in <score_metrics>). Also, this will only take *the first five of the <report_cols>* that you submit here. More
    than that, the plots could get congested.
    :param bin_column: str, The name of the categorical column that defines the bins.
    :param score_metrics: list of {'MAAPE', 'MAAPE_inv', 'MAE', 'MdAE', 'sMAPE', 'RMSE', 'R_squared'}, The score metrics that you'd like to report
    on given the <report_cols>. So, there will be one plot for each of these, and on each plot there will be lines for each column in <report_cols>.
    :param scale_metric: list of {'median_actuals', 'median_preds', 'mean_actuals', 'mean_preds', 'std_actuals', 'std_preds', 'count'},
    For each plot (i.e., for each <score_metric>) there will be a scaled line representing one of these values for the <bin_column>. So,
    within each bin, what does the <scale_metric> (scaled, to fit on the plot) look like?
    :param sortby: str in {'values', 'index'}, How to present the titles for the <bin_column> in the plots. If you sort by values, you're sorting
    by the <scale_metric> values for each item in <bin_column>.
    :return: maptlotlib.pyplot.Figure
    '''

    score_metrics = score_metrics[:3]
    report_cols = report_cols[:5]

    if np.sum(df_all[report_cols + [bin_column, actual_col]].isna().values) > 0:
        print('You have some NA values in the report, bin, or actual columns.')
        return None

    dfs = []

    if isinstance(report_cols, str):
        report_cols = [report_cols]

    for report_col in report_cols:

        values = df_all[bin_column].unique()

        data = {'MAAPE': [], 'MAAPE_inv': [], 'MAE': [], 'MdAE': [], 'MAPE': [], 'MdAPE': [], 'sMAPE': [], 'sMAPE2': [], 'RMSE': [], 'R_squared': [],
                'median_actuals': [], 'median_preds': [], 'mean_actuals': [], 'mean_preds': [],
                'std_actuals': [], 'std_preds': [], 'count': []}

        for value in values:
            mask = df_all[bin_column] == value
            actuals = df_all[actual_col][mask]
            preds = df_all[report_col][mask]
            data['MAAPE'].append(MAAPE(actuals, preds))
            data['MAAPE_inv'].append(MAAPE_inv(actuals, preds))
            data['MAE'].append(mean_absolute_error(actuals, preds))
            data['MdAE'].append(median_absolute_error(actuals, preds))
            data['MAPE'].append(MAPE(actuals, preds))
            data['MdAPE'].append(MdAPE(actuals, preds))
            data['sMAPE'].append(sMAPE(actuals, preds))
            data['sMAPE2'].append(sMAPE2(actuals, preds))
            data['RMSE'].append(RMSE(actuals, preds))
            data['R_squared'].append(r2_score(actuals, preds))
            data['median_actuals'].append(np.median(actuals))
            data['median_preds'].append(np.median(preds))
            data['mean_actuals'].append(np.mean(actuals))
            data['mean_preds'].append(np.mean(preds))
            data['std_actuals'].append(np.std(actuals))
            data['std_preds'].append(np.std(preds))
            data['count'].append(sum(mask))

        df_ = pd.DataFrame(index=values, data=data)

        df_.columns = [col + report_col for col in df_.columns]
        df_.rename(columns={'median_actuals' + report_col: 'median_actuals'}, inplace=True)
        df_.rename(columns={'mean_actuals' + report_col: 'mean_actuals'}, inplace=True)
        df_.rename(columns={'std_actuals' + report_col: 'std_actuals'}, inplace=True)
        df_.rename(columns={'count' + report_col: 'count'}, inplace=True)
        dfs.append(df_)

    df = pd.concat(dfs, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    df_cols = []

    for metric in score_metrics:
        for report_col in report_cols:
            df_cols.append(metric + report_col)

    df_ = df[df_cols]

    scalers = []

    for i in range(len(score_metrics)):
        df_metric = df_.iloc[:, i * len(report_cols): (i + 1) * len(report_cols)]

        scale_min = df_metric.values.flatten().mean() + 1.5 * df_metric.values.flatten().std()
        scale_max = scale_min + df_metric.values.flatten().max()
        scaler = MinMaxScaler((scale_min, scale_max))

        scalers.append(scaler)

    df_ = pd.concat((df_, df[scale_metric]), axis=1)

    if sortby == 'values':
        df_ = df_.sort_values(scale_metric)
    else:
        df_ = df_.sort_index()

    for i, metric in enumerate(score_metrics):
        scaler = scalers[i]
        df_[scale_metric + '_scaled_' + metric] = scaler.fit_transform(df_[scale_metric][:, np.newaxis])

    fig, axes = plt.subplots(1, len(score_metrics))

    for i, metric in enumerate(score_metrics):

        for report_col in report_cols:
            axes[i].plot(df_.index.astype(str), df_[metric + report_col], label=report_col)

        axes[i].plot(df_.index.astype(str), df_[scale_metric + '_scaled_' + metric], label=scale_metric + '_scaled', color='#f25e5e', ls=':')
        axes[i].legend()
        axes[i].set_title(metric)

    for axis in axes:

        if not isinstance(df_.index[0], str) and len(axis.get_xticks()) > 20 and xticklabel_spacing == 1:
            xticklabel_spacing = len(axis.get_xticks()) // 10

        ticks = axis.get_xticks()
        ticklabels = df_.index.astype(str)

        axis.set_xticks(ticks[::xticklabel_spacing])
        axis.set_xticklabels(ticklabels[::xticklabel_spacing])

        for tick in axis.get_xticklabels():
            tick.set_rotation(90)

        axis.set_xlabel(bin_column)

    fig.set_size_inches(len(score_metrics) * 10, 8)

    return fig


# Classification Plots
def diagnose_logr(model, X, y_true, factor, thresh=0.5, hasconst=True):

    x = X[factor]

    def rand_jitter(arr):
        stdev = .01 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    true_mask = (y_true == 1)

    pred_probs = model.predict(X)
    y_preds = np.array((pred_probs > thresh), dtype=int)

    fig, axes = plt.subplots(1, 2)

    if hasconst:
        intercept = model.params['const']
    else:
        intercept = 0

    coeff = model.params[factor]

    # We evaluate the threshold when everything else is zero
    thresh_pt = (np.log(thresh / (1 - thresh)) - intercept) / coeff
    axes[0].plot(x[true_mask], pred_probs[true_mask], 'bo', label='True Success', alpha=1, markersize=3)
    axes[0].plot(x[~true_mask], pred_probs[~true_mask], 'ro', label='True Failure', alpha=0.4, markersize=3)
    sns.regplot(x=x, y=y_preds, fit_reg=False,
                ax=axes[0],
                scatter_kws={'s': 4, 'color': 'black'})
    axes[0].axhline(y=thresh, color='orange', label='Model Probability Threshold')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel(factor)
    axes[0].set_ylabel('Probability of Success')
    axes[0].set_title('Logistic regression prediction')
    axes[0].legend(loc='lower right')

    axes[1].plot(x[true_mask], rand_jitter(y_preds[true_mask]), 'bo', label='True Success', alpha=1)
    axes[1].plot(x[~true_mask], rand_jitter(y_preds[~true_mask]), 'ro', label='True Failure', alpha=0.4)
    axes[1].axvline(x=thresh_pt, color='orange', label='Marginal Probability Threshold')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_yticks((0, 1))
    axes[1].set_xlabel(factor)
    axes[1].set_ylabel('Pedicted Class (1 == Success)')
    axes[1].set_title('Logistic regression prediction')
    axes[1].legend(loc='lower right')
    fig.set_size_inches(18, 10)

    conf_m = confusion_matrix(y_true, y_preds)
    print(pd.DataFrame(data=conf_m, index=['Real Pos', 'Real Neg'], columns=['Pred Pos', 'Pred Neg']), '\n')

    return fig, axes


def ROC(model, X_test, y_test, lw=2):

    y_score = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC Curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")


def ROC_multi(X_test, y_test, y_probas=None, classes=None, model=None, lw=2):
    # Binarize y_test
    if classes is None:
        try:
            classes = model.classes_
        except AttributeError:
            print('Make sure your model has a classes_ attribute, or provide the correct classes *in order* wrt y_probas or .predict_proba.')
            return None

    labels = label_binarize(y_test, classes=classes)

    # Get y_scores
    if model is not None:
        y_score = model.predict_proba(X_test)
    elif y_probas is not None:
        y_score = y_probas
    else:
        print('Please define a model or y_preds.')
        return None

    n_classes = y_score.shape[-1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(12, 8))

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    styles = cycle(['-', '--', '-.', ':'])
    for i, color, style in zip(range(n_classes), colors, styles):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, ls=style,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-Class')
    plt.legend(loc="lower right")
    plt.show()

    preds = classes[y_score.argmax(axis=1)]
    preds = label_binarize(preds, classes=classes)

    print('Precision (micro):\t', np.round(average_precision_score(labels, preds, average='micro'), 3))
    print('Precision (macro):\t', np.round(average_precision_score(labels, preds, average='macro'), 3))
    print('Recall (micro):\t\t', np.round(recall_score(labels, preds, average='micro'), 3))
    print('Recall (macro):\t\t', np.round(recall_score(labels, preds, average='macro'), 3))
    print('F1 (micro):\t\t', np.round(f1_score(labels, preds, average='micro'), 3))
    print('F1 (macro):\t\t', np.round(f1_score(labels, preds, average='macro'), 3))
    print('Accuracy:\t\t', np.round(accuracy_score(labels, preds), 3))


# Time Series Plots
def plot_compare_overtime(series1, series2, y1_label='Series 1', y2_label='Series 2', lastday='2018-09-30', title='Time Series Comparison',
                          x_label='Date', days_per_lag=1, weekday_mark='Monday', figsize=(20, 8), xticklabel_spacing=1):
    priorday = pd.Timestamp(lastday)
    dates = [priorday.strftime('%Y-%m-%d')]

    for i in range(len(series1) - 1):
        priorday = priorday - pd.Timedelta(days=days_per_lag)
        dates.insert(0, priorday.strftime('%Y-%m-%d'))

    plt.figure(figsize=(figsize))
    axis = plt.subplot()

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y1_label, color='#e74c3c')
    axis.plot(series1, marker='D', markersize=5, color='red')
    axis.tick_params(axis='y', labelcolor='#e74c3c')

    axis1 = axis.twinx()
    axis1.set_ylabel(y2_label, color='blue')
    axis1.plot(series2, marker='o', label=y2_label, markersize=5, color='blue')
    axis1.tick_params(axis='y', labelcolor='blue')

    weekdaydict = {k: v for k, v in zip(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], range(7))}

    maxy = series1.max()

    for i, date in enumerate(dates):
        if pd.Timestamp(date).weekday() == weekdaydict[weekday_mark]:
            axis.axvline(x=i, color='orange')
            axis.annotate(weekday_mark, xy=(i, maxy), xytext=(i + 0.1, maxy), color='orange')

    axis.set_xticks(list(range(len(dates)))[::xticklabel_spacing])
    axis.set_xticklabels(dates[::xticklabel_spacing])

    for tick in axis.get_xticklabels():
        tick.set_rotation(90)


def plot_compare_overtime2(series1, series2, steps, y1_label=None, y2_label=None, title='Time Series Comparison',
                           x_label='Step', figsize=(20, 8), xticklabel_spacing=1, samey=False, y_label='Values'):

    if hasattr(series1, 'values'):
        series1 = series1.values

    if y1_label is None:
        if hasattr(series1, 'name'):
            y1_label = series1.name

        else:
            y1_label = "Series 1"

    if hasattr(series2, 'values'):
        series2 = series2.values

    if y2_label is None:
        if hasattr(series2, 'name'):
            y2_label = series2.name

        else:
            y2_label = "Series 2"

    fig = plt.figure(figsize=(figsize))
    axis = plt.subplot()

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y1_label, color='#e74c3c')
    axis.plot(series1, marker='D', label=y1_label, markersize=5, color='red')
    axis.tick_params(axis='y', labelcolor='#e74c3c')

    if not samey:
        axis1 = axis.twinx()
        axis1.set_ylabel(y2_label, color='blue')
        axis1.plot(series2, marker='o', label=y2_label, markersize=5, color='blue')
        axis1.tick_params(axis='y', labelcolor='blue')
    else:
        axis.set_ylabel(y_label, color='black')
        axis.plot(series2, marker='o', label=y2_label, markersize=5, color='blue')
        axis.tick_params(axis='y', labelcolor='black')
        axis.legend()

    # weekdaydict = {k: v for k, v in zip(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], range(7))}
    # monthdict = {k: v for k, v in zip(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
    # 'November', 'December'], range(1, 13))}

    # maxy = series1.max()

    # if mark_by == 'week':
    #     markdict = weekdaydict
    # else:
    #     markdict = monthdict

    # if mark:
    #     for i, date in enumerate(timestamps):
    #         if mark_by == 'week':
    #             checkmark = date.weekday()
    #         else:
    #             checkmark = date.month

    #         if checkmark == markdict[mark_label]:
    #             axis.axvline(x=i, color='orange')
    #             axis.annotate(mark_label, xy=(i, maxy), xytext=(i + 0.1, maxy), color='orange')

    axis.set_xticks(list(range(len(steps)))[::xticklabel_spacing])
    try:
        axis.set_xticklabels([day.strftime('%Y-%m-%d') for day in steps][::xticklabel_spacing])
    except AttributeError:
        axis.set_xticklabels([str(step) for step in steps][::xticklabel_spacing])

    for tick in axis.get_xticklabels():
        tick.set_rotation(90)

    return fig


def plot_series_topbottom(df_over_time,
                          x_col,
                          y_top_col,
                          y_bottom_col,
                          x_label=None,
                          y_top_label=None,
                          y_bottom_label=None,
                          hue_col=None,
                          top_smoothing=1,
                          bottom_smoothing=1,
                          top_legend_loc='lower right',
                          bottom_legend_loc='lower right',
                          savefig_path=None,
                          savefig_kws=None,
                          figsize=(18, 10)):

    fig, axes = plt.subplots(2, 1, sharex=True)

    y_top = smooth(df_over_time[y_top_col], top_smoothing)
    y_bottom = smooth(df_over_time[y_bottom_col], bottom_smoothing)

    if hue_col is None:
        hue = None
    else:
        hue = df_over_time[hue_col]

    sns.lineplot(x=df_over_time[x_col],
                 y=y_top,
                 hue=hue,
                 ax=axes[0])

    sns.lineplot(x=df_over_time[x_col],
                 y=y_bottom,
                 hue=hue,
                 ax=axes[1])

    fig.set_size_inches(figsize[0], figsize[1])

    axes[0].set_ylabel(y_top_col if y_top_label is None else y_top_label)
    axes[0].legend(loc=top_legend_loc)

    axes[1].set_ylabel(y_bottom_col if y_bottom_label is None else y_bottom_label)
    axes[1].legend(loc=bottom_legend_loc)

    axes[1].set_xlabel(x_col if x_label is None else x_label)
    
    if savefig_path is not None:
        if savefig_kws is None:
            savefig_kws = {}
        plt.savefig(savefig_path, **savefig_kws)

    return fig, axes

