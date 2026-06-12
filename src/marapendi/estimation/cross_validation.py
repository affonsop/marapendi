import numpy as np
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from marapendi.dynamic.simulation.estimation import ParameterEstimation

def run_cross_validation(
    base_model,
    parameter_indices,
    case_list,
    build_model,
    get_exp_data,
    min_parameters=1,
    cross_validation_results=None,
    estimate_kwargs=None,
    rmse_scale=1e3,
    checkpoint_callback=None,
):
    """
    Leave-one-out cross-validation with resume capability
    and automatic checkpoint saving.

    Parameters
    ----------
    base_model : ParameterEstimation
        Configured estimator with .unknown_parameters and .params.
    build_model : callable
        build_model(case_list, params) -> model_fn where
        model_fn(params_dict) -> np.ndarray.
    checkpoint_callback : callable or None
        Function called after each complexity level.
        Signature: checkpoint_callback(cross_validation_results)
    """

    if estimate_kwargs is None:
        estimate_kwargs = dict(
            print_iterations=False,
            popsize=10,
            ftol=1e-5,
            penalty_threshold=0,
            rtol=0.1,
            maxiter=120,
        )

    # ---------------------------------------------
    # Initialize / Resume
    # ---------------------------------------------
    if cross_validation_results is None:
        cross_validation_results = []
        completed_parameter_counts = []
    else:
        completed_parameter_counts = [
            folds[0]["n_parameters"]
            for folds in cross_validation_results
        ]

    full_parameter_range = range(
        min_parameters,
        len(parameter_indices) + 1
    )

    # ---------------------------------------------
    # Main Loop
    # ---------------------------------------------
    for n_parameters in full_parameter_range:

        if n_parameters in completed_parameter_counts:
            continue

        selected_parameters = [
            base_model.unknown_parameters[idx]
            for idx in parameter_indices[:n_parameters]
        ]

        fold_results = []

        for fold_index, test_case in enumerate(case_list):

            training_cases = [
                case for case in case_list
                if case != test_case
            ]

            model_fn = build_model(training_cases, base_model.params)
            fold_pe = ParameterEstimation(
                model_fn, base_model.params, selected_parameters
            )

            print(
                f"[CV] n_parameters={n_parameters}, "
                f"test_case={test_case!r}"
            )

            start_time = time.time()

            optimization_result, params_estimated = fold_pe.estimate(
                get_exp_data(training_cases),
                **estimate_kwargs
            )

            elapsed_time = time.time() - start_time

            rmse = np.sqrt(optimization_result.fun) * rmse_scale

            fold_results.append({
                "n_parameters": n_parameters,
                "test_case": test_case,
                "training_cases": training_cases,
                "model_parameters": params_estimated,
                "optimization_result": optimization_result,
                "estimated_parameters": {
                    up.key: params_estimated[up.key]
                    for up in selected_parameters
                },
                "rmse": rmse,
                "elapsed_time": elapsed_time,
            })

        cross_validation_results.append(fold_results)

        # -----------------------------------------
        # Automatic checkpoint
        # -----------------------------------------
        if checkpoint_callback is not None:
            checkpoint_callback(cross_validation_results)

    # Ensure sorted by complexity
    cross_validation_results.sort(
        key=lambda folds: folds[0]["n_parameters"]
    )

    return cross_validation_results

def save_cross_validation_results(
    cross_validation_results,
    filename,
    output_dir=".",
):
    """
    Save cross-validation results to CSV.
    """

    filepath = os.path.join(output_dir, f"results_{filename}.csv")

    rows = []

    for fold_results in cross_validation_results:

        for fold in fold_results:

            row = {
                "n_parameters": fold["n_parameters"],
                "computation_time": fold["elapsed_time"],
                "objective_value": (
                    fold["optimization_result"].fun
                    if hasattr(fold["optimization_result"], "fun")
                    else fold["optimization_result"]
                ),
                "test_case": fold["test_case"],
            }

            # Store estimated parameters
            for param_name, param_value in fold["estimated_parameters"].items():
                row[param_name] = param_value

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)

    return filepath

def load_cross_validation_results(
    filepath,
    base_model,
):
    """
    Load cross-validation results from CSV and rebuild fold structure.

    Returns
    -------
    cross_validation_results : list or None
        Returns None if file does not exist.
    """

    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)

    cross_validation_results = []

    for n_parameters in sorted(df["n_parameters"].unique()):

        df_n = df[df["n_parameters"] == n_parameters]
        fold_results = []

        for _, row in df_n.iterrows():

            test_case = row["test_case"]

            # Rebuild parameter dictionary
            parameters = base_model.params.copy()

            for column in df.columns[4:]:  # skip metadata columns
                value = row[column]
                if not np.isnan(value):
                    parameters[column] = value

            fold_results.append({
                "n_parameters": n_parameters,
                "test_case": test_case,
                "training_cases": None,
                "model_parameters": parameters,
                "optimization_result": row["objective_value"],
                "estimated_parameters": dict(
                    zip(df.columns[4:], row[df.columns[4:]].values)
                ),
                "rmse": np.sqrt(row["objective_value"]),
                "elapsed_time": row["computation_time"],
            })

        cross_validation_results.append(fold_results)

    return cross_validation_results

def plot_rmse_vs_complexity_extrapolation(
    cv_results,
    training_test_case,
    extrapolation_cases,
    parameter_indices,
    base_model,
    model_builder,
    simulate_callback,
    ylabel="RMSE",
    quantity_multiplier=1000,
    condition_color={},
    use_median=True,
    figsize=(10, 4),
    xrotation=45,
    save_path=None,
    dpi=300,
):
    """
    Plot RMSE vs model complexity for conditions that are neither in the
    training nor the testing dataset, for a given fixed test case.

    Parameters
    ----------
    cv_results : dict
        Cross-validation results.
    training_test_case : int or str
        The test case (left-out condition) used to select the fold.
    extrapolation_cases : list
        Conditions not included in training or testing, to evaluate on.
    """

    complexity_levels = get_complexity_levels(cv_results)

    rmse_values = []
    case_column = []
    complexity_column = []

    # ------------------------------------------------------------
    # Collect RMSE values
    # ------------------------------------------------------------
    for n_parameters in complexity_levels:

        folds = get_folds_for_complexity(cv_results, n_parameters)

        # Select only the fold corresponding to the fixed test case
        matching_folds = [f for f in folds if f["test_case"] == training_test_case]

        for fold in matching_folds:

            model_parameters = fold["model_parameters"]
            cell_model = model_builder(model_parameters, training_test_case)

            for case in extrapolation_cases:

                x_sim, y_sim, x_exp, y_exp = simulate_callback(case, cell_model)

                residuals = y_exp - y_sim
                rmse = np.sqrt(np.dot(residuals, residuals) / len(residuals)) * quantity_multiplier

                rmse_values.append(rmse)
                case_column.append(case)
                complexity_column.append(n_parameters)

    rmse_df = pd.DataFrame({
        "rmse": rmse_values,
        "case": case_column,
        "n_parameters": complexity_column,
    })

    # ------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------
    agg_fn = "median" if use_median else "mean"
    overall_trend = rmse_df.groupby("n_parameters")["rmse"].agg(agg_fn)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if len(condition_color) == 0: 
        condition_color = {
            case: f"C{k}" for k, case in enumerate(extrapolation_cases)
        }

    handles, labels = [], []

    # Individual RMSE per extrapolation condition
    for k, case in enumerate(extrapolation_cases):

        case_df = rmse_df[rmse_df["case"] == case]

        line, = ax.semilogy(
            case_df["n_parameters"],
            case_df["rmse"],
            "s" + condition_color[case],
            markersize=5,
            alpha=1,
        )

        handles.append(line)
        labels.append(f"{case}")

    # Overall trend
    ax.semilogy(
        complexity_levels,
        overall_trend.values,
        color="dimgray",
        label=("Median" if use_median else "Average") + " RMSE - extrapolation",
    )

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------
    full_parameter_range = range(1, len(parameter_indices) + 1)

    ax.set_ylabel(ylabel)
    ax.grid()

    ax.set_xticks(full_parameter_range)
    ax.set_xlim(0.5, len(parameter_indices) + 0.5)

    param_labels = [
        base_model.unknown_parameters[idx].label
        for idx in parameter_indices
    ]
    ax.set_xticklabels(param_labels, rotation=xrotation)

    # Top x-axis
    ax_top = ax.twiny()
    ax_top.set_xticks(full_parameter_range)
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xlabel("Number of selected parameters")

    # Legends
    leg1 = ax.legend(loc=0)
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(0.99, 0.9),
        fontsize=9,
        title="Condition",
    )
    ax.add_artist(leg1)

    # ax.set_title(f"Extrapolation RMSE (fold: test case = {training_test_case})")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, rmse_df

def plot_rmse_vs_complexity(
    cv_results,
    case_list,
    parameter_indices,
    base_model,
    model_builder,
    simulate_callback,
    ylabel="RMSE",
    quantity_multiplier=1000,
    use_median = True, 
    plot_one_sigma_interval=False,
    figsize=(10,4),
    xrotation=45,
    save_path=None,
    dpi=300,
):
    """
    Plot train and test RMSE vs model complexity.
    Compatible with cv_results format.
    """

    complexity_levels = get_complexity_levels(cv_results)

    rmse_values = []
    case_column = []
    test_case_column = []
    complexity_column = []

    # ------------------------------------------------------------
    # Collect RMSE values
    # ------------------------------------------------------------
    for n_parameters in complexity_levels:

        folds = get_folds_for_complexity(cv_results, n_parameters)

        for fold in folds:

            test_case = fold["test_case"]
            model_parameters = fold["model_parameters"]
            cell_model = model_builder(model_parameters, test_case)

            for case in case_list:

                x_sim, y_sim, x_exp, y_exp = simulate_callback(case, cell_model)

                residuals = y_exp - y_sim
                rmse = np.sqrt(np.dot(residuals, residuals) / len(residuals)) * quantity_multiplier

                rmse_values.append(rmse)
                case_column.append(case)
                test_case_column.append(test_case)
                complexity_column.append(n_parameters)

    rmse_df = pd.DataFrame({
        "rmse": rmse_values,
        "case": case_column,
        "test_case": test_case_column,
        "n_parameters": complexity_column,
    })

    # ------------------------------------------------------------
    # Separate train and test
    # ------------------------------------------------------------
    test_df = rmse_df[rmse_df["case"] == rmse_df["test_case"]]
    train_df = rmse_df[rmse_df["case"] != rmse_df["test_case"]]

    test_mean = test_df.groupby("n_parameters")["rmse"].mean()
    test_median = test_df.groupby("n_parameters")["rmse"].median()
    test_std = test_df.groupby("n_parameters")["rmse"].std()
    train_mean = train_df.groupby("n_parameters")["rmse"].mean()
    train_median = train_df.groupby("n_parameters")["rmse"].median()

    # ------------------------------------------------------------
    # Optimal number of parameters based on 1 std deviaiton rule 
    # ------------------------------------------------------------
    # See https://esl.hohoweiya.xyz/book/The%20Elements%20of%20Statistical%20Learning.pdf
    
    optimal_n = test_mean.idxmin()

    best = test_mean.idxmin()
    threshold = test_mean[best] + test_std[best]

    for n in test_mean.index:
        if test_mean[n] <= threshold:
            optimal_n = n
            break

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)

    condition_color = {
        case: f"C{k}" for k, case in enumerate(case_list)
    }

    handles, labels = [], []

    # Individual test RMSE per condition
    for k, case in enumerate(case_list):

        case_df = test_df[test_df["case"] == case]

        line, = ax.plot(
            case_df["n_parameters"],
            case_df["rmse"],
            "s" + condition_color[case],
            markersize=5,
            alpha=1,
        )

        handles.append(line)
        labels.append(f"{k+1}")

    # Average test/train curves
    ax.plot(
        complexity_levels,
        (test_median if use_median else test_mean).values,
        color="dimgray",
        label=("Median" if use_median else "Average") + " RMSE - test"
    )
 
    ax.plot(
        complexity_levels,
        (train_median if use_median else train_mean).values,
        color="dimgray",
        linestyle="--",
        label=("Median" if use_median else "Average") + " RMSE - train"
    )
    if plot_one_sigma_interval: 
        ax.fill_between(
            complexity_levels,
            test_mean.values - test_std.values,
            test_mean.values + test_std.values,
            color="dimgray",
            alpha=0.3,
            label="$\pm$ 1$\sigma$ RMSE - test"
        )


    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------
    full_parameter_range = range(1, len(parameter_indices) + 1)

    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid()

    # X-axis spans ALL unknown parameters
    ax.set_xticks(full_parameter_range)
    ax.set_xlim(0.5, len(parameter_indices) + 0.5)

    param_labels = [
        base_model.unknown_parameters[idx].label
        for idx in parameter_indices
    ]

    ax.set_xticklabels(param_labels, rotation=xrotation)

    # Top x-axis: number of selected parameters
    ax_top = ax.twiny()
    ax_top.set_xticks(full_parameter_range)
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xlabel("Number of selected parameters")

    # Legends
    leg1 = ax.legend(loc=0)
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(0.99, 0.9),
        fontsize=9,
        title="Condition"
    )

    ax.add_artist(leg1)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

   

    return fig, ax, rmse_df, optimal_n

def plot_parameter_vs_complexity(
    cv_results,
    parameter_indices,
    base_model,
    parameter_units,
    condition_color,
    initial_parameters,
    plot_ci=False,
    n_cols=6,
    figsize=(14, 12),
    confidence=0.95,
    save_path=None,
    dpi=300,
):
    """
    Plot evolution of parameters across complexity levels.

    Parameters not yet estimated remain empty (NaN).
    """

    parameter_counts = sorted([
        folds[0]["n_parameters"] for folds in cv_results
    ])

    n_display = len(parameter_indices)
    n_rows = int(np.ceil(n_display / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    n_cases = len(cv_results[0])
    n_steps = len(parameter_counts)

    all_values = np.full((n_cases, n_steps, n_display), np.nan)

    # ------------------------------------------------------------
    # 1) Collect scaled values (only if estimated)
    # ------------------------------------------------------------
    for case_index in range(n_cases):

        test_case = cv_results[0][case_index]["test_case"]

        for step_index, folds in enumerate(cv_results):

            fold = folds[case_index]
            estimated_params = fold["estimated_parameters"]

            for param_position, param_idx in enumerate(parameter_indices):

                up = base_model.unknown_parameters[param_idx]
                param_name = up.key
                scale = parameter_units[param_name][1]

                # Parameter estimated only if within complexity
                if param_name in estimated_params:
                    value = estimated_params[param_name] / scale
                    all_values[case_index, step_index, param_position] = value

        # Plot markers
        for param_position in range(n_display):

            row = param_position // n_cols
            col = param_position % n_cols
            axis = axes[row, col]

            axis.plot(
                parameter_counts,
                all_values[case_index, :, param_position],
                marker='.',
                linestyle='None',
                color=condition_color[test_case],
                alpha=1,
            )

    # ------------------------------------------------------------
    # 2) Mean + CI (NaN-aware + log-aware)
    # ------------------------------------------------------------
    mean_vals = np.full((n_steps, n_display), np.nan)
    lower_ci = np.full_like(mean_vals, np.nan)
    upper_ci = np.full_like(mean_vals, np.nan)

    for param_position, param_idx in enumerate(parameter_indices):

        up = base_model.unknown_parameters[param_idx]
        param_name = up.key
        is_log = up.log_scale

        values = all_values[:, :, param_position]

        for step_index in range(n_steps):

            valid = ~np.isnan(values[:, step_index])
            valid_values = values[valid, step_index]

            if len(valid_values) < 2:
                continue  # not enough data for CI

            if is_log:
                log_values = np.log10(valid_values)
                mean_log = np.mean(log_values)
                std_log = np.std(log_values, ddof=1)

                t_multiplier = stats.t.ppf(
                    (1 + confidence) / 2.0,
                    df=len(valid_values) - 1
                )

                ci_half = t_multiplier * std_log / np.sqrt(len(valid_values))

                mean_vals[step_index, param_position] = 10 ** mean_log
                lower_ci[step_index, param_position] = 10 ** (mean_log - ci_half)
                upper_ci[step_index, param_position] = 10 ** (mean_log + ci_half)

            else:
                mean_lin = np.mean(valid_values)
                std_lin = np.std(valid_values, ddof=1)

                t_multiplier = stats.t.ppf(
                    (1 + confidence) / 2.0,
                    df=len(valid_values) - 1
                )

                ci_half = t_multiplier * std_lin / np.sqrt(len(valid_values))

                mean_vals[step_index, param_position] = mean_lin
                lower_ci[step_index, param_position] = mean_lin - ci_half
                upper_ci[step_index, param_position] = mean_lin + ci_half

    # ------------------------------------------------------------
    # 3) Plot mean + CI + reference
    # ------------------------------------------------------------
    for param_position, param_idx in enumerate(parameter_indices):

        row = param_position // n_cols
        col = param_position % n_cols
        axis = axes[row, col]

        up = base_model.unknown_parameters[param_idx]
        param_name = up.key
        scale = parameter_units[param_name][1]

        axis.plot(
            parameter_counts,
            mean_vals[:, param_position],
            color='dimgray',
            linewidth=2,
            label='Mean'
        )

        if plot_ci:
            axis.fill_between(
                parameter_counts,
                lower_ci[:, param_position],
                upper_ci[:, param_position],
                color='dimgray',
                alpha=0.3,
                label=f'{int(confidence*100)}% CI'
            )

        # Reference line (always visible)
        ref = 0.99 * initial_parameters[param_name] / scale
        axis.plot(
            [1, len(parameter_indices)],
            np.ones(2) * ref,
            linestyle='--',
            color='dimgray',
            linewidth=1.1,
            label='Reference'
        )

    # ------------------------------------------------------------
    # 4) Formatting (unchanged logic)
    # ------------------------------------------------------------
    for param_position, param_idx in enumerate(parameter_indices):

        row = param_position // n_cols
        col = param_position % n_cols
        axis = axes[row, col]

        up = base_model.unknown_parameters[param_idx]
        param_name = up.key
        scale = parameter_units[param_name][1]

        axis.set_title(f'{param_position + 1} – {up.label}', fontsize=9)
        axis.set_ylabel(f'({parameter_units[param_name][2]})', fontsize=8)

        axis.set_xlim((1, len(parameter_indices)))
        axis.set_ylim([up.lower / scale, up.upper / scale])

        if up.log_scale:
            axis.set_yscale('log')

        axis.grid(True)

    for col in range(n_cols):
        axes[-1, col].set_xlabel('Number of selected\nparameters')

    handles, labels = axes[0, -1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc='upper center', ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, axes, mean_vals, lower_ci, upper_ci

def plot_cross_validation_curves(
    n_parameters,
    cv_results,
    case_list,
    condition_color,
    simulate_callback,
    model_builder,
    quantity_name="Cell voltage",
    quantity_symbol=r"$V_{cell}$",
    quantity_unit="V",
    case_titles = None,
    x_label=r"$i$ (A/cm$^2$)",
    save_path=None,
    dpi=300,
    uncertainty=0.1
):
    """
    Cross-validation curve plotting compatible with new cv_results format.
    """

    # ------------------------------------------------------------
    # Select correct complexity level
    # ------------------------------------------------------------
    folds = get_folds_for_complexity(cv_results, n_parameters)

    n_cases = len(case_list)

    fig, ax = plt.subplots(
        figsize=(12, 10),
        nrows=n_cases,
        ncols=n_cases,
        sharex=True,
        sharey=True
    )

    fig.set_tight_layout(True)

    rmse_list = []
    maxerr_list = []

    # ------------------------------------------------------------
    # Loop over folds (left-out condition)
    # ------------------------------------------------------------
    for k, fold in enumerate(folds):

        case_left_out = fold["test_case"]

        # Rebuild model from stored parameters
        cell_model = model_builder(fold["model_parameters"], case_left_out)

        for i, case in enumerate(case_list):

            # ---- Simulation ----
            x_sim, y_sim, x_exp, y_exp = simulate_callback(case, cell_model)

            # ---- Errors ----
            y_sim_interp = np.interp(x_exp, x_sim, y_sim)

            rmse = np.sqrt(np.mean((y_sim_interp - y_exp) ** 2))
            maxerr = np.max(np.abs(y_sim_interp - y_exp))

            if case == case_left_out:
                rmse_list.append(rmse)
                maxerr_list.append(maxerr)

            # ---- Plot ----
            ax[k, i].plot(
                x_sim,
                y_sim,
                '-' + condition_color[case]
            )
            if uncertainty:  
                ax[k, i].fill_between(
                    x_sim, 
                    (1-uncertainty)*y_sim,(1+uncertainty)*y_sim, 
                    color=condition_color[case],
                    alpha=0.3)

            ax[k, i].plot(
                x_exp,
                y_exp,
                's' + condition_color[case],
                markersize=3.5, 
                alpha=0.5
            )

            if i == k:
                ax[k, i].set_facecolor('#f0f0f0')

            # Column titles
            if k == 0:
                ax[0, i].set_title(
                    case_titles[case],
                    fontsize=9
                )

            # Row labels
            if i == 0:
                ax[k, 0].set_ylabel(
                    f'Cond. {k+1}\nleft out\n'
                    f'{quantity_symbol} ({quantity_unit})'
                )

            if k == n_cases - 1:
                ax[k, i].set_xlabel(x_label)

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color='C0'),
            plt.Line2D([0], [0], marker='s', linestyle='None', color='C0'),
            Patch(facecolor='C0', alpha=0.3)
        ],
        labels=['Sim.', 'Exp.', f'± {uncertainty*100:.0f} %'],
        loc='upper left', 
        bbox_to_anchor=(0, 1.05)
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)

    # ------------------------------------------------------------
    # LaTeX-ready metrics
    # ------------------------------------------------------------
    rmse_string = (
        rf"{quantity_name} RMSE & "
        + " & ".join([f"{v:.2f}" for v in rmse_list])
        + r" \\"
    )

    maxerr_string = (
        rf"{quantity_name} max. error & "
        + " & ".join([f"{v:.2f}" for v in maxerr_list])
        + r" \\"
    )

    return fig, ax, rmse_list, maxerr_list, rmse_string, maxerr_string

def get_complexity_levels(cv_results):
    """
    Return sorted list of available model complexities
    (number of estimated parameters).
    """
    if cv_results is None or len(cv_results) == 0:
        return []

    return sorted([
        folds[0]["n_parameters"]
        for folds in cv_results
    ])

def get_folds_for_complexity(cv_results, n_parameters):
    """
    Return fold list for a given number of parameters.
    Raises ValueError if not available.
    """
    for folds in cv_results:
        if folds[0]["n_parameters"] == n_parameters:
            return folds

    available = get_complexity_levels(cv_results)
    raise ValueError(
        f"n_parameters={n_parameters} not available. "
        f"Available levels: {available}"
    )

class CrossValidation:
    """
    Simplified, stateful interface around the free functions in this module.

    Wraps :func:`run_cross_validation`, the save/load helpers, and the
    plotting functions so a notebook only needs to provide a
    ``build_model(case_list) -> model_fn`` callable and an experimental-data
    getter — no manual adapters or repeated argument lists.

    Parameters
    ----------
    base_model : ParameterEstimation
        Configured estimator with ``.unknown_parameters`` and ``.params``.
    parameter_indices : sequence of int
        Order in which parameters are added across complexity levels
        (e.g. from ``base_model.plot_parameter_ranking()``).
    case_list : list
        All experimental cases/conditions.
    build_model : callable
        ``build_model(case_list) -> model_fn`` where
        ``model_fn(params_dict) -> np.ndarray``.
    get_exp_data : callable
        ``get_exp_data(case_list) -> np.ndarray``.
    model_builder, simulate_callback : callable, optional
        Forwarded to the plotting helpers — see
        :func:`plot_rmse_vs_complexity` and :func:`plot_cross_validation_curves`.
    filename : str, optional
        Base name used for checkpoint/result files
        (``results_{filename}_cv.csv``) in ``output_dir``.
    output_dir : str
        Directory for checkpoint/result files (default ``"results"``).
    estimate_kwargs : dict, optional
        Default kwargs forwarded to ``ParameterEstimation.estimate``.

    Examples
    --------
    ::

        cv = CrossValidation(
            base_model=pe,
            parameter_indices=P,
            case_list=full_case_list,
            build_model=build_model,
            get_exp_data=get_cases_exp_data,
            model_builder=model_builder,
            simulate_callback=simulate_callback,
            filename=filename,
        )
        cv_results = cv.run()
        fig, ax, rmse_df, optimal_n = cv.plot_rmse_vs_complexity()
    """

    def __init__(
        self,
        base_model,
        parameter_indices,
        case_list,
        build_model,
        get_exp_data,
        model_builder=None,
        simulate_callback=None,
        filename=None,
        output_dir="results",
        estimate_kwargs=None,
        rmse_scale=1e3,
        min_parameters=1,
        condition_color=None,
    ):
        self.base_model = base_model
        self.parameter_indices = parameter_indices
        self.case_list = case_list
        self.build_model = build_model
        self.get_exp_data = get_exp_data
        self.model_builder = model_builder
        self.simulate_callback = simulate_callback
        self.filename = filename
        self.output_dir = output_dir
        self.estimate_kwargs = estimate_kwargs
        self.rmse_scale = rmse_scale
        self.min_parameters = min_parameters
        self.condition_color = condition_color or {
            case: f"C{k}" for k, case in enumerate(case_list)
        }

        self.results = None
        self.rmse_df = None
        self.optimal_n = None

    # ------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------

    @property
    def filepath(self):
        if self.filename is None:
            return None
        return os.path.join(self.output_dir, f"results_{self.filename}_cv.csv")

    def load(self):
        """Load previously saved results into ``self.results``."""
        self.results = load_cross_validation_results(
            filepath=self.filepath,
            base_model=self.base_model,
        )
        return self.results

    def save(self):
        """Save ``self.results`` to ``self.filepath``."""
        return save_cross_validation_results(
            cross_validation_results=self.results,
            filename=f"{self.filename}_cv",
            output_dir=self.output_dir,
        )

    def run(self, resume=True, checkpoint=True, **estimate_kwargs):
        """Run leave-one-out cross-validation.

        Parameters
        ----------
        resume : bool
            Load and skip complexity levels already saved at ``self.filepath``.
        checkpoint : bool
            Save results after each complexity level.
        **estimate_kwargs
            Override entries of ``self.estimate_kwargs``.
        """
        cross_validation_results = self.load() if resume and self.filepath else None

        def _build_model(case_list, _params):
            return self.build_model(case_list)

        def _checkpoint(results):
            self.results = results
            if checkpoint and self.filename:
                self.save()

        self.results = run_cross_validation(
            base_model=self.base_model,
            parameter_indices=self.parameter_indices,
            case_list=self.case_list,
            build_model=_build_model,
            get_exp_data=self.get_exp_data,
            min_parameters=self.min_parameters,
            cross_validation_results=cross_validation_results,
            estimate_kwargs={**(self.estimate_kwargs or {}), **estimate_kwargs},
            rmse_scale=self.rmse_scale,
            checkpoint_callback=_checkpoint,
        )
        return self.results

    # ------------------------------------------------------------
    # Complexity-level helpers
    # ------------------------------------------------------------

    def get_complexity_levels(self):
        return get_complexity_levels(self.results)

    def get_folds_for_complexity(self, n_parameters):
        return get_folds_for_complexity(self.results, n_parameters)

    # ------------------------------------------------------------
    # Plots / tables
    # ------------------------------------------------------------

    def plot_rmse_vs_complexity(self, **kwargs):
        fig, ax, rmse_df, optimal_n = plot_rmse_vs_complexity(
            cv_results=self.results,
            case_list=self.case_list,
            parameter_indices=self.parameter_indices,
            base_model=self.base_model,
            model_builder=self.model_builder,
            simulate_callback=self.simulate_callback,
            **kwargs,
        )
        self.rmse_df = rmse_df
        self.optimal_n = optimal_n
        return fig, ax, rmse_df, optimal_n

    def plot_rmse_vs_complexity_extrapolation(
        self, training_test_case, extrapolation_cases, **kwargs
    ):
        return plot_rmse_vs_complexity_extrapolation(
            cv_results=self.results,
            training_test_case=training_test_case,
            extrapolation_cases=extrapolation_cases,
            parameter_indices=self.parameter_indices,
            base_model=self.base_model,
            model_builder=self.model_builder,
            simulate_callback=self.simulate_callback,
            condition_color=kwargs.pop('condition_color', self.condition_color),
            **kwargs,
        )

    def plot_parameter_vs_complexity(self, parameter_units, initial_parameters, **kwargs):
        return plot_parameter_vs_complexity(
            cv_results=self.results,
            parameter_indices=self.parameter_indices,
            base_model=self.base_model,
            parameter_units=parameter_units,
            condition_color=kwargs.pop('condition_color', self.condition_color),
            initial_parameters=initial_parameters,
            **kwargs,
        )

    def plot_cross_validation_curves(self, n_parameters=None, **kwargs):
        return plot_cross_validation_curves(
            n_parameters=n_parameters if n_parameters is not None else self.optimal_n,
            cv_results=self.results,
            case_list=self.case_list,
            condition_color=kwargs.pop('condition_color', self.condition_color),
            simulate_callback=self.simulate_callback,
            model_builder=self.model_builder,
            **kwargs,
        )

    def rmse_complexity_table(self, filename=None):
        return rmse_complexity_table(self.rmse_df, filename=filename)


def rmse_complexity_table(rmse_vs_complexity_df, filename=None):
    
    group = (rmse_vs_complexity_df[rmse_vs_complexity_df.case == rmse_vs_complexity_df.test_case]
        .groupby(['n_parameters', 'test_case']).mean()
        .reset_index()
        .groupby('n_parameters'))

    per_case = (rmse_vs_complexity_df[rmse_vs_complexity_df.case == rmse_vs_complexity_df.test_case]
        .groupby(['n_parameters', 'test_case']).mean()
        .reset_index()
        .pivot(index='n_parameters', columns='test_case', values='rmse'))

    stats_df = (group.agg(['min', 'max', 'median', 'mean'])
        .drop(columns=['case', 'test_case'], level=0))
    stats_df.columns = ['_'.join(col) for col in stats_df.columns]

    stats_df = stats_df.join(per_case)

    # reorder
    stat_cols = [c for c in stats_df.columns if isinstance(c, str) and '_' in c]
    case_cols = [c for c in stats_df.columns if c not in stat_cols]
    stats_df = stats_df[case_cols + stat_cols]

    # rename
    stats_df = stats_df.rename(columns={
        'rmse_min': 'Min.',
        'rmse_max': 'Max.',
        'rmse_median': 'Median',
        'rmse_mean': 'Mean',
    })

    stats_df.rename(
        columns={c: f'{c:.0f}' for c in case_cols}
    )
    # build latex
    n_cases = len(case_cols)
    latex = (stats_df.to_latex(
            float_format="%.1f",
            na_rep="-",
            caption="RMSE vs complexity",
            label="tab:rmse_complexity",
            position="h"
        ).replace(
            r'\toprule',
            r'\toprule' + '\n' +
            rf' & \multicolumn{{{n_cases}}}{{c}}{{Cases}} \\' + '\n' +
            rf'\cmidrule(lr){{2-{n_cases + 1}}}'
        )
    )

    if filename:
        with open(filename, 'w') as f:
            f.write(latex)

    return stats_df, latex

