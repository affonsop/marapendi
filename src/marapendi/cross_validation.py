import numpy as np
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats

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
    checkpoint_callback : callable or None
        Function called after each complexity level.
        Signature: checkpoint_callback(cross_validation_results)
    """

    if estimate_kwargs is None:
        estimate_kwargs = dict(
            t=0,
            print_iterations=False,
            popsize=10,
            ftol=1e-5,
            penalty_threshold=0,
            vectorized=False,
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
            print(f"Skipping n_parameters={n_parameters} (already computed)")
            continue

        print(f"\nRunning CV for {n_parameters} parameters")

        selected_parameters = [
            base_model.unknown_p_list[idx]
            for idx in parameter_indices[:n_parameters]
        ]

        fold_results = []

        for fold_index, test_case in enumerate(case_list):

            training_cases = [
                case for case in case_list
                if case != test_case
            ]

            fold_simulator = build_model(training_cases, base_model.p)
            fold_simulator.set_unknown_params(selected_parameters)

            start_time = time.time()

            optimization_result, estimated_parameters = (
                fold_simulator.estimate(
                    get_exp_data(training_cases),
                    **estimate_kwargs
                )
            )

            elapsed_time = time.time() - start_time

            fold_simulator.p.update({
                name: value
                for name, value in zip(
                    fold_simulator.p_i_name,
                    estimated_parameters
                )
            })

            rmse = np.sqrt(optimization_result.fun) * rmse_scale

            fold_results.append({
                "n_parameters": n_parameters,
                "test_case": test_case,
                "training_cases": training_cases,
                "model_parameters": fold_simulator.p,
                "optimization_result": optimization_result,
                "estimated_parameters": dict(
                    zip(
                        fold_simulator.p_i_name,
                        estimated_parameters
                    )
                ),
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
        print(f"No existing CV results found at: {filepath}")
        return None

    df = pd.read_csv(filepath)

    cross_validation_results = []

    for n_parameters in sorted(df["n_parameters"].unique()):

        df_n = df[df["n_parameters"] == n_parameters]
        fold_results = []

        for _, row in df_n.iterrows():

            test_case = row["test_case"]

            # Rebuild parameter dictionary
            parameters = base_model.p.copy()

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

def plot_rmse_vs_complexity(
    cv_results,
    case_list,
    parameter_indices,
    base_model,
    model_builder,
    simulate_callback,
    quantity_name="Cell voltage",
    quantity_unit="mV",
    quantity_multiplier=1000,
    figsize=(10,4),
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
            cell_model = model_builder(model_parameters)

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
    test_std = test_df.groupby("n_parameters")["rmse"].std()
    train_mean = train_df.groupby("n_parameters")["rmse"].mean()

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
        test_mean.values,
        color="dimgray",
        label="Average RMSE - test"
    )

    ax.plot(
        complexity_levels,
        train_mean.values,
        color="dimgray",
        linestyle="--",
        label="Average RMSE - train"
    )

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

    ax.set_ylabel(f"{quantity_name}\nRMSE ({quantity_unit})")
    ax.set_ylim(bottom=0)
    ax.grid()

    # X-axis spans ALL unknown parameters
    ax.set_xticks(full_parameter_range)
    ax.set_xlim(0.5, len(parameter_indices) + 0.5)

    param_labels = [
        base_model.unknown_p_list[idx][-1]
        for idx in parameter_indices
    ]

    ax.set_xticklabels(param_labels, rotation=45)

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

                param_name = base_model.unknown_p_list[param_idx][0]
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

        param_info = base_model.unknown_p_list[param_idx]
        param_name = param_info[0]
        is_log = not param_info[2]

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

        param_info = base_model.unknown_p_list[param_idx]
        param_name = param_info[0]
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

        param_info = base_model.unknown_p_list[param_idx]
        param_name = param_info[0]
        scale = parameter_units[param_name][1]

        axis.set_title(f'{param_position + 1} – {param_info[-1]}', fontsize=9)
        axis.set_ylabel(f'({parameter_units[param_name][2]})', fontsize=8)

        axis.set_xlim((1, len(parameter_indices)))
        axis.set_ylim([bound / scale for bound in param_info[1]])

        if not param_info[2]:
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
    case_table,
    condition_color,
    simulate_callback,
    model_builder,
    quantity_name="Cell voltage",
    quantity_symbol=r"$V_{cell}$",
    quantity_unit="V",
    x_label=r"$i$ (A/cm$^2$)",
    save_path=None,
    dpi=300
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
        sharey='row'
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
        cell_model = model_builder(fold["model_parameters"])

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

            ax[k, i].plot(
                x_exp,
                y_exp,
                's' + condition_color[case],
                markersize=4
            )

            if i == k:
                ax[k, i].set_facecolor('#f0f0f0')

            # Column titles
            if k == 0:
                ax[0, i].set_title(
                    f'Condition {i+1}\n'
                    r'$T_{cell}$: '
                    f'{case_table.loc[case, "temperature"]:.0f} °C\n'
                    f'{case_table.loc[case, "koh_concentration"]:.1f} M KOH',
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
            plt.Line2D([0], [0], color='black'),
            plt.Line2D([0], [0], marker='s', linestyle='None', color='black')
        ],
        labels=['Sim.', 'Exp.'],
        loc='upper left'
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