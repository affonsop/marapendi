import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .polarization_curve_calibration import collect_rmse_df, optimal_n_1se

def plot_global_sensitivity(model, fig=None, ax=None, cmap='viridis', color='C0',
                            xlabel_angle=45, xlabel_ha='center', figsize=(4, 3), parameter_order=[]):
    """Scatter all Sobol sensitivity samples and overlay the median sensitivity per parameter."""
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    n_params = len(model.unknown_p_list)
    xi = np.arange(n_params) if len(parameter_order) == 0 else parameter_order

    for position, parameter in enumerate(xi):
        ax.plot(position * np.ones_like(model.norm_s_i[:, parameter]), model.norm_s_i[:, parameter], '.' + color, alpha=0.2)

    ax.semilogy(np.arange(n_params), model.S_med_i[xi], '-s' + color)
    ax.set_xticks(np.arange(n_params), labels=np.array(model.p_i_symbol)[xi])
    ax.set_xlim(-0.5, n_params - 0.5)

    if xlabel_angle > 0:
        plt.setp(ax.get_xticklabels(), rotation=xlabel_angle, ha=xlabel_ha)

    ax.set_ylabel("Median normalized sensitivity")
    fig.tight_layout()
    return fig, ax

def plot_colinearity_map(model, fig=None, ax=None, cmap='viridis', xlabel_angle=45,
                        figsize=(4, 3), write_text=True):
    """Heatmap of the median pairwise co-linearity index matrix (values in [0, 1])."""
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    xi = np.arange(len(model.unknown_p_list))
    im = ax.pcolormesh(xi, xi, model.cosPhi_med_ij, vmin=0, vmax=1, cmap=plt.colormaps[cmap])

    if write_text:
        for i in xi:
            for j in xi:
                ax.text(j, i, f'{model.cosPhi_med_ij[i, j]:.2f}', ha="center", va="center", color="w")

    fig.colorbar(im, ax=ax)
    ax.set_xticks(xi, labels=model.p_i_symbol)
    ax.set_yticks(xi, labels=model.p_i_symbol)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    if xlabel_angle > 0:
        plt.setp(ax.get_xticklabels(), rotation=xlabel_angle, ha="center", va='bottom',
                 rotation_mode="default")

    plt.gca().set_aspect('equal')
    fig.tight_layout()
    return fig, ax


def plot_parameter_ranking(model):
    n_unknown_p = len(model.unknown_p_list)
    model.get_smallest_hessian_eigenvalues()

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.semilogy(1 + np.arange(n_unknown_p), np.abs(model.min_eigvals), '-s')
    ax.set_xticks(1 + np.arange(n_unknown_p))
    ax.set_xticklabels([model.unknown_p_list[i].symbol for i in model.P], rotation=45)

    ax2 = ax.twiny()
    ax.set_xlim([0.5, n_unknown_p + 0.5])
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim([0.5, n_unknown_p + 0.5])
    ax2.set_xlabel('Number of selected parameters')

    ax.set_xlabel('Ranked parameters')
    ax.set_ylabel('Smallest Hessian\neigenvalue')
    ax.grid()
    fig.tight_layout()

    return fig, ax, ax2



def plot_rmse_vs_complexity(
    model,
    cv_results,
    variable='voltage',
    ylabel="RMSE",
    quantity_multiplier=1000,
    use_median=True,
    plot_one_sigma_interval=False,
    figsize=(10, 4),
    xrotation=45,
    save_path=None,
    dpi=300,
):
    """Plot train and test RMSE vs model complexity."""
    rmse_df = collect_rmse_df(model, cv_results, variable, quantity_multiplier)

    test_df = rmse_df[rmse_df["is_test"]]
    train_df = rmse_df[~rmse_df["is_test"]]

    test_grouped = test_df.groupby("n_params")["rmse"]
    train_grouped = train_df.groupby("n_params")["rmse"]
    test_mean, test_median, test_std = test_grouped.mean(), test_grouped.median(), test_grouped.std()
    train_mean, train_median = train_grouped.mean(), train_grouped.median()

    opt_n = optimal_n_1se(test_mean, test_std)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    handles, labels = [], []

    for k, case in enumerate(cv_results.fold_id.unique()):
        case_df = test_df[test_df["case"] == case]
        line, = ax.plot(case_df["n_params"], case_df["rmse"], f"sC{k}", markersize=5)
        handles.append(line)
        labels.append(f"{k+1}")

    agg_label = "Median" if use_median else "Average"
    agg_test = test_median if use_median else test_mean
    agg_train = train_median if use_median else train_mean
    complexity_levels = cv_results.n_params.unique()

    ax.plot(complexity_levels, agg_test.values, color="dimgray", label=f"{agg_label} RMSE - test")
    ax.plot(complexity_levels, agg_train.values, color="dimgray", linestyle="--", label=f"{agg_label} RMSE - train")

    if plot_one_sigma_interval:
        ax.fill_between(
            complexity_levels,
            test_mean.values - test_std.values,
            test_mean.values + test_std.values,
            color="dimgray", alpha=0.3, label=r"$\pm$ 1$\sigma$ RMSE - test",
        )

    param_labels = [model.unknown_p_list[idx].symbol for idx in model.P]
    full_range = range(1, len(model.P) + 1)

    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_xticks(full_range)
    ax.set_xlim(0.5, len(model.P) + 0.5)
    ax.set_xticklabels(param_labels, rotation=xrotation)
    ax.grid()

    ax_top = ax.twiny()
    ax_top.set_xticks(full_range)
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xlabel("Number of selected parameters")

    leg1 = ax.legend(loc=0)
    fig.legend(handles=handles, labels=labels, loc="upper left",
               bbox_to_anchor=(0.99, 0.9), fontsize=9, title="Condition")
    ax.add_artist(leg1)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, rmse_df, opt_n


def plot_parameter_vs_complexity(
    model,
    cv_results,
    n_cols=6,
    figsize=(14, 12),
    save_path=None,
    dpi=300,
):
    """Plot evolution of estimated parameters across complexity levels."""
    n_display = len(model.unknown_p_list)
    n_rows = int(np.ceil(n_display / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    estimated_parameter_keys = cv_results.columns[4:]
    n_params = cv_results.n_params.unique()

    for k, p_key in enumerate(estimated_parameter_keys):
        p = model.unknown_p_list[model.p_i_index[p_key]]
        axis = axes[k // n_cols, k % n_cols]

        for fold_id in cv_results.fold_id.unique():
            fold_results = cv_results[cv_results.fold_id == fold_id]
            axis.plot(n_params, fold_results[p_key] / p.factor, '.')

        axis.plot(
            n_params,
            cv_results[['n_params', 'fold_id', p_key]].groupby('n_params').mean()[p_key] / p.factor,
            'dimgray', linewidth=1.1, label='Mean'
        )

        ref = 0.99 * p.initial_guess / p.factor
        axis.plot([1, model.n_unkown_p], [ref, ref],
                  linestyle='--', color='dimgray', linewidth=1.1, label='Reference')

        axis.set_title(f'{k + 1} – {p.symbol}', fontsize=9)
        axis.set_ylabel(f'({p.units})', fontsize=8)
        axis.set_xlim((1, len(model.unknown_parameters)))
        axis.set_ylim([p.lower_bound / p.factor, p.upper_bound / p.factor])
        if not p.is_linear:
            axis.set_yscale('log')
        axis.grid(True)

    for col in range(n_cols):
        axes[-1, col].set_xlabel('Number of selected\nparameters')

    handles, labels = axes[0, -1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, axes


def plot_cross_validation_curves(
    model,
    cv_results,
    n_params, 
    variable='voltage',
    quantity_symbol=r"$V_{cell}$",
    quantity_unit="V",
    x_label=r"$i$ (A/cm$^2$)",
    save_path=None,
    dpi=300,
    uncertainty=0.1
):
    """
    Cross-validation curve plotting compatible with new cv_results format.
    """
    n_cases = len(model.full_case_list)
    n_folds = len(cv_results.fold_id.unique())

    fig, ax = plt.subplots(
        figsize=(12, 10),
        nrows=n_folds,
        ncols=n_cases,
        sharex=True,
        sharey=True
    )

    folds_results = cv_results[cv_results.n_params == n_params]

    for k, fold_id in enumerate(cv_results.fold_id.unique()):
        fold_id = int(fold_id)
        fold_results = folds_results[folds_results.fold_id == fold_id]
        voltage_cases, hfr_cases, state_cases = model.simulate_for_fold_results(fold_results)

        for i, case in enumerate(model.full_case_list):
            y_sim = voltage_cases[case] if variable == 'voltage' else hfr_cases[case] * model.hfr_mask[case]
            x_sim = state_cases[case].current_density
            case_dataset = model.get_case_dataset(case)
            y_exp = case_dataset[variable]
            x_exp = case_dataset['current-density']


            ax[k, i].plot(x_sim, y_sim, f'-C{i}')

            if uncertainty:  
                ax[k, i].fill_between(x_sim, (1-uncertainty)*y_sim, 
                    (1+uncertainty)*y_sim, color=f'C{i}', alpha=0.3)

            ax[k, i].plot(x_exp, y_exp, f'sC{i}', markersize=3.5,  alpha=0.5)
    
            if case in model.k_folds[fold_id]:
                ax[k, i].set_facecolor('#f0f0f0')

            # Column titles
            if k == 0:
                ax[0, i].set_title(f'Case {case}',
                    fontsize=9
                )

            # Row labels
            if i == 0:
                ax[k, 0].set_ylabel(
                    f'Fold {k+1}\n'
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

    return fig, ax
