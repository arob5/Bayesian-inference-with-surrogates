import seaborn as sns

def set_plot_theme():
    sns.set_theme(style='white', palette='colorblind')
    sns.set_context("paper", font_scale=1.5)

    # Specific Paul Tol color scheme when comparing different posteriors
    colors = {
        'exact': "#4477AA",
        'mean': "#EE6677",
        'eup': "#228833",
        'ep': "#CCBB44",
        'aux': "#888888"
    }

    return colors