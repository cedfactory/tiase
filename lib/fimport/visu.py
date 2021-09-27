import matplotlib.pyplot as plt
from scipy.stats import norm

def display_from_dataframe(df, name, export_name=""):
    plt.figure(figsize=(15, 5))
    plt.plot(df[name])
    #plt.xticks(range(0, df.shape[0], 2000), df['Date'].loc[::2000], rotation=0)
    plt.ylabel(name, fontsize=18)
    plt.title(name, fontsize=20)
    plt.legend([name], fontsize='x-large', loc='best')
    if export_name == "":
        plt.show()
    else:
        plt.savefig(export_name)
    plt.close()

def display_histogram_from_dataframe(df, name, bins="auto", export_name=""):
    hist = df.hist(column=name, bins=bins)
    fig = hist[0][0].get_figure()
    fig.savefig(export_name)

def display_histogram_fitted_gaussian(data, bins="auto", export_name=""):
    (mu, sigma) = norm.fit(data)

    n, bins, patches = plt.hist(data, bins, density=True, alpha=0.75)

    x = bins
    y = norm.pdf(bins, mu, sigma)
    plt.plot(x, y, 'r--', linewidth=2)
    plt.grid(True)
    plt.title(r'fitted gaussian: $\mu={:2f}$, $\sigma={:3f}$'.format(mu, sigma))

    if export_name == "":
        plt.show()
    else:
        plt.savefig(export_name)
    plt.close()

def display_outliers_from_dataframe(df, d1, export_name=""):
    df['simple_rtn'] = df.close.pct_change()
    d1['simple_rtn'] = d1.close.pct_change()

    plt.clf()

    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df.index, df.simple_rtn, color='red', label='Normal')
    ax.plot(d1.index, d1.simple_rtn, color='blue', label='Anomaly_removed')
    #ax.set_title(str(len(df)))
    ax.legend(loc='lower right')

    fig.savefig(export_name)
    plt.close()