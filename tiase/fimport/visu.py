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

def display_outliers_from_dataframe(df_original, df_result, feature, export_name=""):
    plt.clf()
    df_original.to_csv("./tmp/__original.csv")
    df_result.to_csv("./tmp/__post.csv")
      
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df_original.index, df_original[feature], color='red', label='Original')
    ax.plot(df_result.index, df_result[feature], color='blue', label='Outliers removed')
    #ax.set_title(str(len(df)))
    ax.legend(loc='lower right')

    fig.savefig(export_name)
    plt.close()