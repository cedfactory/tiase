import matplotlib.pyplot as plt

def DisplayFromDataframe(df, name, exportName=""):
    plt.figure(figsize=(15, 5))
    plt.plot(df[name])
    #plt.xticks(range(0, df.shape[0], 2000), df['Date'].loc[::2000], rotation=0)
    plt.ylabel(name, fontsize=18)
    plt.title(name, fontsize=20)
    plt.legend([name], fontsize='x-large', loc='best')
    if exportName == "":
        plt.show()
    else:
        plt.savefig(exportName)

def DisplayHistogramFromDataframe(df, name, exportName=""):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    axes.hist(df[name])

    if exportName == "":
        plt.show()
    else:
        fig.savefig(exportName)
    plt.clf()

def DisplayOutliersFromDataframe(df, d1, exportName=""):
    df['simple_rtn'] = df.close.pct_change()
    d1['simple_rtn'] = d1.close.pct_change()

    plt.clf()

    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df.index, df.simple_rtn, color='red', label='Normal')
    ax.plot(d1.index, d1.simple_rtn, color='blue', label='Anomaly_removed')
    #ax.set_title(str(len(df)))
    ax.legend(loc='lower right')

    fig.savefig(exportName)
    plt.clf()