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