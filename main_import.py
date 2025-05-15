from tiase.fimport import fimport,synthetic,visu

###
### AS A SCRIPT
### python -m import.py
###

_usage_str = """
Options:
    [ --cac40, --nasdaq100]
"""

def _usage():
    print(_usage_str)

def _test1():
    df = fimport.get_dataframe_from_yahoo('AI.PA')
    print(df)
    print()

def _test2():
    hist = fimport.get_dataframe_from_csv('./tiase/data/CAC40/AI.PA.csv')
    visu.display_from_dataframe(hist, "Close", "AI.PA.close.png")

def _test3():
    y = synthetic.get_sinusoid(length=100, amplitude=1, frequency=.1, phi=0, height = 0)
    df = synthetic.create_dataframe(y, .1)
    visu.display_from_dataframe(df,"Close", "close.png")


def _download(values):
    if values == "cac40":
        fimport.download_from_yahoo(fimport.cac40.keys())
    elif values == "nasdaq100":
        fimport.download_from_yahoo(fimport.nasdaq100.keys())

    
if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 1:
        if sys.argv[1] == "--test1" or sys.argv[1] == "-t1":
            _test1()
        elif sys.argv[1] == "--test2" or sys.argv[1] == "-t2":
            _test2()
        elif sys.argv[1] == "--test3" or sys.argv[1] == "-t3":
            _test3()
        elif sys.argv[1] == "--cac40":
            _download('cac40')
        elif sys.argv[1] == "--nasdaq100":
            _download('nasdaq100')
        else:
            _usage()
    else: _usage()
