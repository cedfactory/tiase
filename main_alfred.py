from tiase import alfred

_usage_str = """
Options:
    [--execute xmlfile, --summary]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        if len(sys.argv) == 2 and (sys.argv[1] == "--summary" or sys.argv[1] == "-s"):
            alfred.summary()
        elif len(sys.argv) == 3 and (sys.argv[1] == "--execute" or sys.argv[1] == "-e"):
            xmlfile = sys.argv[2]
            alfred.execute(xmlfile)
        elif len(sys.argv) == 3 and (sys.argv[1] == "--details" or sys.argv[1] == "-d"):
            value = sys.argv[2]
            alfred.details_for_value(value)
        else:
            _usage()
    else:
        _usage()
    
