from tiase import alfred

_usage_str = """
Options:
    [xmlfile]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    xmlfile = "alfred_config.xml"
    if len(sys.argv) > 1:
        xmlfile = sys.argv[1]
    else:
        _usage()
    alfred.execute(xmlfile)
