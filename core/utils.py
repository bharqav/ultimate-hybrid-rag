def safe_import(name, fromlist=None):
    try:
        module = __import__(name, fromlist=fromlist) if fromlist else __import__(name)
        return module
    except ImportError:
        return None
