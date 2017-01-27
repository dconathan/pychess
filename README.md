Usage:
    pychess.py process -i <in> -o <out>
    pychess.py analyze -i <in> -o <out> [--catmap <cat_file> --catthreshold <cat_thresh>]

Arguments:

    <in> (required) : path to input file or directory.  in can be one of:
        (a) directory of .csv or .xlsx files like a seva or eldertree export,
        (b) .yaml file pointing to urls where to load data (like for bundling), or
        (c) a .json or .pickle file containing the serialized study objects
            (e.g. for loading the output of process into analyze)
            
    <out> (required) : where to write the output json or pickle file
        Must have a `.json` or `.pickle` (or `.pkl`) extension. If `.json`, data will be serialized in a
        quasi-human-readable format that will be easier to load into other applications/languages/do whatever you want.
        If `.pickle`, data will be serialized using pickle, which will preserve all the data types and structures.
        
    <catmap> (optional) : a .yaml file that maps "action" categories to another category.
        For getting activities to correspond across studies (e.g. seva and bundling).
        See seva_to_bundling.yaml or reduce_bundling.yaml for examples.
        If no catmap supplied, categories are left as is.
        
    <cat_thresh> (optional) : a .yaml file that maps "action" categories to its "threshold".
        That is, how many times must this action be done in one period in order for that period to count as
        "active" for that category for that period.  If no catmap is supplied, thresholds will be automatically tuned
        to achieve a target inactivity rate. See Analysis options in this code for more details.

Example:
    > python pychess.py process -i seva_export/ -o seva_study.json
    > python pychess.py analyze -i seva_study.json -o seva_analysis.json