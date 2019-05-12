
# got from https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def python_shell() -> str:
    """
    Determine python shell
    See also https://stackoverflow.com/a/37661854

    :return:
            'shell' (started python on command line using "python") <br/>
            'ipython' (started ipython on command line using "ipython")<br/>
            'ipython-notebook' (e.g., running in Spyder or started with "ipython qtconsole")<br/>
            'jupyter-notebook' (running in a Jupyter notebook)

    """
    import os
    env = os.environ
    shell = 'shell'
    program = os.path.basename(env['_'])

    if 'jupyter-notebook' in program:
        shell = 'jupyter-notebook'
    elif 'JPY_PARENT_PID' in env or 'ipython' in program:
        shell = 'ipython'
        if 'JPY_PARENT_PID' in env:
            shell = 'ipython-notebook'

    return shell

