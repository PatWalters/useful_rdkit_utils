

def set_sns_defaults():
    """Set seaborn plots defaults"""
    import seaborn as sns
    sns.set(rc={'figure.figsize': (10, 10)})
    sns.set_style('whitegrid')
    sns.set_context('talk')


def set_sns_size(width, height):
    """Set seaborn plot slize"""
    import seaborn as sns
    sns.set(rc={'figure.figsize': (width, height)})

