import seaborn as sns


def set_sns_defaults():
    """Set seaborn plots defaults"""
    sns.set(rc={'figure.figsize': (10, 10)})
    sns.set_style('whitegrid')
    sns.set_context('talk')


def set_sns_size(width, height):
    """Set seaborn plot slize"""
    sns.set(rc={'figure.figsize': (width, height)})

