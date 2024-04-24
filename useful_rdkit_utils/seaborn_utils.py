
import seaborn as sns

def set_sns_defaults()  -> None:
    """Set seaborn plots defaults"""
    sns.set(rc={'figure.figsize': (10, 10)})
    sns.set_style('whitegrid')
    sns.set_context('talk')


def set_sns_size(width, height) -> None:
    """Set seaborn plot slize

    :param width: width of the plot
    :param height: height of the plot
    :return: None
    """
    sns.set(rc={'figure.figsize': (width, height)})

