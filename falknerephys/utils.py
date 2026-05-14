import numpy as np

def rotate_xy(x, y, rot):
    """
    Rotates coordinates by a given angle.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.
    rot : float or str
        Rotation angle in degrees or a string key for predefined angles.

    Returns
    -------
    tuple
        Rotated X and Y coordinates.
    """
    rot_dict = {'rni': 0,
                'none': 0,
                'irn': 120,
                'nir': 240,
                'RNI': 0,
                'IRN': 120,
                'NIR': 240}
    if type(rot) == str:
        rot_deg = rot_dict[rot]
    else:
        rot_deg = rot
    in_rad = np.radians(rot_deg)
    c, s = np.cos(in_rad), np.sin(in_rad)
    rot_mat = [[c, -s], [s, c]]
    xy = rot_mat @ np.vstack((x, y))
    return xy[0, :], xy[1, :]


def xy2theta(x_pos, y_pos, rot=0):
    r = np.sqrt(x_pos**2 + y_pos**2)
    rot_x, rot_y = rotate_xy(x_pos, y_pos, rot)
    theta = np.arctan2(rot_y, rot_x)
    return theta, r


