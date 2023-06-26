import numpy as np
from sklearn.neighbors import DistanceMetric


def great_circle_distance(latlong_1, latlong_2, earth_radius=3959):
    '''
    Estimated shortest distance between two points on a perfect sphere. Here, we assume earth is a perfect
    sphere.

    Parameters
    ----------
    latlong_1 : tuple
        Latitude and Longitude (1)
    latlong_2 : tuple
        Latitude and Longitude (2)
    earth_radius : float
        Earth's radius. You choose units, default is miles.

    Returns
    -------
    (float) Distance between two coordinates in the units of `earth_radius`.
    '''

    # Extract latitude and longitude
    lat1 = latlong_1[0]
    lon1 = latlong_1[1]

    lat2 = latlong_2[0]
    lon2 = latlong_2[1]

    pi = np.math.pi

    # convert degrees to radians
    phi1 = lat1 * pi / 180
    phi2 = lat2 * pi / 180

    d_phi = (lat2 - lat1) * pi / 180
    d_lambda = (lon2 - lon1) * pi / 180

    # Use haversine formula as defined at https://www.movable-type.co.uk/scripts/latlong.html
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2

    # Convert haversine to distance
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = earth_radius * c  # earth_radius is in miles by default

    return d


def latlongs_to_dist_graph(X_latlong, radius=3959):
    '''
    Given matrix where columns are [latitude, longitude], return an n x n matrix A where A[i, j] is the
    distance between points X[i] and X[j] in the units shared by `radius`.

    Parameters
    ----------
    X_latlong : np.array

    radius : float

    Returns
    -------
    np.array
    '''
    X = np.array(X_latlong) * np.math.pi / 180
    haversine = DistanceMetric.get_metric('haversine')
    distances = haversine.pairwise(X)
    distances *= radius

    return distances
