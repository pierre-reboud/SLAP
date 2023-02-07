import numpy as np
from typing import List, Dict, Any, Union, Type
from dataclasses import dataclass

@dataclass
class Point:

    # Global count of points in map
    count : int = 0

    def __init__(self, coordinates : np.ndarray, id : Union[int, None] = None):
        # Coordinates of the homogeneous point in 3D (4d array) 
        self._cartesian_coordinates : np.ndarray = self._get_coordinates(coordinates)
        # Point id in the global 3d map
        self._id : Union[int, None] = Point.count
        # Increment classe's count
        Point.count += 1
    
    def _get_coordinates(self, coordinates : Union[np.ndarray, List]):
        if isinstance(coordinates, List):
            coordinates = np.array(coordinates)
        if len(coordinates) == 3:
            return coordinates
        elif len(coordinates) == 4 and coordinates[3] == 1:
            return coordinates[:3]
        else:
            raise Exception("Passed erroneous point cordinates, must be cartesian or homogeneous")

    @property
    def cartesian(self) -> np.ndarray:
        return self._cartesian_coordinates

    @property
    def homogeneous(self) -> np.ndarray:
        return np.concatenate((self._cartesian_coordinates,[1]),axis = 0)

    @property
    def id(self) -> int:
        return self._id
    
    def __repr__(self) -> str:
        return f"pt{self.id}{self._cartesian_coordinates}"
    
    def __add__(self, summand):# : Union[Point, np.ndarray]) -> Point:
        if isinstance(summand, Point):
            return Point(coordinates = self.cartesian + summand.cartesian)
        elif isinstance(summand, np.ndarray):
            return Point(coordinates = self.cartesian + summand)
    
    def __matmul__(self, matrix):# : Union[Point, np.ndarray]) -> Point:
        if isinstance(matrix, Point):
            return Point(coordinates = self.cartesian.T@matrix.cartesian)
        elif isinstance(matrix, np.ndarray):
            return Point(coordinates = matrix@self.homogeneous)
    
    def __hash__(self) -> int:
        return hash(self.id)
        