"""
Common face recognition/detection related functionality.
Wrapped in a way that is compatible with the rest of this project.
"""

from typing import Dict, List, Tuple, Optional
from types import ModuleType

from gance.gance_types import LabeledCoordinates


class FaceFinderProxy(object):
    """
    Under the hood, `face_recognition` uses dnnlib, which stylegan also uses.
    An init function is called upon import, which makes the loading of models within the
    subprocesses impossible. There's probably a better way around this but this was expedient.
    """

    def __init__(self: "FaceFinderProxy") -> None:
        self._imported: bool = False
        self._face_recognition: Optional[ModuleType] = None

    def _import_just_in_time(self: "FaceFinderProxy") -> None:
        """
        The library functions are about to be used, try and actually
        do the import. If the lib has already been imported, so nothing.
        :return: None
        """

        if not self._imported:
            import face_recognition

            self._face_recognition = face_recognition
            self._imported = True

    def face_locations(self: "FaceFinderProxy", *args, **kwargs) -> Tuple[LabeledCoordinates]:
        """
        Call face_recognition.face_locations through the proxy.
        :param args: Forwarded to lib function.
        :param kwargs: Forwarded to lib function.
        :return: Library function return cast as a tuple.
        """

        self._import_just_in_time()
        return tuple(self._face_recognition.face_locations(*args, **kwargs))

    def face_landmarks(
        self: "FaceFinderProxy", *args, **kwargs
    ) -> List[Dict[str, Tuple[int, ...]]]:
        """
        Call face_recognition.face_landmarks through the proxy.
        :param args: Forwarded to lib function.
        :param kwargs: Forwarded to lib function.
        :return: Returned directly from the library function.
        """

        self._import_just_in_time()
        return self._face_recognition.face_landmarks(*args, **kwargs)
