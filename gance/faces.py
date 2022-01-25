"""
Common face recognition/detection related functionality.
Wrapped in a way that is compatible with the rest of this project.
"""

from types import ModuleType
from typing import Dict, List, Optional, Tuple, cast

from gance.gance_types import LabeledCoordinates


class FaceFinderProxy:
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
            import face_recognition  # pylint: disable=import-outside-toplevel

            self._face_recognition = face_recognition
            self._imported = True

    def face_locations(  # type: ignore[no-untyped-def]
        self: "FaceFinderProxy", *args, **kwargs
    ) -> Tuple[LabeledCoordinates]:
        """
        Call face_recognition.face_locations through the proxy.
        :param args: Forwarded to lib function.
        :param kwargs: Forwarded to lib function.
        :return: Library function return cast as a tuple.
        """

        self._import_just_in_time()
        return cast(
            Tuple[LabeledCoordinates],
            tuple(
                self._face_recognition.face_locations(*args, **kwargs)  # type: ignore[union-attr]
            ),
        )

    def face_landmarks(  # type: ignore[no-untyped-def]
        self: "FaceFinderProxy", *args, **kwargs
    ) -> List[Dict[str, Tuple[int, ...]]]:
        """
        Call face_recognition.face_landmarks through the proxy.
        :param args: Forwarded to lib function.
        :param kwargs: Forwarded to lib function.
        :return: Returned directly from the library function.
        """

        self._import_just_in_time()
        return cast(
            List[Dict[str, Tuple[int, ...]]],
            self._face_recognition.face_landmarks(*args, **kwargs),  # type: ignore[union-attr]
        )
