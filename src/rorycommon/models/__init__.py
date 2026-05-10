from dataclasses import dataclass, field
from typing import Tuple,Dict,Optional
import numpy.typing as npt

@dataclass
class PutPlaintextResult:
    path: Optional[str]
    extension: str
    bucket_id: str
    ball_id: str
    tags: Dict[str, str]
    shape: Tuple[int,int]
    dtype: npt.DTypeLike
    read_time: float
    segment_time: float
    upload_time: float

from enum import Enum
class SourceType(Enum):
    FILE = "file"
    URL = "url"
    DATABASE = "database"
    IN_MEMORY = "in_memory"
    CLOUD = "cloud"
    OTHER = "other"

from typing import Generic,TypeVar,List
from Pyfhel import PyCtxt
TList = TypeVar('TList', List[bytes], List[PyCtxt], List[float], List[int], npt.NDArray)
@dataclass
class GetResult(Generic[TList]):
    source: Optional[SourceType] = field(default=None)
    raw_value: Optional[TList] = field(default=None)
    read_time: Optional[float] = field(default=None)

    def to_nd_array(self) -> Optional[npt.NDArray]:
        if self.raw_value is not None and isinstance(self.raw_value, npt.NDArray):
            return self.raw_value
        return None
    def to_list(self) -> Optional[List]:
        if self.raw_value is not None and isinstance(self.raw_value, list):
            return self.raw_value
        return None
    def to_pyctxt_list(self) -> Optional[List[PyCtxt]]:
        if self.raw_value is not None and isinstance(self.raw_value, list) and all(isinstance(item, PyCtxt) for item in self.raw_value):
            return self.raw_value
        return None
    def to_bytes_list(self) -> Optional[List[bytes]]:
        if self.raw_value is not None and isinstance(self.raw_value, list) and all(isinstance(item, bytes) for item in self.raw_value):
            return self.raw_value
        return None
    def to_float_list(self) -> Optional[List[float]]:
        if self.raw_value is not None and isinstance(self.raw_value, list) and all(isinstance(item, float) for item in self.raw_value):
            return self.raw_value
        return None
    def to_int_list(self) -> Optional[List[int]]:
        if self.raw_value is not None and isinstance(self.raw_value, list) and all(isinstance(item, int) for item in self.raw_value):
            return self.raw_value
        return None



@dataclass
class PutCiphertextResult:
    path: Optional[str]
    extension: str
    bucket_id: str
    ball_id: str
    tags: Dict[str, str]
    shape: Tuple[int,int]
    dtype: npt.DTypeLike
    read_time: float
    segment_time: float
    encrypt_time: float
    upload_time: float
    