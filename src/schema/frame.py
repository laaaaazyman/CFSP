from typing import List, Optional, Union
from pydantic import BaseModel, Field, PrivateAttr

class Fe(BaseModel):
    fe_id: int
    fe_name: str
    fe_ename: str
    fe_abbr: str
    fe_def: str

#模型三用的结构，不需要那么多key
class FeResult(BaseModel):
    fe_id: int
    fe_name: str
    fe_abbr: str = Field(None)

#带分数的前十个可能角色标注
class FeWithScore(BaseModel):
    fe_id: int
    fe_name: str
    fe_abbr: str = Field(None)
    score: float

#查询时用的结构
class Frame(BaseModel):
    frame_id: int
    rame_element: List[Fe]
    frame_name: str
    frame_def: str
    frame_enname: str

#模型一输出的结构
class FrameResult(BaseModel):
    frame_id: int
    frame_name: str

#带分数的前10个框架
class FrameWithScore(BaseModel):
    frame_id: int
    frame_name: str
    score: float


class Lexical(BaseModel):
    lexical_id: int
    lexical: str
    POS_id: int
    frame_id: int


class FrameInfo(Frame, BaseModel):
    fes: List[Fe]
    lexicals: List[Lexical]


class POS(BaseModel):
    POS_id: int
    POS_name: str
    POS_mark: str

