from typing import List, Optional
from pydantic import BaseModel, Field
from .frame import POS, FrameResult, FrameWithScore, FeResult, FeWithScore



class Span(BaseModel):
    start: int
    end: int


class Word(BaseModel):
    """
    分词，词汇
    """
    word: str
    pos: POS


class PylemmaTag(Span, BaseModel):
    """
    某个框架元素的标注结果
    """
    fe: Optional[FeResult] = Field(None) #分数最高那一个
    fe_with_score: Optional[List[FeWithScore]] = Field(None)


class Target(Word, BaseModel):
    """
    目标词，额外包含框架信息
    """
    start: int
    end: int
    frame: Optional[FrameResult] = Field(None)
    frame_with_score: Optional[List[FrameWithScore]] = Field(None)


class TargetParsing(BaseModel):
    """ 
        单个 目标词解析，包含目标词，角色标注结果，框架，
    """
    target: Optional[Target] = Field(None)
    arguments: Optional[List[PylemmaTag]] = Field(None)
    ...


class Sentence(BaseModel):
    """
    句子，包含分词结果，句子内容，目标词，等等
    """
    text: str
    words: List[Word] = Field(None)


class SentenceResult(BaseModel):

    sentence: Sentence
    parsing: Optional[List[TargetParsing]] = Field(None) #Field([]) # 多个目标词

    def __str__(self):
        return str(self.model_dump_json())  