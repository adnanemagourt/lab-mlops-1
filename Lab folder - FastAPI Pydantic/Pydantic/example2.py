from typing import List
from pydantic import BaseModel


class Foo(BaseModel):
    count: int
    size: float = 0.0


class Bar(BaseModel):
    apple: str = "x"
    banana: str = "y"


class Spam(BaseModel):
    foo: Foo
    bars: List[Bar]


m = Spam(foo={"count": 4}, bars=[{"apple": "x1"}, {"apple": "x2"}])

print(m)
print(m.dict())
