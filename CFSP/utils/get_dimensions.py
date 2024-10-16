from typing import List

def get_dimensions(mylist):
    if not isinstance(mylist, List):
        return 0
    if not mylist:
        return 1
    return 1 + max(get_dimensions(item) for item in mylist)
